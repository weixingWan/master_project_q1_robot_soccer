#!/usr/bin/env python3
"""Headless EGL offscreen 720p@50fps MP4 video exporter for both Q1 courses.

Works inside rootless Ubuntu containers without GLFW / X11 display as long as
`libEGL` and `imageio-ffmpeg` are installed (verified available in `env_q1`).

Examples:
  # Render pole-weaving nominal-01 & eval-01 videos
  python scripts/render_videos.py --task pole \
    --checkpoint logs/q1-pole-weaving/run-20261126-210333/best.pt \
    --case nominal-01 --out videos/pole_nominal_best.mp4

  # Render pole-weaving under disturbance (push + sensor/action noise)
  python scripts/render_videos.py --task pole \
    --checkpoint logs/q1-pole-weaving/run-20261126-210333/best.pt \
    --case nominal-01 --randomize --observation-noise --action-noise 0.03 \
    --out videos/pole_nominal_disturbed.mp4

  # Render football-kicking episodes (with camera FOV frustum)
  python scripts/render_videos.py --task soccer \
    --checkpoint logs/q1-robot-supermimic-20261126-211035/best.pt \
    --steps 2500 --out-dir videos/soccer_baseline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")


def render_pole(args: argparse.Namespace) -> None:
    import imageio.v2 as imageio
    import mujoco

    from pole_task import COURSE_ROOT
    from pole_task.inference import load_pole_actor
    from pole_task.source_environment import SourcePoleEnvironment
    from pole_task.source_playback import load_course_bundle
    from pole_task.source_runtime import run_source_episode

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    actor, _ = load_pole_actor(args.checkpoint, args.device)
    with tempfile.TemporaryDirectory(prefix="q1-pole-video-") as tmp:
        tmp_dir = Path(tmp)
        robot, motions, cases, _ = load_course_bundle(
            COURSE_ROOT / "pole_task/config.json",
            tmp_dir / "references",
            device=args.device,
        )
        case = cases[args.case]
        clip = next(i for i, c in enumerate(motions.cases) if c.case_id == args.case)
        scene_path = tmp_dir / "pole-video.xml"
        env = SourcePoleEnvironment(
            case, scene_path, robot, seed=args.seed, randomize=args.randomize
        )
        episode = run_source_episode(
            actor,
            robot,
            motions,
            case,
            clip,
            scene_path,
            start_frame=0,
            seed=args.seed,
            device=args.device,
            randomize=args.randomize,
            observation_noise=args.observation_noise,
            action_noise=args.action_noise,
            environment=env,
        )

        env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, args.width)
        env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        camera.azimuth = 135.0
        camera.elevation = -22.0
        camera.distance = 4.2

        nq = env.model.nq
        writer = imageio.get_writer(
            str(out_path),
            fps=50,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=2,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )
        try:
            for state in episode.states:
                env.data.qpos[:] = state[:nq]
                mujoco.mj_forward(env.model, env.data)
                camera.lookat[:] = env.data.qpos[:3]
                renderer.update_scene(env.data, camera=camera)
                writer.append_data(renderer.render())
        finally:
            writer.close()
            renderer.close()

    res = episode.result
    print(
        f"[render_videos] Saved {out_path} ({len(episode.states)} frames) | "
        f"case={args.case} success={res['success']} rmse={res['root_xy_rmse_m']:.4f}m "
        f"end={res['endpoint_distance_m']:.4f}m pole_hit={res['pole_collision']} fall={res['fall']}",
        flush=True,
    )


def render_soccer(args: argparse.Namespace) -> None:
    from q1_soccer.play import StudentPlayback

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    playback = StudentPlayback(args.checkpoint, seed=args.seed)
    playback.run(headless=True, steps=args.steps, out_video=out_dir, show_frustum=True)
    print(f"[render_videos] Saved soccer episode videos under: {out_dir}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("pole", "soccer"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    # Pole-specific options
    parser.add_argument("--case", choices=("nominal-01", "eval-01", "straight", "left_turn", "right_turn"), default="nominal-01")
    parser.add_argument("--randomize", action="store_true", help="Enable domain randomization & random pushes")
    parser.add_argument("--observation-noise", action="store_true", help="Enable sensor observation noise")
    parser.add_argument("--action-noise", type=float, default=0.0, help="Gaussian action std")
    parser.add_argument("--out", type=Path, default=Path("videos/pole_playback.mp4"))

    # Soccer-specific options
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--out-dir", type=Path, default=Path("videos/soccer"))

    args = parser.parse_args(argv)
    if args.task == "pole":
        render_pole(args)
    else:
        render_soccer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Robustness stress-test suite for project-trained pole-weaving checkpoints.

Evaluates one or more pole checkpoints across four disturbance tiers using
``run_source_episode`` inside a temporary directory so no ``.xml`` or ``.npz``
artifacts are written into ``experiments/``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
POLE_COURSE_ROOT = REPO_ROOT / "courses/pole-weaving"
if str(POLE_COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(POLE_COURSE_ROOT))

from pole_task import COURSE_ROOT  # noqa: E402
from pole_task.acceptance import nominal_trial_passes  # noqa: E402
from pole_task.inference import load_pole_actor  # noqa: E402
from pole_task.source_playback import load_course_bundle  # noqa: E402
from pole_task.source_runtime import run_source_episode  # noqa: E402


@dataclass(frozen=True)
class StressTier:
    """Single disturbance configuration for headless pole rollout stress testing."""

    name: str
    randomize: bool
    observation_noise: bool
    action_noise: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "randomize": self.randomize,
            "observation_noise": self.observation_noise,
            "action_noise": self.action_noise,
        }


STRESS_TIERS: tuple[StressTier, ...] = (
    StressTier("tier0_clean", randomize=False, observation_noise=False, action_noise=0.0),
    StressTier(
        "tier1_sensor_action_noise",
        randomize=False,
        observation_noise=True,
        action_noise=0.04,
    ),
    StressTier(
        "tier2_domain_rand_push",
        randomize=True,
        observation_noise=True,
        action_noise=0.02,
    ),
    StressTier(
        "tier3_extreme_stress",
        randomize=True,
        observation_noise=True,
        action_noise=0.06,
    ),
)


def parse_checkpoint_specs(items: list[str]) -> list[tuple[str, Path]]:
    """Parse ``label=path`` checkpoint arguments and reject duplicate labels."""
    specs: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for item in items:
        if "=" in item:
            label, raw_path = item.split("=", 1)
            label = label.strip()
            raw_path = raw_path.strip()
        else:
            raw_path = item.strip()
            label = Path(raw_path).stem
        if not label or not raw_path:
            raise ValueError(f"Invalid checkpoint specification: {item!r}")
        if label in seen_labels:
            raise ValueError(f"Duplicate checkpoint label: {label!r}")
        seen_labels.add(label)
        specs.append((label, Path(raw_path).expanduser().resolve()))
    return specs


def summarize_tier_trials(rows: list[dict]) -> dict:
    """Compute aggregate pass/failure rates and tracking error statistics."""
    if not rows:
        raise ValueError("Cannot summarize an empty trial list")
    rmses = np.asarray([float(row["root_xy_rmse_m"]) for row in rows], dtype=np.float64)
    endpoints = np.asarray(
        [float(row["endpoint_distance_m"]) for row in rows], dtype=np.float64
    )
    steps = np.asarray([int(row["steps"]) for row in rows], dtype=np.float64)
    passed_flags = [bool(row["passed"]) for row in rows]
    return {
        "trials": len(rows),
        "passed_count": int(sum(passed_flags)),
        "pass_rate": float(np.mean(passed_flags)),
        "route_completed_rate": float(
            np.mean([bool(row["route_completed"]) for row in rows])
        ),
        "fall_rate": float(np.mean([bool(row["fall"]) for row in rows])),
        "pole_collision_rate": float(
            np.mean([bool(row["pole_collision"]) for row in rows])
        ),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "endpoint_mean": float(np.mean(endpoints)),
        "endpoint_std": float(np.std(endpoints)),
        "steps_mean": float(np.mean(steps)),
    }


def evaluate_tier(
    actor,
    robot,
    motions,
    case,
    clip: int,
    scene_path: Path,
    *,
    config: dict,
    tier: StressTier,
    trials: int,
    base_seed: int,
    device: str,
) -> dict:
    """Run ``trials`` episodes for one (checkpoint, case, tier) slice."""
    trial_rows: list[dict] = []
    for trial in range(trials):
        trial_seed = base_seed + trial
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(trial_seed)
            episode = run_source_episode(
                actor,
                robot,
                motions,
                case,
                clip,
                scene_path,
                start_frame=0,
                seed=trial_seed,
                device=device,
                randomize=tier.randomize,
                observation_noise=tier.observation_noise,
                action_noise=tier.action_noise,
            )
        row = {
            **episode.result,
            "trial": trial,
            "seed": trial_seed,
        }
        row["passed"] = nominal_trial_passes(row, config)
        trial_rows.append(row)
    summary = summarize_tier_trials(trial_rows)
    summary["trial_results"] = trial_rows
    return summary


def format_markdown_table(report: dict) -> str:
    """Render a clean Markdown comparison table grouped by case and tier."""
    lines = [
        "# Pole-Weaving Stage 2 Robustness Stress Report",
        "",
        f"- **Seed**: `{report['seed']}`",
        f"- **Trials per (case, tier)**: `{report['trials']}`",
        f"- **Device**: `{report['device']}`",
        f"- **Cases**: {', '.join(f'`{c}`' for c in report['cases'])}",
        "",
        "## Checkpoints",
        "",
    ]
    for entry in report["checkpoints"]:
        lines.append(f"- **{entry['label']}**: `{entry['path']}`")

    lines.extend(
        [
            "",
            "## Stress Comparison Table",
            "",
            "| Case | Tier | Checkpoint | Pass Rate | Route Done | Fall Rate | Collision Rate | RMSE (m) | Endpoint Dist (m) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for case_id in report["cases"]:
        for tier in report["tiers"]:
            tier_name = tier["name"]
            for entry in report["checkpoints"]:
                stats = entry["results"][case_id][tier_name]
                pass_pct = 100.0 * stats["pass_rate"]
                route_pct = 100.0 * stats["route_completed_rate"]
                fall_pct = 100.0 * stats["fall_rate"]
                coll_pct = 100.0 * stats["pole_collision_rate"]
                rmse_str = f"{stats['rmse_mean']:.4f} ± {stats['rmse_std']:.4f}"
                end_str = f"{stats['endpoint_mean']:.4f} ± {stats['endpoint_std']:.4f}"
                lines.append(
                    f"| `{case_id}` | `{tier_name}` | `{entry['label']}` | "
                    f"{pass_pct:.1f}% ({stats['passed_count']}/{stats['trials']}) | "
                    f"{route_pct:.1f}% | {fall_pct:.1f}% | {coll_pct:.1f}% | "
                    f"{rmse_str} | {end_str} |"
                )

    lines.append("")
    return "\n".join(lines)


def run_stress_suite(
    checkpoint_specs: list[tuple[str, Path]],
    *,
    trials: int,
    seed: int,
    device: str,
    case_ids: list[str],
    config_path: Path,
) -> dict:
    """Evaluate all checkpoints across every requested case and stress tier."""
    if trials <= 0:
        raise ValueError(f"--trials must be positive, got {trials}")

    checkpoints_report: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="q1-pole-stress-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        robot, motions, cases, config = load_course_bundle(
            config_path,
            tmp_path / "references",
            device=device,
        )
        clip_by_case = {
            case.case_id: idx for idx, case in enumerate(motions.cases)
        }
        missing = [
            case_id
            for case_id in case_ids
            if case_id not in cases or case_id not in clip_by_case
        ]
        if missing:
            raise ValueError(
                f"Unknown evaluation cases {missing}; available: {sorted(cases)}"
            )

        for label, ckpt_path in checkpoint_specs:
            actor, identity = load_pole_actor(ckpt_path, device)
            actor.eval()
            case_results: dict[str, dict] = {}
            for case_id in case_ids:
                case = cases[case_id]
                clip = clip_by_case[case_id]
                scene_path = tmp_path / f"stress-{case_id}.xml"
                tier_results: dict[str, dict] = {}
                for tier in STRESS_TIERS:
                    tier_results[tier.name] = evaluate_tier(
                        actor,
                        robot,
                        motions,
                        case,
                        clip,
                        scene_path,
                        config=config,
                        tier=tier,
                        trials=trials,
                        base_seed=seed,
                        device=device,
                    )
                case_results[case_id] = tier_results
            checkpoints_report.append(
                {
                    "label": label,
                    "path": str(ckpt_path),
                    "identity": identity,
                    "results": case_results,
                }
            )

    return {
        "seed": seed,
        "trials": trials,
        "device": device,
        "cases": list(case_ids),
        "tiers": [tier.to_dict() for tier in STRESS_TIERS],
        "checkpoints": checkpoints_report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more label=path checkpoint pairs",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["nominal-01", "eval-01"],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=COURSE_ROOT / "pole_task/config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/stage2_pole_stress/stress_report.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("experiments/stage2_pole_stress/stress_table.md"),
    )
    args = parser.parse_args(argv)

    checkpoint_specs = parse_checkpoint_specs(args.checkpoints)
    report = run_stress_suite(
        checkpoint_specs,
        trials=args.trials,
        seed=args.seed,
        device=args.device,
        case_ids=args.cases,
        config_path=args.config,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown_path = Path(args.markdown).expanduser().resolve()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_text = format_markdown_table(report)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    print(markdown_text)
    print(f"Saved JSON report to: {output_path}")
    print(f"Saved Markdown table to: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Heterogeneous CPU (Intel 12th/14th-gen P-core / E-core) affinity launcher.

Why this exists:
On hybrid CPUs like i7-12700 (8P+4E = 20 threads) and i7-14700 (8P+12E = 28
threads), single-threaded MuJoCo rollouts and small-MLP PyTorch forwards suffer
badly when the Linux scheduler migrates tasks onto E-cores or spawns 20-37
OpenMP threads across asymmetric cores.

This rootless launcher:
1. Detects true P-cores vs E-cores from Linux sysfs topology & max frequency
   (intersected with the current container/cgroup allowed CPU mask);
2. Pins the child process to a clean slice of P-cores (`--slot` / `--total-slots`)
   for either single-job speed or multi-experiment packing on 3090/4090;
3. Caps OMP / MKL / OPENBLAS / TORCH thread counts to eliminate spinlock waste;
4. Optionally runs a 1-time single-process Warp/CUDA JIT warmup (`--warmup-warp`)
   before concurrent launches so multiple jobs do not corrupt `~/.cache/warp`.

Examples:
  python scripts/run_pcore.py --inspect
  python scripts/run_pcore.py --threads 4 -- python -m pole_task.training ...
  python scripts/run_pcore.py --slot 0 --total-slots 2 --threads 4 -- python ...
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _parse_cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def detect_cpu_topology() -> dict:
    allowed = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    allowed_set = set(allowed)
    sys_cpu = Path("/sys/devices/system/cpu")
    cores: dict[tuple[int, int], dict] = {}

    for cpu in allowed:
        topo = sys_cpu / f"cpu{cpu}" / "topology"
        freq_file = sys_cpu / f"cpu{cpu}" / "cpufreq" / "cpuinfo_max_freq"
        pkg = int((topo / "physical_package_id").read_text().strip()) if (topo / "physical_package_id").is_file() else 0
        core_id = int((topo / "core_id").read_text().strip()) if (topo / "core_id").is_file() else cpu
        siblings = (
            [c for c in _parse_cpu_list((topo / "thread_siblings_list").read_text()) if c in allowed_set]
            if (topo / "thread_siblings_list").is_file()
            else [cpu]
        )
        max_khz = int(freq_file.read_text().strip()) if freq_file.is_file() else 0
        key = (pkg, core_id)
        entry = cores.setdefault(
            key, {"pkg": pkg, "core_id": core_id, "cpus": [], "max_khz": 0, "siblings_count": len(siblings)}
        )
        if cpu not in entry["cpus"]:
            entry["cpus"].append(cpu)
        entry["max_khz"] = max(entry["max_khz"], max_khz)

    core_list = sorted(cores.values(), key=lambda item: item["cpus"][0])
    if not core_list:
        return {"allowed": allowed, "p_cores": [], "e_cores": [], "is_hybrid": False}

    max_freq = max(c["max_khz"] for c in core_list)
    min_freq = min(c["max_khz"] for c in core_list)
    max_sib = max(c["siblings_count"] for c in core_list)
    min_sib = min(c["siblings_count"] for c in core_list)

    p_cores, e_cores = [], []
    is_hybrid = False
    if max_freq > 0 and min_freq > 0 and max_freq > int(min_freq * 1.12):
        is_hybrid = True
        cutoff = int(max_freq * 0.88)
        for c in core_list:
            (p_cores if c["max_khz"] >= cutoff else e_cores).append(c)
    elif max_sib > min_sib:
        # Fallback on Intel hybrid when cpufreq is hidden inside container:
        # P-cores expose 2 SMT siblings while E-cores expose 1 thread.
        is_hybrid = True
        for c in core_list:
            (p_cores if c["siblings_count"] == max_sib else e_cores).append(c)
    else:
        p_cores = core_list

    for group in (p_cores, e_cores):
        for c in group:
            c["cpus"].sort()
    return {
        "allowed": allowed,
        "p_cores": p_cores,
        "e_cores": e_cores,
        "is_hybrid": is_hybrid,
    }


def allocate_slot_cpus(topo: dict, slot: int, total_slots: int, include_smt: bool) -> list[int]:
    pool = topo["p_cores"] or topo["e_cores"]
    if not pool:
        return topo["allowed"]
    if not (0 <= slot < total_slots):
        raise ValueError(f"--slot must be in 0..{total_slots - 1}, got {slot}")
    n = len(pool)
    lo = (slot * n) // total_slots
    hi = ((slot + 1) * n) // total_slots
    chosen = pool[lo:hi] or [pool[slot % n]]
    if include_smt:
        return sorted(cpu for core in chosen for cpu in core["cpus"])
    # Prefer 1 primary hardware thread per physical P-core first (avoids SMT contention)
    return sorted(core["cpus"][0] for core in chosen)


def warmup_warp_jit() -> None:
    print("[run_pcore] Running single-process CUDA + Warp JIT cache warmup...", flush=True)
    code = (
        "import torch, warp as wp; "
        "wp.init(); "
        "print('[run_pcore] CUDA:', torch.cuda.is_available(), "
        "torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd: list[str] = []
    if "--" in argv:
        sep = argv.index("--")
        ours, cmd = argv[:sep], argv[sep + 1 :]
    else:
        ours = argv

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inspect", action="store_true", help="Print detected P-core / E-core layout and exit")
    parser.add_argument("--slot", type=int, default=0, help="0-based worker slot index")
    parser.add_argument("--total-slots", type=int, default=1, help="Total concurrent experiment slots sharing P-cores")
    parser.add_argument("--threads", type=int, default=0, help="OMP/PyTorch intra-op threads (default: auto = physical P-cores in slot, capped at 4)")
    parser.add_argument("--include-smt", action="store_true", help="Include SMT sibling threads in CPU affinity mask")
    parser.add_argument("--warmup-warp", action="store_true", help="Warm up Warp CUDA JIT cache before launching command")
    args = parser.parse_args(ours)

    topo = detect_cpu_topology()
    p_flat = [cpu for c in topo["p_cores"] for cpu in c["cpus"]]
    p_phys = [c["cpus"][0] for c in topo["p_cores"]]
    e_flat = [cpu for c in topo["e_cores"] for cpu in c["cpus"]]

    if args.inspect:
        print(f"Allowed logical CPUs ({len(topo['allowed'])}): {topo['allowed']}")
        print(f"Hybrid architecture detected: {topo['is_hybrid']}")
        print(f"P-cores ({len(topo['p_cores'])} physical / {len(p_flat)} logical): primary={p_phys}, all={p_flat}")
        print(f"E-cores ({len(topo['e_cores'])} physical / {len(e_flat)} logical): {e_flat}")
        return 0

    if args.warmup_warp:
        warmup_warp_jit()

    if not cmd:
        parser.error("No command specified after '--' (or pass --inspect)")

    selected = allocate_slot_cpus(topo, args.slot, args.total_slots, args.include_smt)
    threads = args.threads if args.threads > 0 else max(1, min(4, len(selected)))

    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
        env[var] = str(threads)
    env.setdefault("OMP_PROC_BIND", "true")
    env.setdefault("OMP_PLACES", "cores")

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(selected))
        affinity_note = f"pinned to P-core CPUs {selected}"
    else:
        affinity_note = f"sched_setaffinity unavailable on {sys.platform} (selected={selected})"

    print(
        f"[run_pcore] slot {args.slot + 1}/{args.total_slots}: {affinity_note} | threads={threads} | cmd={' '.join(cmd)}",
        flush=True,
    )
    completed = subprocess.run(cmd, env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

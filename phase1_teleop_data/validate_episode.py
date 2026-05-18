#!/usr/bin/env python3
"""
Phase 1 HDF5 dataset validator.

Run after collecting an episode to confirm the file is structurally sound
before using it for training.

Usage:
    python validate_episode.py datasets/teleop_episode_001.hdf5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


REQUIRED_GROUPS = ["observations", "actions", "normalized", "timestamps"]
REQUIRED_OBS = ["images", "joint_positions", "joint_velocities"]
REQUIRED_ACT = ["expert"]
REQUIRED_NORM = ["joint_positions_z", "joint_velocities_z", "expert_actions_z"]


def check(condition: bool, label: str, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def validate(path: Path) -> bool:
    if not path.exists():
        print(f"\n[ERROR] File not found: {path}")
        return False

    print(f"\nValidating: {path}  ({path.stat().st_size / 1024:.1f} KB)\n")
    passed = 0
    total = 0

    with h5py.File(str(path), "r") as f:
        # --- root attributes ---
        print("-- Root attributes --")
        for attr in ("schema_version", "created_utc", "target_hz"):
            ok = check(attr in f.attrs, f"attr:{attr}")
            passed += int(ok); total += 1

        # --- required groups ---
        print("\n-- Required groups --")
        for g in REQUIRED_GROUPS:
            ok = check(g in f, f"group:{g}")
            passed += int(ok); total += 1

        if not all(g in f for g in REQUIRED_GROUPS):
            print("\n[ABORT] Missing required groups, cannot continue validation.")
            return False

        # --- sample counts ---
        print("\n-- Sample counts --")
        counts = {}
        for key in REQUIRED_OBS:
            if key in f["observations"]:
                counts[f"observations/{key}"] = len(f[f"observations/{key}"])
        for key in REQUIRED_ACT:
            if key in f["actions"]:
                counts[f"actions/{key}"] = len(f[f"actions/{key}"])
        for key in REQUIRED_NORM:
            if key in f["normalized"]:
                counts[f"normalized/{key}"] = len(f[f"normalized/{key}"])
        if "synced" in f["timestamps"]:
            counts["timestamps/synced"] = len(f["timestamps/synced"])

        n_samples = counts.get("observations/images", 0)
        ok = check(n_samples > 0, "non-empty dataset", f"N={n_samples}")
        passed += int(ok); total += 1

        all_equal = all(v == n_samples for v in counts.values())
        ok = check(all_equal, "all datasets same length",
                   ", ".join(f"{k}={v}" for k, v in counts.items()))
        passed += int(ok); total += 1

        # --- shape checks ---
        print("\n-- Shapes --")
        if "observations/images" in f:
            shape = f["observations/images"].shape
            ok = check(len(shape) == 4, "images ndim==4", str(shape))
            passed += int(ok); total += 1
            ok = check(shape[3] == 3, "images channels==3 (RGB)", str(shape))
            passed += int(ok); total += 1

        if "observations/joint_positions" in f:
            shape = f["observations/joint_positions"].shape
            ok = check(len(shape) == 2, "joint_positions ndim==2", str(shape))
            passed += int(ok); total += 1

        if "actions/expert" in f:
            shape = f["actions/expert"].shape
            ok = check(len(shape) == 2, "actions/expert ndim==2", str(shape))
            passed += int(ok); total += 1

        # --- stats group ---
        print("\n-- Normalization stats --")
        has_stats = "stats" in f
        ok = check(has_stats, "stats group written on close")
        passed += int(ok); total += 1
        if has_stats:
            for key in ("joints_pos_mean", "joints_pos_std", "actions_mean", "actions_std"):
                ok = check(key in f["stats"], f"stats/{key}")
                passed += int(ok); total += 1

        # --- data sanity ---
        print("\n-- Data sanity --")
        if n_samples > 0 and "observations/joint_positions" in f:
            jp = f["observations/joint_positions"][:]
            ok = check(not np.any(np.isnan(jp)), "joint_positions has no NaN")
            passed += int(ok); total += 1
            ok = check(not np.any(np.isinf(jp)), "joint_positions has no Inf")
            passed += int(ok); total += 1

        if n_samples > 0 and "actions/expert" in f:
            act = f["actions/expert"][:]
            ok = check(not np.any(np.isnan(act)), "actions/expert has no NaN")
            passed += int(ok); total += 1

        if n_samples > 0 and "timestamps/synced" in f:
            ts = f["timestamps/synced"][:]
            ok = check(np.all(np.diff(ts) >= 0), "timestamps are monotonic")
            passed += int(ok); total += 1

    print(f"\n{'='*40}")
    result = "PASS" if passed == total else "FAIL"
    print(f"RESULT: {result}  ({passed}/{total} checks passed)")
    print(f"{'='*40}\n")
    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Phase 1 HDF5 episode file")
    parser.add_argument("file", type=Path, help="Path to HDF5 episode file")
    args = parser.parse_args()
    ok = validate(args.file)
    sys.exit(0 if ok else 1)

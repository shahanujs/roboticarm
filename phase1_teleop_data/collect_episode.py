#!/usr/bin/env python3
"""
Standalone Phase 1 data collector — no ROS2 required.

Reads from your existing teleop.py leader/follower interface and logs
synchronized camera + joint + action data into HDF5.

Run inside the roboarm pyenv:
    python collect_episode.py --episode 1

Press Ctrl+C to stop and finalize the dataset file.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import h5py
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURE THESE for your actual hardware before running
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0                  # /dev/video0 by default; change if needed
EXPECTED_JOINT_COUNT: int = 6          # number of joints on your arm
EXPECTED_ACTION_DIM: int = 6           # same as joint count for position control
TARGET_HZ: float = 30.0                # target recording frequency
IMAGE_HEIGHT: int = 480
IMAGE_WIDTH: int = 640
DATASETS_DIR: Path = Path.home() / "Documents" / "anujrobotics" / "datasets"
# ---------------------------------------------------------------------------


def ts() -> float:
    """Current wall-clock time in seconds."""
    return time.time()


class StandaloneCollector:
    """Capture synchronized camera + joint + action data into HDF5.

    How to extend for your real arm hardware
    -----------------------------------------
    Replace the two stub methods:
        read_joint_states()  ->  read from your serial/USB/SDK interface
        read_expert_action() ->  read from leader arm position or joystick
    """

    def __init__(self, output_path: Path, target_hz: float = TARGET_HZ) -> None:
        self._output_path = output_path
        self._interval = 1.0 / target_hz
        self._cap: cv2.VideoCapture | None = None
        self._h5: h5py.File | None = None
        self._size = 0

        # Online running stats for normalization (Welford)
        self._jp_mean = np.zeros(EXPECTED_JOINT_COUNT)
        self._jp_m2 = np.zeros(EXPECTED_JOINT_COUNT)
        self._jv_mean = np.zeros(EXPECTED_JOINT_COUNT)
        self._jv_m2 = np.zeros(EXPECTED_JOINT_COUNT)
        self._act_mean = np.zeros(EXPECTED_ACTION_DIM)
        self._act_m2 = np.zeros(EXPECTED_ACTION_DIM)

    # ------------------------------------------------------------------
    # HARDWARE STUBS — replace these with your real interface calls
    # ------------------------------------------------------------------

    def read_joint_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions [J], velocities [J]) from follower arm.

        Replace this stub with your real SDK/serial call.
        Example for SO-ARM101 via lerobot:
            obs = arm.read_state()
            return obs["pos"], obs["vel"]
        """
        positions = np.zeros(EXPECTED_JOINT_COUNT, dtype=np.float32)
        velocities = np.zeros(EXPECTED_JOINT_COUNT, dtype=np.float32)
        return positions, velocities

    def read_expert_action(self) -> np.ndarray:
        """Return target joint positions from leader arm as action vector [A].

        Replace this stub with your real leader arm readout.
        Example:
            return leader_arm.read_positions()
        """
        return np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # HDF5 SCHEMA
    # ------------------------------------------------------------------

    def _build_hdf5(self, first_frame: np.ndarray, joint_names: list[str]) -> None:
        h, w, c = first_frame.shape
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        f = h5py.File(str(self._output_path), "w")
        f.attrs["schema_version"] = "1.0"
        f.attrs["created_utc"] = datetime.now(UTC).isoformat()
        f.attrs["target_hz"] = TARGET_HZ
        f.attrs["image_shape_hwc"] = [h, w, c]
        f.attrs["joint_count"] = EXPECTED_JOINT_COUNT
        f.attrs["action_dim"] = EXPECTED_ACTION_DIM

        obs = f.create_group("observations")
        obs.create_dataset("images", shape=(0, h, w, c), maxshape=(None, h, w, c),
                           chunks=(1, h, w, c), dtype=np.uint8, compression="gzip", compression_opts=4)
        obs.create_dataset("joint_positions", shape=(0, EXPECTED_JOINT_COUNT),
                           maxshape=(None, EXPECTED_JOINT_COUNT), chunks=(256, EXPECTED_JOINT_COUNT),
                           dtype=np.float32, compression="gzip", compression_opts=4)
        obs.create_dataset("joint_velocities", shape=(0, EXPECTED_JOINT_COUNT),
                           maxshape=(None, EXPECTED_JOINT_COUNT), chunks=(256, EXPECTED_JOINT_COUNT),
                           dtype=np.float32, compression="gzip", compression_opts=4)
        obs.create_dataset("joint_names",
                           data=np.array(joint_names, dtype=h5py.string_dtype(encoding="utf-8")))

        act = f.create_group("actions")
        act.create_dataset("expert", shape=(0, EXPECTED_ACTION_DIM),
                           maxshape=(None, EXPECTED_ACTION_DIM), chunks=(256, EXPECTED_ACTION_DIM),
                           dtype=np.float32, compression="gzip", compression_opts=4)

        norm = f.create_group("normalized")
        for name, dim in [("joint_positions_z", EXPECTED_JOINT_COUNT),
                          ("joint_velocities_z", EXPECTED_JOINT_COUNT),
                          ("expert_actions_z", EXPECTED_ACTION_DIM)]:
            norm.create_dataset(name, shape=(0, dim), maxshape=(None, dim),
                                chunks=(256, dim), dtype=np.float32,
                                compression="gzip", compression_opts=4)

        tsg = f.create_group("timestamps")
        for name in ("synced",):
            tsg.create_dataset(name, shape=(0,), maxshape=(None,),
                               chunks=(1024,), dtype=np.float64,
                               compression="gzip", compression_opts=4)

        self._h5 = f

    def _welford_update(self, mean: np.ndarray, m2: np.ndarray, count: int,
                        x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = x - mean
        mean = mean + delta / count
        m2 = m2 + delta * (x - mean)
        return mean, m2

    def _normalize(self, x: np.ndarray, mean: np.ndarray, m2: np.ndarray,
                   count: int, eps: float = 1e-6) -> np.ndarray:
        std = np.sqrt(np.maximum(m2 / max(count - 1, 1), eps))
        return (x - mean) / std

    def _append(self, image: np.ndarray, pos: np.ndarray, vel: np.ndarray,
                action: np.ndarray, t: float) -> None:
        assert self._h5 is not None

        self._size += 1
        c = self._size

        # Update running stats
        self._jp_mean, self._jp_m2 = self._welford_update(self._jp_mean, self._jp_m2, c, pos)
        self._jv_mean, self._jv_m2 = self._welford_update(self._jv_mean, self._jv_m2, c, vel)
        self._act_mean, self._act_m2 = self._welford_update(self._act_mean, self._act_m2, c, action)

        pos_z = self._normalize(pos, self._jp_mean, self._jp_m2, c).astype(np.float32)
        vel_z = self._normalize(vel, self._jv_mean, self._jv_m2, c).astype(np.float32)
        act_z = self._normalize(action, self._act_mean, self._act_m2, c).astype(np.float32)

        def _ext(ds: h5py.Dataset, val: np.ndarray) -> None:
            if ds.ndim == 1:
                ds.resize((self._size,))
                ds[self._size - 1] = val
            else:
                ds.resize((self._size, *ds.shape[1:]))
                ds[self._size - 1] = val

        obs = self._h5["observations"]
        _ext(obs["images"], image)
        _ext(obs["joint_positions"], pos)
        _ext(obs["joint_velocities"], vel)

        act = self._h5["actions"]
        _ext(act["expert"], action)

        norm = self._h5["normalized"]
        _ext(norm["joint_positions_z"], pos_z)
        _ext(norm["joint_velocities_z"], vel_z)
        _ext(norm["expert_actions_z"], act_z)

        _ext(self._h5["timestamps"]["synced"], np.float64(t))

        if self._size % 50 == 0:
            self._h5.flush()
            fps = self._size / (time.time() - self._start_time)
            print(f"  Samples: {self._size:5d}  live fps: {fps:.1f}")

    def _write_final_stats(self) -> None:
        if self._h5 is None or self._size < 2:
            return
        if "stats" in self._h5:
            del self._h5["stats"]
        stats = self._h5.create_group("stats")

        def _std(m2: np.ndarray) -> np.ndarray:
            return np.sqrt(np.maximum(m2 / max(self._size - 1, 1), 1e-6))

        stats.create_dataset("joints_pos_mean", data=self._jp_mean)
        stats.create_dataset("joints_pos_std", data=_std(self._jp_m2))
        stats.create_dataset("joints_vel_mean", data=self._jv_mean)
        stats.create_dataset("joints_vel_std", data=_std(self._jv_m2))
        stats.create_dataset("actions_mean", data=self._act_mean)
        stats.create_dataset("actions_std", data=_std(self._act_m2))

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start camera and record until Ctrl+C."""
        print(f"\nStarting camera (index {CAMERA_INDEX})...")
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}. Check that camera is connected.")
            sys.exit(1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

        # Read first frame to fix image shape for HDF5 schema
        ret, first_frame = self._cap.read()
        if not ret:
            print("[ERROR] Could not read first camera frame.")
            sys.exit(1)
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        joint_names = [f"joint_{i}" for i in range(EXPECTED_JOINT_COUNT)]
        self._build_hdf5(first_frame_rgb, joint_names)

        print(f"Recording to: {self._output_path}")
        print("Teleop your arm now. Press Ctrl+C to stop and save.\n")
        self._start_time = time.time()

        try:
            while True:
                loop_start = time.time()

                ret, frame = self._cap.read()
                if not ret:
                    print("[WARN] Camera read failed, skipping frame")
                    continue

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pos, vel = self.read_joint_states()
                action = self.read_expert_action()
                t = time.time()

                self._append(image_rgb, pos.astype(np.float32),
                             vel.astype(np.float32), action.astype(np.float32), t)

                elapsed = time.time() - loop_start
                sleep_time = self._interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\nStopped. Finalizing {self._size} samples...")

        finally:
            if self._cap is not None:
                self._cap.release()
            if self._h5 is not None:
                self._write_final_stats()
                self._h5.flush()
                self._h5.close()
            print(f"Saved: {self._output_path}  ({self._size} samples)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 standalone data collector")
    p.add_argument("--episode", type=int, default=1, help="Episode number (used in filename)")
    p.add_argument("--hz", type=float, default=TARGET_HZ, help="Target recording Hz")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera index")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out = DATASETS_DIR / f"teleop_episode_{args.episode:03d}.hdf5"
    collector = StandaloneCollector(output_path=out, target_hz=args.hz)
    collector.run()

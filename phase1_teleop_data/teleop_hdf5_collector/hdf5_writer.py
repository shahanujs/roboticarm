"""HDF5 writer for synchronized teleoperation trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray


UInt8Array = NDArray[np.uint8]
Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


@dataclass(frozen=True)
class WriterMetadata:
    """Metadata attached to generated dataset files."""

    image_topic: str
    joint_state_topic: str
    action_topic: str
    target_hz: float
    slop_sec: float
    frame_id: str


class Hdf5TeleopWriter:
    """Append-only HDF5 writer for synchronized multi-modal teleoperation data."""

    def __init__(
        self,
        output_path: Path,
        joint_count: int,
        action_dim: int,
        image_shape_hwc: tuple[int, int, int],
        joint_names: list[str],
        metadata: WriterMetadata,
    ) -> None:
        self._output_path = output_path
        self._joint_count = joint_count
        self._action_dim = action_dim
        self._image_shape_hwc = image_shape_hwc
        self._joint_names = joint_names
        self._metadata = metadata
        self._size = 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(str(output_path), "w")
        self._build_schema()

    @property
    def size(self) -> int:
        """Number of samples currently written."""
        return self._size

    def _build_schema(self) -> None:
        h, w, c = self._image_shape_hwc
        root = self._h5
        root.attrs["schema_version"] = "1.0"
        root.attrs["created_utc"] = datetime.now(UTC).isoformat()
        root.attrs["image_topic"] = self._metadata.image_topic
        root.attrs["joint_state_topic"] = self._metadata.joint_state_topic
        root.attrs["action_topic"] = self._metadata.action_topic
        root.attrs["target_hz"] = self._metadata.target_hz
        root.attrs["slop_sec"] = self._metadata.slop_sec
        root.attrs["frame_id"] = self._metadata.frame_id

        obs = root.create_group("observations")
        act = root.create_group("actions")
        norm = root.create_group("normalized")
        ts = root.create_group("timestamps")

        obs.create_dataset(
            "images",
            shape=(0, h, w, c),
            maxshape=(None, h, w, c),
            chunks=(1, h, w, c),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )
        obs.create_dataset(
            "joint_positions",
            shape=(0, self._joint_count),
            maxshape=(None, self._joint_count),
            chunks=(256, self._joint_count),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        obs.create_dataset(
            "joint_velocities",
            shape=(0, self._joint_count),
            maxshape=(None, self._joint_count),
            chunks=(256, self._joint_count),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        obs.create_dataset(
            "joint_names",
            data=np.asarray(self._joint_names, dtype=h5py.string_dtype(encoding="utf-8")),
        )

        act.create_dataset(
            "expert",
            shape=(0, self._action_dim),
            maxshape=(None, self._action_dim),
            chunks=(256, self._action_dim),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )

        norm.create_dataset(
            "joint_positions_z",
            shape=(0, self._joint_count),
            maxshape=(None, self._joint_count),
            chunks=(256, self._joint_count),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        norm.create_dataset(
            "joint_velocities_z",
            shape=(0, self._joint_count),
            maxshape=(None, self._joint_count),
            chunks=(256, self._joint_count),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )
        norm.create_dataset(
            "expert_actions_z",
            shape=(0, self._action_dim),
            maxshape=(None, self._action_dim),
            chunks=(256, self._action_dim),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )

        for name in ("image", "joint_state", "action", "synced"):
            ts.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(1024,),
                dtype=np.float64,
                compression="gzip",
                compression_opts=4,
            )

    def _append_row(self, dataset: h5py.Dataset, value: np.ndarray) -> None:
        new_size = self._size + 1
        if dataset.ndim == 1:
            dataset.resize((new_size,))
            dataset[self._size] = value
        else:
            dataset.resize((new_size, *dataset.shape[1:]))
            dataset[self._size, ...] = value

    def append(
        self,
        image: UInt8Array,
        joint_positions: Float32Array,
        joint_velocities: Float32Array,
        expert_action: Float32Array,
        joint_positions_z: Float32Array,
        joint_velocities_z: Float32Array,
        expert_action_z: Float32Array,
        image_ts: float,
        joint_ts: float,
        action_ts: float,
        synced_ts: float,
    ) -> None:
        """Append one synchronized sample to all modalities."""
        if image.shape != self._image_shape_hwc:
            raise ValueError(f"Image shape mismatch: expected {self._image_shape_hwc}, got {image.shape}")
        if joint_positions.shape != (self._joint_count,):
            raise ValueError("joint_positions shape mismatch")
        if joint_velocities.shape != (self._joint_count,):
            raise ValueError("joint_velocities shape mismatch")
        if expert_action.shape != (self._action_dim,):
            raise ValueError("expert_action shape mismatch")

        obs = self._h5["observations"]
        act = self._h5["actions"]
        norm = self._h5["normalized"]
        ts = self._h5["timestamps"]

        self._append_row(obs["images"], image)
        self._append_row(obs["joint_positions"], joint_positions)
        self._append_row(obs["joint_velocities"], joint_velocities)
        self._append_row(act["expert"], expert_action)

        self._append_row(norm["joint_positions_z"], joint_positions_z)
        self._append_row(norm["joint_velocities_z"], joint_velocities_z)
        self._append_row(norm["expert_actions_z"], expert_action_z)

        self._append_row(ts["image"], np.asarray(image_ts, dtype=np.float64))
        self._append_row(ts["joint_state"], np.asarray(joint_ts, dtype=np.float64))
        self._append_row(ts["action"], np.asarray(action_ts, dtype=np.float64))
        self._append_row(ts["synced"], np.asarray(synced_ts, dtype=np.float64))

        self._size += 1

    def write_stats(
        self,
        joints_pos_mean: Float64Array,
        joints_pos_std: Float64Array,
        joints_vel_mean: Float64Array,
        joints_vel_std: Float64Array,
        actions_mean: Float64Array,
        actions_std: Float64Array,
    ) -> None:
        """Persist final normalization statistics for training reproducibility."""
        if "stats" in self._h5:
            del self._h5["stats"]
        stats = self._h5.create_group("stats")
        stats.create_dataset("joints_pos_mean", data=joints_pos_mean)
        stats.create_dataset("joints_pos_std", data=joints_pos_std)
        stats.create_dataset("joints_vel_mean", data=joints_vel_mean)
        stats.create_dataset("joints_vel_std", data=joints_vel_std)
        stats.create_dataset("actions_mean", data=actions_mean)
        stats.create_dataset("actions_std", data=actions_std)

    def flush(self) -> None:
        """Flush all pending data to disk."""
        self._h5.flush()

    def close(self) -> None:
        """Close underlying file handle."""
        self._h5.close()

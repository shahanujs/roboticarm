"""ROS2 node for synchronized teleoperation data logging to HDF5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import message_filters
import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from .config import CollectorConfig
from .hdf5_writer import Hdf5TeleopWriter, WriterMetadata
from .normalization import RunningStats


Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True)
class DecodedImage:
    """Decoded image payload and timestamp."""

    data: UInt8Array
    ts_sec: float
    frame_id: str


def stamp_to_sec(stamp: object) -> float:
    """Convert ROS stamp-like object to float seconds."""
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        raise ValueError("Stamp missing sec/nanosec")
    return float(sec) + (float(nanosec) * 1e-9)


def decode_image(msg: Image, allowed_encodings: tuple[str, ...]) -> DecodedImage:
    """Decode `sensor_msgs/Image` into HWC uint8 array.

    Supported encodings: rgb8, bgr8, mono8
    """
    if msg.encoding not in allowed_encodings:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    if msg.height <= 0 or msg.width <= 0:
        raise ValueError("Invalid image dimensions")

    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding in ("rgb8", "bgr8"):
        channels = 3
        expected = msg.height * msg.width * channels
        if raw.size != expected:
            raise ValueError(f"Image buffer size mismatch: expected {expected}, got {raw.size}")
        image = raw.reshape((msg.height, msg.width, channels))
        if msg.encoding == "bgr8":
            image = image[:, :, ::-1]
    elif msg.encoding == "mono8":
        expected = msg.height * msg.width
        if raw.size != expected:
            raise ValueError(f"Image buffer size mismatch: expected {expected}, got {raw.size}")
        gray = raw.reshape((msg.height, msg.width))
        image = np.repeat(gray[:, :, np.newaxis], repeats=3, axis=2)
    else:
        raise ValueError(f"Encoding parser missing for: {msg.encoding}")

    return DecodedImage(
        data=image.astype(np.uint8, copy=False),
        ts_sec=stamp_to_sec(msg.header.stamp),
        frame_id=msg.header.frame_id,
    )


class TeleopDataCollectorNode(Node):
    """Collect synchronized teleoperation data and persist it in HDF5."""

    def __init__(self, config: CollectorConfig) -> None:
        super().__init__(config.node_name)
        self._cfg = config
        self._writer: Optional[Hdf5TeleopWriter] = None
        self._joint_name_order: Optional[list[str]] = None
        self._last_logged_time_sec: float = -1e9
        self._dropped_samples: int = 0

        self._joint_pos_stats = RunningStats(config.expected_joint_count)
        self._joint_vel_stats = RunningStats(config.expected_joint_count)
        self._action_stats = RunningStats(config.expected_action_dim)

        self._image_sub = message_filters.Subscriber(self, Image, config.image_topic)
        self._joint_sub = message_filters.Subscriber(self, JointState, config.joint_state_topic)
        self._action_sub = message_filters.Subscriber(self, Float64MultiArray, config.action_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            fs=[self._image_sub, self._joint_sub, self._action_sub],
            queue_size=config.queue_size,
            slop=config.slop_sec,
            allow_headerless=True,
        )
        self._sync.registerCallback(self._on_synced_sample)

        self.get_logger().info(
            (
                "Collector started: "
                f"image={config.image_topic} "
                f"joint={config.joint_state_topic} "
                f"action={config.action_topic} "
                f"output={config.output_path} "
                f"target_hz={config.target_hz:.2f}"
            )
        )

    def _extract_joint_vectors(self, joint_msg: JointState) -> tuple[Float32Array, Float32Array, list[str]]:
        names = list(joint_msg.name)
        if len(names) != self._cfg.expected_joint_count:
            raise ValueError(
                f"Expected {self._cfg.expected_joint_count} joints, got {len(names)}"
            )
        if len(joint_msg.position) != self._cfg.expected_joint_count:
            raise ValueError("Joint position size mismatch")
        if len(joint_msg.velocity) != self._cfg.expected_joint_count:
            raise ValueError("Joint velocity size mismatch")

        if self._joint_name_order is None:
            self._joint_name_order = names
        elif names != self._joint_name_order:
            raise ValueError("Joint ordering changed during capture")

        pos = np.asarray(joint_msg.position, dtype=np.float32)
        vel = np.asarray(joint_msg.velocity, dtype=np.float32)
        return pos, vel, names

    def _extract_action(self, action_msg: Float64MultiArray) -> Float32Array:
        data = np.asarray(action_msg.data, dtype=np.float32)
        if data.shape != (self._cfg.expected_action_dim,):
            raise ValueError(
                f"Expected action dim {self._cfg.expected_action_dim}, got {data.shape}"
            )
        return data

    def _create_writer_if_needed(
        self,
        image: DecodedImage,
        joint_names: list[str],
    ) -> None:
        if self._writer is not None:
            return

        metadata = WriterMetadata(
            image_topic=self._cfg.image_topic,
            joint_state_topic=self._cfg.joint_state_topic,
            action_topic=self._cfg.action_topic,
            target_hz=self._cfg.target_hz,
            slop_sec=self._cfg.slop_sec,
            frame_id=image.frame_id,
        )
        self._writer = Hdf5TeleopWriter(
            output_path=self._cfg.output_path,
            joint_count=self._cfg.expected_joint_count,
            action_dim=self._cfg.expected_action_dim,
            image_shape_hwc=(image.data.shape[0], image.data.shape[1], image.data.shape[2]),
            joint_names=joint_names,
            metadata=metadata,
        )

    def _should_log(self, synced_ts: float) -> bool:
        if synced_ts - self._last_logged_time_sec < self._cfg.min_interval_sec:
            return False
        return True

    def _on_synced_sample(self, image_msg: Image, joint_msg: JointState, action_msg: Float64MultiArray) -> None:
        try:
            image = decode_image(image_msg, self._cfg.allowed_encodings)
            pos, vel, joint_names = self._extract_joint_vectors(joint_msg)
            action = self._extract_action(action_msg)
            joint_ts = stamp_to_sec(joint_msg.header.stamp)
            action_ts = float(self.get_clock().now().nanoseconds) * 1e-9
            synced_ts = max(image.ts_sec, joint_ts, action_ts)

            if not self._should_log(synced_ts):
                return

            self._create_writer_if_needed(image=image, joint_names=joint_names)
            if self._writer is None:
                raise RuntimeError("Writer failed to initialize")

            self._joint_pos_stats.update(pos)
            self._joint_vel_stats.update(vel)
            self._action_stats.update(action)

            pos_z = self._joint_pos_stats.normalize(pos).astype(np.float32)
            vel_z = self._joint_vel_stats.normalize(vel).astype(np.float32)
            action_z = self._action_stats.normalize(action).astype(np.float32)

            self._writer.append(
                image=image.data,
                joint_positions=pos,
                joint_velocities=vel,
                expert_action=action,
                joint_positions_z=pos_z,
                joint_velocities_z=vel_z,
                expert_action_z=action_z,
                image_ts=image.ts_sec,
                joint_ts=joint_ts,
                action_ts=action_ts,
                synced_ts=synced_ts,
            )
            self._last_logged_time_sec = synced_ts

            if self._writer.size % self._cfg.flush_every_n == 0:
                self._writer.flush()
                self.get_logger().info(f"Flushed {self._writer.size} samples")

        except Exception as exc:  # noqa: BLE001
            self._dropped_samples += 1
            self.get_logger().error(f"Sample dropped ({self._dropped_samples}): {exc}")

    def close(self) -> None:
        """Flush and close resources; write final normalization stats."""
        if self._writer is None:
            return

        self._writer.write_stats(
            joints_pos_mean=self._joint_pos_stats.mean,
            joints_pos_std=self._joint_pos_stats.std(),
            joints_vel_mean=self._joint_vel_stats.mean,
            joints_vel_std=self._joint_vel_stats.std(),
            actions_mean=self._action_stats.mean,
            actions_std=self._action_stats.std(),
        )
        self._writer.flush()
        self._writer.close()
        self.get_logger().info(f"Writer closed. Final samples: {self._writer.size}")

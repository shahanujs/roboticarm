"""Configuration model and loader for the teleoperation data collector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CollectorConfig:
    """Runtime configuration for ROS2 teleoperation data collection."""

    node_name: str
    output_path: Path
    target_hz: float
    slop_sec: float
    queue_size: int
    flush_every_n: int
    expected_joint_count: int
    expected_action_dim: int
    image_topic: str
    joint_state_topic: str
    action_topic: str
    allowed_encodings: tuple[str, ...]

    @property
    def min_interval_sec(self) -> float:
        """Minimum interval between logged samples in seconds."""
        if self.target_hz <= 0.0:
            raise ValueError("target_hz must be > 0")
        return 1.0 / self.target_hz


def _require(mapping: dict[str, Any], key: str) -> Any:
    """Return required key from mapping or raise ValueError."""
    if key not in mapping:
        raise ValueError(f"Missing required key: {key}")
    return mapping[key]


def load_config(config_path: Path) -> CollectorConfig:
    """Load collector config from YAML file.

    Args:
        config_path: Path to YAML configuration.

    Returns:
        Parsed and validated CollectorConfig instance.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    node = _require(raw, "node")
    topics = _require(raw, "topics")
    image = _require(raw, "image")
    if not isinstance(node, dict) or not isinstance(topics, dict) or not isinstance(image, dict):
        raise ValueError("node, topics, and image sections must be mappings")

    allowed = tuple(str(x) for x in _require(image, "allowed_encodings"))

    return CollectorConfig(
        node_name=str(_require(node, "name")),
        output_path=Path(str(_require(node, "output_path"))).expanduser().resolve(),
        target_hz=float(_require(node, "target_hz")),
        slop_sec=float(_require(node, "slop_sec")),
        queue_size=int(_require(node, "queue_size")),
        flush_every_n=int(_require(node, "flush_every_n")),
        expected_joint_count=int(_require(node, "expected_joint_count")),
        expected_action_dim=int(_require(node, "expected_action_dim")),
        image_topic=str(_require(topics, "image")),
        joint_state_topic=str(_require(topics, "joint_state")),
        action_topic=str(_require(topics, "action")),
        allowed_encodings=allowed,
    )

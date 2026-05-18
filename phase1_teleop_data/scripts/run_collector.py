#!/usr/bin/env python3
"""Command line entrypoint for the teleoperation data collector node."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import rclpy

from teleop_hdf5_collector.collector_node import TeleopDataCollectorNode
from teleop_hdf5_collector.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="ROS2 teleoperation HDF5 data collector")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> int:
    """Run collector node until interrupted."""
    args = parse_args()
    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load config: {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    node = TeleopDataCollectorNode(cfg)
    try:
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

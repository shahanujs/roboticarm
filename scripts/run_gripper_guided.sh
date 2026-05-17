#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# This mode does not use cameras, but stop any stale viewer anyway to reduce bus load.
pkill -f "scripts/view_cameras.py" >/dev/null 2>&1 || true

exec python scripts/gripper_input_arm_follow.py "$@"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Stop any stale viewer instance first.
pkill -f "scripts/view_cameras.py" >/dev/null 2>&1 || true

exec python scripts/view_cameras.py --mode web --host 0.0.0.0 --port "${PORT:-8080}"

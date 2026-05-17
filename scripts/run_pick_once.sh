#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Camera devices cannot be shared between viewer and inference.
pkill -f "scripts/view_cameras.py" >/dev/null 2>&1 || true

python inference.py \
  --mode scripted \
  --attempts "${ATTEMPTS:-1}" \
  --drop_x "${DROP_X:-0.22}" \
  --drop_y "${DROP_Y:--0.18}" \
  --drop_z "${DROP_Z:-0.08}" \
  --pick_z "${PICK_Z:-0.05}" \
  "$@"

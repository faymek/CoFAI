#!/usr/bin/env bash
# Extract ImageNet train features for PQFC (DINOv3 slot24 / blk23).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_env.sh"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SOURCE_ROOT"

DEVICE="${DEVICE:-cuda:0}"
EXTRA=()
if [[ -n "${IMAGENET_ROOT:-}" ]]; then
  EXTRA+=(--imagenet_root "$IMAGENET_ROOT")
fi
if [[ -n "${PATHNAME_LIST:-}" ]]; then
  EXTRA+=(--pathname_list "$PATHNAME_LIST")
elif [[ -f "$(dirname "$PROJECT_ROOT")/utils/imagenet_selected_pathname5000.txt" ]]; then
  EXTRA+=(--pathname_list "$(dirname "$PROJECT_ROOT")/utils/imagenet_selected_pathname5000.txt")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  EXTRA+=(--output_dir "$OUTPUT_DIR")
fi

exec $PYTHON "$PQFC_DIR/offline/extract_train_features.py" \
  --device "$DEVICE" \
  "${EXTRA[@]}" \
  "$@"

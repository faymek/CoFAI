#!/bin/bash
# Extract ImageNet train features for PQFC (DINOv3 slot24 / blk23).
# All unknown args are forwarded to extract_train_features.py.
#
# Usage:
#   bash examples/pqfc/scripts/extract_train.sh
#   bash examples/pqfc/scripts/extract_train.sh --img_size 224 --max_images 1000 --device cuda:0
#   IMAGENET_ROOT=/path/to/val bash examples/pqfc/scripts/extract_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
FEATCODEC_ROOT="$(dirname "$COFAI_ROOT")"
cd "$COFAI_ROOT"
export PROJECT_ROOT="$COFAI_ROOT"
unset PYTHONPATH

PYTHON="${PYTHON:-poetry run python}"
DEVICE="${DEVICE:-cuda:0}"

EXTRA=()
if [ -n "${IMAGENET_ROOT:-}" ]; then
  EXTRA+=(--imagenet_root "$IMAGENET_ROOT")
fi
if [ -n "${PATHNAME_LIST:-}" ]; then
  EXTRA+=(--pathname_list "$PATHNAME_LIST")
elif [ -f "$FEATCODEC_ROOT/utils/imagenet_selected_pathname5000.txt" ]; then
  EXTRA+=(--pathname_list "$FEATCODEC_ROOT/utils/imagenet_selected_pathname5000.txt")
fi
if [ -n "${OUTPUT_DIR:-}" ]; then
  EXTRA+=(--output_dir "$OUTPUT_DIR")
fi

exec $PYTHON "$PQFC_DIR/offline/extract_train_features.py" \
  --device "$DEVICE" \
  "${EXTRA[@]}" \
  "$@"

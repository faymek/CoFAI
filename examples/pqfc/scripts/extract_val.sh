#!/usr/bin/env bash
# Extract CTC val features (ADE20K semseg + NYUv2 depth).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_env.sh"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SOURCE_ROOT"

CONFIG="${CONFIG:-$SOURCE_ROOT/examples/pqfc/configs/dinov3_blk23.yaml}"
GPU="${GPU:-0}"

HAS_TASK=0
for a in "$@"; do
  if [[ "$a" == "--task" ]]; then HAS_TASK=1; break; fi
done

run_one() {
  local task=$1
  shift
  echo "========== extract val: $task =========="
  $PYTHON "$PQFC_DIR/offline/extract_val_features.py" \
    --config "$CONFIG" \
    --task "$task" \
    --gpu "$GPU" \
    "$@"
}

if [[ "$HAS_TASK" -eq 1 ]]; then
  $PYTHON "$PQFC_DIR/offline/extract_val_features.py" --config "$CONFIG" --gpu "$GPU" "$@"
else
  run_one semseg "$@"
  run_one depth "$@"
fi

echo "Val features under: \$PROJECT_ROOT/features/dinov3_vitl16_slot24/{semseg,depth}/"

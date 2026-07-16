#!/bin/bash
# Extract CTC val features (ADE20K semseg + NYUv2 depth) via VTM extract helper.
# Writes features/dinov3_vitl16_slot24/{semseg,depth}/{tokens,meta}/
#
# Usage:
#   bash examples/pqfc/scripts/extract_val.sh
#   bash examples/pqfc/scripts/extract_val.sh --task semseg --gpu 0
#   bash examples/pqfc/scripts/extract_val.sh --subset examples/vtm/subsets/ade20k_smoke.txt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
cd "$COFAI_ROOT"
export PROJECT_ROOT="$COFAI_ROOT"
unset PYTHONPATH

PYTHON="${PYTHON:-poetry run python}"
VTM_PY="${VTM_PY:-examples/vtm/run_vtm_dinov3.py}"
CONFIG="${CONFIG:-examples/vtm/configs/dinov3_slot24.yaml}"
GPU="${GPU:-0}"

# If user passed --task, only that task; else both.
HAS_TASK=0
for a in "$@"; do
  if [ "$a" = "--task" ]; then HAS_TASK=1; break; fi
done

run_one() {
  local task=$1
  shift
  echo "========== extract val: $task =========="
  $PYTHON "$VTM_PY" extract \
    --config "$CONFIG" \
    --task "$task" \
    --gpu "$GPU" \
    "$@"
}

if [ "$HAS_TASK" -eq 1 ]; then
  $PYTHON "$VTM_PY" extract --config "$CONFIG" --gpu "$GPU" "$@"
else
  run_one semseg "$@"
  run_one depth "$@"
fi

echo "Val features under: features/dinov3_vitl16_slot24/{semseg,depth}/"

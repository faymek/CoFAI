#!/bin/bash
# One-click online CTC eval for DINOv3 SoftPQ release weights.
# Sweeps plan multi_run (8 rate points) x {semseg, depth} with multi-GPU parallel.
#
# Usage:
#   bash examples/orfc_2446/dinov3/scripts/run_eval_release_ctc.sh
#   GPUS=0,1,2,3 TASKS=both bash .../run_eval_release_ctc.sh
#   GPUS=0 TASK=semseg bash .../run_eval_release_ctc.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
unset PYTHONPATH

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"

GPUS="${GPUS:-0,1,2,3}"
TASK="${TASK:-${TASKS:-both}}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
PYTHON="${PYTHON:-poetry run python}"

echo "============================================================"
echo "  DINOv3 SoftPQ release CTC eval (online Engine)"
echo "  task=${TASK}  gpus=${GPUS}  output=${OUTPUT_DIR}"
echo "  Started: $(date)"
echo "============================================================"

$PYTHON examples/orfc_2446/dinov3/run_eval_orfc_2446_dinov3.py \
  --task "$TASK" \
  --multi-run \
  --gpus "$GPUS" \
  --cuda --real \
  --output_dir "$OUTPUT_DIR"

echo "Complete: $(date)"
echo "Results under: ${OUTPUT_DIR}/SoftPQ/dinov3-vitl16-slot24/"

#!/usr/bin/env bash
# Full ADE20K val semseg: extract -> VTM (QP 22/32/42, 24 workers) -> replay.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PROJECT_ROOT="$ROOT"
cd "$ROOT"

PY="${PYTHON:-poetry run python}"
SCRIPT="examples/vtm/run_vtm_dinov3.py"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-24}"
QPS="${QPS:-22 32 42}"

echo "========== semseg extract (2000 val, GPU ${GPU}) =========="
CUDA_VISIBLE_DEVICES="${GPU}" $PY "$SCRIPT" extract --task semseg --gpu 0

echo "========== semseg VTM (CPU only, workers=${WORKERS}) =========="
CUDA_VISIBLE_DEVICES="" $PY "$SCRIPT" vtm --task semseg --qps $QPS --workers "$WORKERS"

echo "========== semseg replay (GPU ${GPU}) =========="
for qp in $QPS; do
  CUDA_VISIBLE_DEVICES="${GPU}" $PY "$SCRIPT" replay --mode vtm --task semseg --qps "$qp" --gpu 0
done

echo "[run_seg_vtm] DONE"

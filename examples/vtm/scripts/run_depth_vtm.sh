#!/usr/bin/env bash
# Full NYUv2 test depth: extract -> VTM (QP 22/32/42, 24 workers) -> replay.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PROJECT_ROOT="$ROOT"
cd "$ROOT"

PY="${PYTHON:-poetry run python}"
SCRIPT="examples/vtm/run_vtm_dinov3.py"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-24}"
QPS="${QPS:-22 32 42}"

echo "========== depth extract (654 test, GPU ${GPU}) =========="
CUDA_VISIBLE_DEVICES="${GPU}" $PY "$SCRIPT" extract --task depth --gpu 0

echo "========== depth VTM (CPU only, workers=${WORKERS}) =========="
CUDA_VISIBLE_DEVICES="" $PY "$SCRIPT" vtm --task depth --qps $QPS --workers "$WORKERS"

echo "========== depth replay (GPU ${GPU}) =========="
for qp in $QPS; do
  CUDA_VISIBLE_DEVICES="${GPU}" $PY "$SCRIPT" replay --mode vtm --task depth --qps "$qp" --gpu 0
done

echo "[run_depth_vtm] DONE"

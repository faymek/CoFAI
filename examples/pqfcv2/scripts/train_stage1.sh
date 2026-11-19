#!/usr/bin/env bash
# Stage 1: orthogonal transform + K2048 e8 PQ, lambda 100, on the combined 20k features.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export PROJECT_ROOT PYTHONPATH="$SOURCE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU="${GPU:-1}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u "$SOURCE_ROOT/examples/pqfcv2/train_stage1.py" \
  --config "$SOURCE_ROOT/examples/pqfcv2/configs/stage1_k2048e8_postln_l100_splitlr.yaml" \
  --gpu 0 "$@"

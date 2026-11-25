#!/usr/bin/env bash
# Train PQFCv2 stage 1 (defaults to DINOv3 slot24 baseline; override via CONFIG=...).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SOURCE_ROOT}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
export PROJECT_ROOT PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-$SOURCE_ROOT/examples/pqfcv2/configs/stage1_k2048e8_postln_l100_splitlr.yaml}"
GPU="${GPU:-0}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u "$SOURCE_ROOT/examples/pqfcv2/train_stage1.py" \
  --config "$CONFIG" \
  --gpu 0 "$@"

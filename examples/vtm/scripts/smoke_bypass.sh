#!/usr/bin/env bash
# Smoke: extract + bypass replay + verify against cofai-eval (8 samples each).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PROJECT_ROOT="$ROOT"
cd "$ROOT"

PY="${PYTHON:-poetry run python}"
SCRIPT="examples/vtm/run_vtm_dinov3.py"
GPU="${GPU:-0}"

echo "========== semseg smoke =========="
$PY "$SCRIPT" extract --task semseg --gpu "$GPU" --subset examples/vtm/subsets/ade20k_smoke.txt
$PY "$SCRIPT" replay --mode bypass --task semseg --gpu "$GPU" --subset examples/vtm/subsets/ade20k_smoke.txt
$PY "$SCRIPT" verify --task semseg --subset examples/vtm/subsets/ade20k_smoke.txt

echo "========== depth smoke =========="
$PY "$SCRIPT" extract --task depth --gpu "$GPU" --subset examples/vtm/subsets/nyu_smoke.txt
$PY "$SCRIPT" replay --mode bypass --task depth --gpu "$GPU" --subset examples/vtm/subsets/nyu_smoke.txt
$PY "$SCRIPT" verify --task depth --subset examples/vtm/subsets/nyu_smoke.txt

echo "[smoke_bypass] ALL PASS"

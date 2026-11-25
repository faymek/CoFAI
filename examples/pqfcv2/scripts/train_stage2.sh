#!/usr/bin/env bash
set -euo pipefail
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SOURCE_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
N="${1:?usage: train_stage2.sh LATENT_TOKENS [extra arguments]}"; shift
export PYTHONPATH="$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON_BIN:-python}" -u "$SOURCE_ROOT/examples/pqfcv2/train_stage2.py" \
 --config "${CONFIG:-$SOURCE_ROOT/examples/pqfcv2/plan/stage2_baseline.yaml}" \
 --stage1 "${STAGE1:-$ROOT/weights/pqfcv2/dinov3/stage1_K2048_e8_postln_l100_splitlr/stage1_K2048_e8.npz}" \
 --backbone "${BACKBONE:-$ROOT/weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors}" \
 --feature-index "${ADE_FEATURES:-${ADE_INDEX:-$ROOT/features/pqfcv2/dinov3/ade}}" \
 --feature-index "${NYU_FEATURES:-${NYU_INDEX:-$ROOT/features/pqfcv2/dinov3/nyu}}" \
 --output-dir "${OUTPUT_DIR:-$ROOT/weights/pqfcv2/dinov3/stage2_N$N}" --latent-tokens "$N" "$@"

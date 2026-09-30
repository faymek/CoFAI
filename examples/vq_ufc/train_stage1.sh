#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_DIR="${FEATURE_DIR:-$PROJECT_ROOT/features/orfc_2446/dinov3/nyu_1w}"
RMS_PATH="${RMS_PATH:-$PROJECT_ROOT/weights/vq_ufc/dinov3/nyu_1w/prefix_position_patch_rms_s42.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/weights/vq_ufc/dinov3/nyu_1w/P05_PREFIX_K2048_D8_L10_lr0.005_wr1_prefixposrms_noclip_gn1}"

CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PYTHON_BIN" "$SCRIPT_DIR/dinov3/train_vq_ufc_dinov3.py" \
  --train-features "$FEATURE_DIR" --rms-path "$RMS_PATH" --output-dir "$OUTPUT_DIR" \
  --dataset-name nyu_1w --num-embeddings 2048 --embedding-dim 8 --num-chunks 32 \
  --lmbda 10 --commit-weight 0.25 --soft-rate-weight 1 --usage-weight 0 \
  --soft-temperature 1 --soft-temperature-min 0.1 --soft-temperature-decay 0.75 \
  --epochs "${EPOCHS:-6}" --batch-size 8 --lr 0.005 --val-ratio 0.05 --seed 42 \
  --use-soft-assignment --no-use-transform --grad-max-norm 1 \
  --max-train-files "${MAX_TRAIN_FILES:-0}" --max-val-files "${MAX_VAL_FILES:-0}"

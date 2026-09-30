#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 <nyu|ade> <20|48|72|96|120|144|192|240>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DOMAIN="$1"
LATENT_TOKENS="$2"

case "$DOMAIN" in
  nyu) DATASET="nyu_1w" ;;
  ade) DATASET="ade_1w" ;;
  *) echo "Domain must be nyu or ade" >&2; exit 2 ;;
esac
case "$LATENT_TOKENS" in
  20|48|72|96|120|144|192|240) ;;
  *) echo "Unsupported latent-token count: $LATENT_TOKENS" >&2; exit 2 ;;
esac

FEATURE_DIR="${FEATURE_DIR:-$PROJECT_ROOT/features/orfc_2446/dinov3/$DATASET}"
RMS_PATH="${RMS_PATH:-$PROJECT_ROOT/weights/vq_ufc/dinov3/$DATASET/prefix_position_patch_rms_s42.npz}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$PROJECT_ROOT/weights/vq_ufc/dinov3/nyu_1w/P05_PREFIX_K2048_D8_L10_lr0.005_wr1_prefixposrms_noclip_gn1/best.pth.tar}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/weights/vq_ufc/dinov3/$DATASET/stage2_N$LATENT_TOKENS}"

CUDA_VISIBLE_DEVICES="${GPU:-0}" "$PYTHON_BIN" "$SCRIPT_DIR/dinov3/train_vq_ufc_dinov3.py" \
  --train-features "$FEATURE_DIR" --rms-path "$RMS_PATH" --output-dir "$OUTPUT_DIR" \
  --dataset-name "$DATASET" --num-embeddings 2048 --embedding-dim 8 --num-chunks 32 \
  --lmbda 10 --commit-weight 0.25 --soft-rate-weight 1 --usage-weight 0 \
  --soft-temperature 1 --soft-temperature-min 0.1 --soft-temperature-decay 0.8 \
  --epochs "${EPOCHS:-8}" --batch-size 8 --lr 0.001 --val-ratio 0.05 --seed 42 \
  --use-soft-assignment --use-transform --transform-input-tokens 241 \
  --transform-tokens "$LATENT_TOKENS" --init-checkpoint "$BASE_CHECKPOINT" \
  --freeze-codebook --freeze-entropy --grad-max-norm 1

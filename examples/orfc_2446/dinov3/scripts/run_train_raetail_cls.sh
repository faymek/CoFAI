#!/bin/bash
# DINOv3-L RAEtail + CLS MSE SoftPQ training (mirrors orfc_2446 raetail_cls).
#
# Uses RAEv2 ViTXL decoder at weights/RAE/decoders/dinov3/large/decoder.pt
# Default: slot=24 (k=1), img_size=256, patch16, compress_reg.
#
# Usage:
#   bash examples/orfc_2446/dinov3/scripts/run_train_raetail_cls.sh
#   K=16 GPU=0 CLS_WEIGHT=1.0 bash .../run_train_raetail_cls.sh
#   # Isolate extreme DINOv3 reg tokens (default NORM_MODE):
#   NORM_MODE=split_reg_cls_patch PARALLEL=1 bash .../run_train_raetail_cls.sh
#   # Distortion-only (no rate term):
#   LMBDA=0 PARALLEL=1 GPU0=0 GPU1=1 bash .../run_train_raetail_cls.sh
#   # Parallel K=16 + K=256 on two GPUs:
#   PARALLEL=1 bash .../run_train_raetail_cls.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
unset PYTHONPATH

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"

PYTHON="${PYTHON:-poetry run python}"
TRAIN_SCRIPT="examples/orfc_2446/dinov3/offline/train_raetail_cls_dinov3.py"
LOG_DIR="examples/orfc_2446/dinov3/logs/train_raetail_cls"
mkdir -p "$LOG_DIR"

BACKBONE_CKPT="${BACKBONE_CKPT:-weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.pth}"
DECODER_PATH="${DECODER_PATH:-weights/RAE/decoders/dinov3/large/decoder.pt}"
PATHNAME_LIST="${PATHNAME_LIST:-/data4/workspace/zlt/featcodec/utils/imagenet_selected_pathname5000.txt}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/data4/workspace/zlt/featcodec/data/imagenet/images/val}"
# 256px slot24 cache (261 tokens). Override FEAT_CACHE=none to extract on the fly.
FEAT_CACHE="${FEAT_CACHE:-/data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_256px/slot24}"

SLOT="${SLOT:-24}"
IMG_SIZE="${IMG_SIZE:-256}"
PATCH_SIZE="${PATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
LMBDA="${LMBDA:-0.5}"
BS="${BS:-32}"
TRAIN="${TRAIN:-5000}"
CLS_WEIGHT="${CLS_WEIGHT:-1.0}"
EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
# per_image | split_cls_patch | split_reg_cls_patch | per_token_ln
# split_reg_cls_patch: reg alone vs cls+patch shared (avoids reg outlier domination)
NORM_MODE="${NORM_MODE:-split_reg_cls_patch}"
N_PREFIX="${N_PREFIX:-0}"
PARALLEL="${PARALLEL:-0}"

COMMON_ARGS=(
  --slot "$SLOT" --img_size "$IMG_SIZE" --patch_size "$PATCH_SIZE"
  --compress_reg --max_images "$TRAIN" --epochs "$EPOCHS"
  --embedding_dim "$EMBEDDING_DIM" --lmbda "$LMBDA"
  --tau_start 0.5 --tau_end 0.005 --lr "$LR" --batch_size "$BS"
  --backbone_ckpt "$BACKBONE_CKPT"
  --decoder_path "$DECODER_PATH"
  --pathname_list "$PATHNAME_LIST"
  --imagenet_root "$IMAGENET_ROOT"
  --feat_cache_dir "$FEAT_CACHE"
  --use_rae_tail --decoder_type ViTXL
  --cls_loss_weight "$CLS_WEIGHT"
  --norm_mode "$NORM_MODE" --n_prefix "$N_PREFIX"
)

echo "============================================================"
echo "  DINOv3-L RAEtail + CLS MSE"
echo "  slot=${SLOT}  ${IMG_SIZE}px/${PATCH_SIZE}  cls_w=${CLS_WEIGHT}"
echo "  norm_mode=${NORM_MODE}  n_prefix=${N_PREFIX}  lmbda=${LMBDA}"
echo "  decoder=${DECODER_PATH}"
echo "  Started: $(date)"
echo "============================================================"

run_one() {
  local K=$1
  local GPU=$2
  local norm_tag="${NORM_MODE}"
  local log="$LOG_DIR/train_raetail_decXL_cls${CLS_WEIGHT}_K${K}_lmbda${LMBDA}_${norm_tag}_opq.log"
  echo "[GPU ${GPU}] K=${K} lmbda=${LMBDA} → ${log}"
  CUDA_VISIBLE_DEVICES="$GPU" $PYTHON "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" --K "$K" --device cuda:0 \
    > "$log" 2>&1
}

if [[ "$PARALLEL" == "1" ]]; then
  run_one 16 "${GPU0:-0}" &
  pid0=$!
  run_one 256 "${GPU1:-1}" &
  pid1=$!
  fail=0
  wait $pid0 || fail=$((fail + 1))
  wait $pid1 || fail=$((fail + 1))
  echo "Done. fail=${fail}"
  exit $fail
fi

K="${K:-256}"
GPU="${GPU:-0}"
run_one "$K" "$GPU"
echo "Complete: $(date)"
ls -la "weights/orfc_2446_dinov3/dinov3_large_${IMG_SIZE}px/"*raetail* 2>/dev/null || true

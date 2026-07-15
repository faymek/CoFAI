#!/bin/bash
# DINOv3-L RAEtail L1 SoftPQ — patch tokens only (CLS+reg uncompressed).
#
# Mirrors run_train_patch_opq.sh token scope (train_tokens=patch) with RAEtail
# pixel L1 instead of FrozenTail delta_L_ref. No CLS MSE (prefix is bypassed).
#
# Usage:
#   bash examples/orfc_2446/dinov3/scripts/run_train_raetail_ptpatch.sh
#   K=16 GPU=0 LMBDA=0 bash .../run_train_raetail_ptpatch.sh
#   PARALLEL=1 GPU0=0 GPU1=1 LMBDA=0 bash .../run_train_raetail_ptpatch.sh

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
LOG_DIR="examples/orfc_2446/dinov3/logs/train_raetail_ptpatch"
mkdir -p "$LOG_DIR"

BACKBONE_CKPT="${BACKBONE_CKPT:-weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.pth}"
DECODER_PATH="${DECODER_PATH:-weights/RAE/decoders/dinov3/large/decoder.pt}"
PATHNAME_LIST="${PATHNAME_LIST:-/data4/workspace/zlt/featcodec/utils/imagenet_selected_pathname5000.txt}"
IMAGENET_ROOT="${IMAGENET_ROOT:-/data4/workspace/zlt/featcodec/data/imagenet/images/val}"
FEAT_CACHE="${FEAT_CACHE:-/data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_256px/slot24}"

SLOT="${SLOT:-24}"
IMG_SIZE="${IMG_SIZE:-256}"
PATCH_SIZE="${PATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
LMBDA="${LMBDA:-0.0}"
BS="${BS:-32}"
TRAIN="${TRAIN:-5000}"
# Patch-only: CLS MSE is a no-op (prefix bypassed); keep 0.
CLS_WEIGHT="${CLS_WEIGHT:-0.0}"
EMBEDDING_DIM="${EMBEDDING_DIM:-32}"
# Patch vs prefix stats (same default as run_train_patch_opq.sh).
NORM_MODE="${NORM_MODE:-split_cls_patch}"
N_PREFIX="${N_PREFIX:-0}"
TRAIN_TOKENS="${TRAIN_TOKENS:-patch}"
PARALLEL="${PARALLEL:-0}"

COMMON_ARGS=(
  --slot "$SLOT" --img_size "$IMG_SIZE" --patch_size "$PATCH_SIZE"
  --max_images "$TRAIN" --epochs "$EPOCHS"
  --embedding_dim "$EMBEDDING_DIM" --lmbda "$LMBDA"
  --tau_start 0.5 --tau_end 0.005 --lr "$LR" --batch_size "$BS"
  --backbone_ckpt "$BACKBONE_CKPT"
  --decoder_path "$DECODER_PATH"
  --pathname_list "$PATHNAME_LIST"
  --imagenet_root "$IMAGENET_ROOT"
  --feat_cache_dir "$FEAT_CACHE"
  --use_rae_tail --decoder_type ViTXL
  --cls_loss_weight "$CLS_WEIGHT"
  --train_tokens "$TRAIN_TOKENS"
  --norm_mode "$NORM_MODE" --n_prefix "$N_PREFIX"
)

echo "============================================================"
echo "  DINOv3-L RAEtail L1  (train_tokens=${TRAIN_TOKENS})"
echo "  slot=${SLOT}  ${IMG_SIZE}px/${PATCH_SIZE}  cls_w=${CLS_WEIGHT}"
echo "  norm_mode=${NORM_MODE}  n_prefix=${N_PREFIX}  lmbda=${LMBDA}"
echo "  decoder=${DECODER_PATH}"
echo "  Started: $(date)"
echo "============================================================"

run_one() {
  local K=$1
  local GPU=$2
  local log="$LOG_DIR/train_raetail_decXL_ptpatch_K${K}_lmbda${LMBDA}_${NORM_MODE}_opq.log"
  echo "[GPU ${GPU}] K=${K} lmbda=${LMBDA} train_tokens=${TRAIN_TOKENS} → ${log}"
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
ls -la "weights/orfc_2446_dinov3/dinov3_large_${IMG_SIZE}px/"*raetail*ptpatch* 2>/dev/null || true

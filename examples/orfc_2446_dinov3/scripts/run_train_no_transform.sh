#!/bin/bash
# No orthogonal R: k-means init + PQ codebooks only (K256 e16 baseline).
#
# Usage:
#   export PROJECT_ROOT=/data4/workspace/zlt/featcodec/CoFAI
#   bash examples/orfc_2446_dinov3/scripts/run_train_no_transform.sh
#
# Optional env:
#   GPU=4  EPOCHS=100  TRAIN=1000  BS=8  NORM=split_cls_patch

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$ROOT"

PYTHON="poetry run python"
TRAIN_SCRIPT="examples/orfc_2446_dinov3/offline/train_soft_pq_dinov3.py"
LOG_DIR="examples/orfc_2446_dinov3/logs/train_noR"
mkdir -p "$LOG_DIR"

GPU="${GPU:-4}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
TAU_S="${TAU_S:-0.5}"
TAU_E="${TAU_E:-0.005}"
BS="${BS:-8}"
TRAIN="${TRAIN:-1000}"
NVAL="${NVAL:-50}"
NORM="${NORM:-split_cls_patch}"
LMBDA="${LMBDA:-0.0}"

TAG="blk23_K256_e16_noR"
LOG="$LOG_DIR/${TAG}_ep${EPOCHS}.log"

echo ""
echo "========================================"
echo "  No-transform SoftPQ  K256 e16"
echo "  ep=${EPOCHS}  train=${TRAIN}  norm=${NORM}"
echo "  GPU=${GPU}  log=${LOG}"
echo "========================================"

CUDA_VISIBLE_DEVICES=$GPU \
$PYTHON "$TRAIN_SCRIPT" \
    --K 256 --embedding_dim 16 \
    --norm_mode "$NORM" \
    --no_transform \
    --max_train "$TRAIN" --epochs "$EPOCHS" --batch_size "$BS" --n_val "$NVAL" \
    --lr "$LR" --tau_start "$TAU_S" --tau_end "$TAU_E" \
    --lmbda "$LMBDA" \
    --gpu 0 \
    | tee "$LOG"

grep -q "noR_km" "$LOG" || { echo "missing noR_km in checkpoint name"; exit 1; }
grep -q "k-means in train_soft_pq" "$LOG" || { echo "missing k-means init log"; exit 1; }
! grep -q "OPQ warm-start" "$LOG" || { echo "unexpected OPQ warm-start"; exit 1; }

echo ""
echo "[done] $TAG  log=$LOG"

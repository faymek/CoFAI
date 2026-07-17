#!/bin/bash
# DINOv3 SoftPQ FrozenTail — generic parametric training entry.
#
# All knobs via env vars; no need for a new script per experiment.
#
# Usage:
#   # ADE 5k @ split_reg_cls_patch (default SoftPQ hyperparams)
#   FEAT_DIR=/data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_ade \
#   NORM_MODE=split_reg_cls_patch TRAIN=5000 TOKEN_HW=32,43 \
#   K=256 GPU=0 bash examples/orfc_2446/dinov3/scripts/run_train_frozen_tail.sh
#
#   # Parallel K=16 + K=256
#   FEAT_DIR=... NORM_MODE=split_reg_cls_patch TRAIN=5000 \
#   PARALLEL=1 GPU0=0 GPU1=1 bash .../run_train_frozen_tail.sh
#
#   # COCO (legacy defaults from run_train_pipeline)
#   FEAT_DIR=../ORFC/features/train/dinov3_vitl16_coco/blk23 \
#   NORM_MODE=split_cls_patch TRAIN=1000 TOKEN_HW=64,85 \
#   K=256 bash .../run_train_frozen_tail.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
unset PYTHONPATH

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"

PYTHON="${PYTHON:-poetry run python}"
TRAIN_SCRIPT="examples/orfc_2446/dinov3/offline/train_soft_pq_dinov3.py"
LOG_DIR="${LOG_DIR:-examples/orfc_2446/dinov3/logs/train_frozen_tail}"
mkdir -p "$LOG_DIR"

FEAT_DIR="${FEAT_DIR:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-weights/orfc_2446_dinov3}"
LAYER="${LAYER:-blk23}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
TAU_S="${TAU_S:-0.5}"
TAU_E="${TAU_E:-0.005}"
BS="${BS:-32}"
TRAIN="${TRAIN:-1000}"
NVAL="${NVAL:-50}"
NORM="${NORM_MODE:-${NORM:-split_cls_patch}}"
N_PREFIX="${N_PREFIX:-0}"
LMBDA="${LMBDA:-0.0}"
EMB="${EMB:-${EMBEDDING_DIM:-32}}"
SEED="${SEED:-42}"
TOKEN_HW="${TOKEN_HW:-}"
USE_TRANSFORM="${USE_TRANSFORM:-1}"
KEEP_PT="${KEEP_PT:-0}"
KMEANS_MAX="${KMEANS_MAX:-2000000}"
PARALLEL="${PARALLEL:-0}"

TRANSFORM_FLAG=()
if [[ "$USE_TRANSFORM" == "0" ]]; then
  TRANSFORM_FLAG=(--no_transform)
fi
KEEP_PT_FLAG=()
if [[ "$KEEP_PT" == "1" ]]; then
  KEEP_PT_FLAG=(--keep_pt)
fi
TOKEN_HW_FLAG=()
if [[ -n "$TOKEN_HW" ]]; then
  TOKEN_HW_FLAG=(--token_hw "$TOKEN_HW")
fi
FEAT_FLAG=()
if [[ -n "$FEAT_DIR" ]]; then
  FEAT_FLAG=(--feat_dir "$FEAT_DIR")
fi

COMMON_ARGS=(
  --layer "$LAYER"
  --embedding_dim "$EMB"
  --norm_mode "$NORM"
  --n_prefix "$N_PREFIX"
  --max_train "$TRAIN"
  --n_val "$NVAL"
  --epochs "$EPOCHS"
  --batch_size "$BS"
  --lr "$LR"
  --tau_start "$TAU_S"
  --tau_end "$TAU_E"
  --lmbda "$LMBDA"
  --seed "$SEED"
  --kmeans_max_samples "$KMEANS_MAX"
  --weights_dir "$WEIGHTS_DIR"
  "${FEAT_FLAG[@]}"
  "${TOKEN_HW_FLAG[@]}"
  "${TRANSFORM_FLAG[@]}"
  "${KEEP_PT_FLAG[@]}"
)

echo "============================================================"
echo "  DINOv3 SoftPQ FrozenTail"
echo "  layer=${LAYER}  K(will set)  emb=${EMB}  norm=${NORM}"
echo "  train=${TRAIN}  n_val=${NVAL}  ep=${EPOCHS}  bs=${BS}"
echo "  lmbda=${LMBDA}  lr=${LR}  tau=${TAU_S}->${TAU_E}"
echo "  feat_dir=${FEAT_DIR:-<(config default)>}"
echo "  token_hw=${TOKEN_HW:-<(auto/config)>}"
echo "  Started: $(date)"
echo "============================================================"

run_one() {
  local K=$1
  local GPU=$2
  local tag="${LAYER}_K${K}_e${EMB}_${NORM}_n${TRAIN}"
  local log="$LOG_DIR/${tag}.log"
  echo "[GPU ${GPU}] K=${K} → ${log}"
  CUDA_VISIBLE_DEVICES="$GPU" $PYTHON "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" --K "$K" --gpu 0 \
    > "$log" 2>&1
}

if [[ "$PARALLEL" == "1" ]]; then
  run_one "${K0:-16}" "${GPU0:-0}" &
  pid0=$!
  run_one "${K1:-256}" "${GPU1:-1}" &
  pid1=$!
  fail=0
  wait "$pid0" || fail=$((fail + 1))
  wait "$pid1" || fail=$((fail + 1))
  echo "Done. fail=${fail}"
  exit "$fail"
fi

K="${K:-256}"
GPU="${GPU:-0}"
run_one "$K" "$GPU"
echo "Complete: $(date)"
ls -la "$WEIGHTS_DIR"/*"${NORM}"*n${TRAIN}* 2>/dev/null || \
  ls -la "$WEIGHTS_DIR"/*.npz 2>/dev/null | tail -5 || true

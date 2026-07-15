#!/bin/bash
# Export train-prior .npz sidecars for DINOv3 RAEtail+CLS SoftPQ ckpts.
# Uses the same ImageNet 256px slot24 cache as training.
#
# Usage:
#   bash examples/orfc_2446/dinov3/scripts/run_export_raetail_npz.sh
#   GPU=0 FORCE=1 bash .../run_export_raetail_npz.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
unset PYTHONPATH

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"

PYTHON="${PYTHON:-poetry run python}"
EXPORT="$PYTHON examples/orfc_2446/dinov3/offline/export_npz_dinov3.py"
WD="weights/orfc_2446_dinov3/dinov3_large_256px"
FEAT_DIR="${FEAT_DIR:-/data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_256px/slot24}"
NORM="${NORM_MODE:-split_reg_cls_patch}"
GPU="${GPU:-0}"
MAX_TRAIN="${MAX_TRAIN:-5000}"
FORCE_FLAG=()
[[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG=(--force)

CKPTS=(
  "slot24_K16_e32_lmbda0.5_ep100_n5000_raetail_decXL_cls1.0_split_reg_cls_patch_wReg.pt"
  "slot24_K256_e32_lmbda0.5_ep100_n5000_raetail_decXL_cls1.0_split_reg_cls_patch_wReg.pt"
  "slot24_K16_e32_lmbda0.0_ep100_n5000_raetail_decXL_cls1.0_split_reg_cls_patch_wReg.pt"
  "slot24_K256_e32_lmbda0.0_ep100_n5000_raetail_decXL_cls1.0_split_reg_cls_patch_wReg.pt"
)

echo "Export NPZ  norm=${NORM}  feat=${FEAT_DIR}  gpu=${GPU}"
fail=0
for ckpt in "${CKPTS[@]}"; do
  path="$WD/$ckpt"
  if [[ ! -f "$path" ]]; then
    echo "[skip missing] $ckpt"
    continue
  fi
  echo "---- $ckpt ----"
  if ! $EXPORT \
      --ckpt_path "$path" \
      --feat_dir "$FEAT_DIR" \
      --norm_mode "$NORM" --n_prefix 0 \
      --max_train "$MAX_TRAIN" --batch_size 16 \
      --gpu "$GPU" "${FORCE_FLAG[@]}"; then
    fail=$((fail + 1))
  fi
done

echo ""
ls -lah "$WD"/*split_reg_cls_patch*.npz 2>/dev/null || echo "(no npz yet)"
exit "$fail"

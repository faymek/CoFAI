#!/bin/bash
# Download CTC DINOv3 data/weights, unzip datasets, normalize backbone path.
#
# Usage (from CoFAI root):
#   bash examples/pqfc/scripts/prepare_dinov3.sh
#   PYTHON="poetry run python" bash examples/pqfc/scripts/prepare_dinov3.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
cd "$COFAI_ROOT"
export PROJECT_ROOT="$COFAI_ROOT"
unset PYTHONPATH

PYTHON="${PYTHON:-poetry run python}"
MANIFEST="${MANIFEST:-examples/pqfc/manifests/dinov3-pqfc.manifest.txt}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "============================================================"
echo "  PQFC DINOv3 prepare"
echo "  ROOT: $COFAI_ROOT"
echo "============================================================"

if [ "$SKIP_DOWNLOAD" != "1" ]; then
  echo "[1/3] Download via cofai-download..."
  if command -v poetry >/dev/null 2>&1; then
    poetry run cofai-download "$MANIFEST"
  else
    $PYTHON -m cofai.utils.download "$MANIFEST"
  fi
else
  echo "[1/3] SKIP_DOWNLOAD=1 — skip cofai-download"
fi

echo "[2/3] Unzip datasets if needed..."
mkdir -p data weights/dinov3/backbone features/train/dinov3_vitl16

if [ -f data/ADE20K.zip ] && [ ! -d data/ADEChallengeData2016 ]; then
  echo "  unzip ADE20K.zip -> data/"
  unzip -q -o data/ADE20K.zip -d data/
fi
if [ -f data/NYU_subset_for_training_depth_head.zip ] && [ ! -d data/NYU ]; then
  echo "  unzip NYU zip -> data/"
  unzip -q -o data/NYU_subset_for_training_depth_head.zip -d data/
fi

# Reuse existing ORFC ImageNet train features if present
ORFC_FEAT="$(dirname "$COFAI_ROOT")/ORFC/features/train/dinov3_vitl16/blk23"
LOCAL_FEAT="features/train/dinov3_vitl16/blk23"
if [ ! -e "$LOCAL_FEAT" ] && [ -d "$ORFC_FEAT" ]; then
  ln -s "$ORFC_FEAT" "$LOCAL_FEAT"
  echo "  linked $LOCAL_FEAT -> $ORFC_FEAT"
fi

BB_DIR="weights/dinov3/backbone"
SAFE="$BB_DIR/dinov3_vitl16_pretrain_lvd1689m.safetensors"
PTH="$BB_DIR/dinov3_vitl16_pretrain_lvd1689m.pth"
if [ -f "$SAFE" ] && [ ! -e "$PTH" ]; then
  ln -s "$(basename "$SAFE")" "$PTH"
  echo "  linked $PTH -> $(basename "$SAFE")"
elif [ -f "$PTH" ] && [ ! -e "$SAFE" ]; then
  ln -s "$(basename "$PTH")" "$SAFE"
  echo "  linked $SAFE -> $(basename "$PTH")"
fi

echo "[3/3] Check required paths..."
ok=1
check() {
  if [ -e "$1" ]; then
    echo "  OK  $1"
  else
    echo "  MISSING  $1"
    ok=0
  fi
}
check data/ADEChallengeData2016
check data/NYU
check weights/dinov3/semseg_head/dinov3_vitl16_semseg_ade20k_linear_head.pth
check weights/dinov3/dpt_head/dinov3_vitl16_depth_nyuv2_linear_head.pth
if [ -e "$PTH" ] || [ -e "$SAFE" ]; then
  echo "  OK  backbone (.pth or .safetensors)"
else
  echo "  MISSING  backbone"
  ok=0
fi

echo ""
if [ "$ok" -eq 1 ]; then
  echo "Prepare done. Next:"
  echo "  bash examples/pqfc/scripts/extract_train.sh   # needs ImageNet val"
  echo "  bash examples/pqfc/scripts/extract_val.sh"
  echo "  bash examples/pqfc/scripts/train_dinov3.sh --K 256 --embedding_dim 32"
  echo "  bash examples/pqfc/scripts/replay_dinov3.sh --task semseg --ckpt_path <npz>"
else
  echo "Prepare incomplete — fix MISSING paths above."
  exit 1
fi

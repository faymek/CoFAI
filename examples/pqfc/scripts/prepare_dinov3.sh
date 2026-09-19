#!/usr/bin/env bash
# Download CTC DINOv3 data/weights, unzip datasets, normalize backbone path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_env.sh"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SOURCE_ROOT"

MANIFEST="${MANIFEST:-examples/pqfc/dinov3-pqfc.manifest.txt}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "============================================================"
echo "  PQFC DINOv3 prepare"
echo "  SOURCE_ROOT: $SOURCE_ROOT"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "============================================================"

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
  echo "[1/3] Download via cofai-download..."
  if command -v poetry >/dev/null 2>&1; then
    poetry -C "$PROJECT_ROOT" run cofai-download "$SOURCE_ROOT/$MANIFEST"
  else
    $PYTHON -m cofai.utils.download "$SOURCE_ROOT/$MANIFEST"
  fi
else
  echo "[1/3] SKIP_DOWNLOAD=1 — skip cofai-download"
fi

echo "[2/3] Unzip datasets if needed..."
mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/weights/dinov3/backbone" \
  "$PROJECT_ROOT/features/train/dinov3_vitl16"

if [[ -f "$PROJECT_ROOT/data/ADE20K.zip" && ! -d "$PROJECT_ROOT/data/ADEChallengeData2016" ]]; then
  unzip -q -o "$PROJECT_ROOT/data/ADE20K.zip" -d "$PROJECT_ROOT/data/"
fi
if [[ -f "$PROJECT_ROOT/data/NYU_subset_for_training_depth_head.zip" && ! -d "$PROJECT_ROOT/data/NYU" ]]; then
  unzip -q -o "$PROJECT_ROOT/data/NYU_subset_for_training_depth_head.zip" -d "$PROJECT_ROOT/data/"
fi

ORFC_FEAT="$(dirname "$PROJECT_ROOT")/ORFC/features/train/dinov3_vitl16/blk23"
LOCAL_FEAT="$PROJECT_ROOT/features/train/dinov3_vitl16/blk23"
if [[ ! -e "$LOCAL_FEAT" && -d "$ORFC_FEAT" ]]; then
  ln -s "$ORFC_FEAT" "$LOCAL_FEAT"
  echo "  linked $LOCAL_FEAT -> $ORFC_FEAT"
fi

BB_DIR="$PROJECT_ROOT/weights/dinov3/backbone"
SAFE="$BB_DIR/dinov3_vitl16_pretrain_lvd1689m.safetensors"
PTH="$BB_DIR/dinov3_vitl16_pretrain_lvd1689m.pth"
if [[ -f "$SAFE" && ! -e "$PTH" ]]; then
  ln -s "$(basename "$SAFE")" "$PTH"
elif [[ -f "$PTH" && ! -e "$SAFE" ]]; then
  ln -s "$(basename "$PTH")" "$SAFE"
fi

echo "[3/3] Check required paths..."
ok=1
check() {
  if [[ -e "$1" ]]; then
    echo "  OK  $1"
  else
    echo "  MISSING  $1"
    ok=0
  fi
}
check "$PROJECT_ROOT/data/ADEChallengeData2016"
check "$PROJECT_ROOT/data/NYU"
check "$PROJECT_ROOT/weights/dinov3/semseg_head/dinov3_vitl16_semseg_ade20k_linear_head.pth"
check "$PROJECT_ROOT/weights/dinov3/dpt_head/dinov3_vitl16_depth_nyuv2_linear_head.pth"
if [[ -e "$PTH" || -e "$SAFE" ]]; then
  echo "  OK  backbone (.pth or .safetensors)"
else
  echo "  MISSING  backbone"
  ok=0
fi

echo ""
if [[ "$ok" -eq 1 ]]; then
  echo "Prepare done. Next:"
  echo "  bash examples/pqfc/scripts/extract_train.sh"
  echo "  bash examples/pqfc/scripts/extract_val.sh"
  echo "  bash examples/pqfc/scripts/train_dinov3.sh --K 256 --embedding_dim 32"
else
  echo "Prepare incomplete — fix MISSING paths above."
  exit 1
fi

#!/usr/bin/env bash
# Download and unpack CTC DINOv3 data and codec weights.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_env.sh"
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

echo "[2/3] Unzip datasets and codec weights if needed..."
mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_ori"

if [[ -f "$PROJECT_ROOT/data/ADE20K.zip" && ! -d "$PROJECT_ROOT/data/ADEChallengeData2016" ]]; then
  unzip -q -o "$PROJECT_ROOT/data/ADE20K.zip" -d "$PROJECT_ROOT/data/"
fi
if [[ -f "$PROJECT_ROOT/data/NYU_subset_for_training_depth_head.zip" && ! -d "$PROJECT_ROOT/data/NYU" ]]; then
  unzip -q -o "$PROJECT_ROOT/data/NYU_subset_for_training_depth_head.zip" -d "$PROJECT_ROOT/data/"
fi

ORFC_ZIP="$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_ori.zip"
if [[ -f "$ORFC_ZIP" ]]; then
  unzip -q -j -n "$ORFC_ZIP" 'blk23_*.npz' -d "$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_ori/"
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
check "$PROJECT_ROOT/weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors"
check "$PROJECT_ROOT/weights/dinov3/semseg_head/dinov3_vitl16_semseg_ade20k_linear_head.pth"
check "$PROJECT_ROOT/weights/dinov3/dpt_head/dinov3_vitl16_depth_nyuv2_linear_head.pth"
if ! $PYTHON - <<'PY'
import os
from pathlib import Path

import yaml

root = os.environ["PROJECT_ROOT"]
paths = set()
for plan_path in (Path(os.environ["SOURCE_ROOT"]) / "examples/pqfc/plan/dinov3").glob("*__PQFC*.yaml"):
    plan = yaml.safe_load(plan_path.read_text())
    for config in [plan, *plan["multi_run"].values()]:
        paths.add(config["model"]["dino_codec"]["orfc_weights_path"].replace("${PROJECT_ROOT}", root))
missing = sorted(path for path in paths if not Path(path).is_file())
for path in missing:
    print(f"  MISSING  {path}")
if not missing:
    print(f"  OK  all {len(paths)} codec artifacts required by the plans")
raise SystemExit(bool(missing))
PY
then
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

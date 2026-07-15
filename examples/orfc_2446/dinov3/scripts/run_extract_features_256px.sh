#!/bin/bash
# Extract DINOv3-L/16 ImageNet-5k features at 256px, slot24.
set -euo pipefail
export PYTHONUNBUFFERED=1
unset PYTHONPATH

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"

OUT="${OUT:-/data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_256px/slot24}"
GPU="${GPU:-0}"
BS="${BS:-8}"
LOG_DIR="examples/orfc_2446/dinov3/logs/extract"
mkdir -p "$LOG_DIR" "$OUT"

echo "Extract → ${OUT}  GPU=${GPU}"
CUDA_VISIBLE_DEVICES="$GPU" poetry run python \
  examples/orfc_2446/dinov3/offline/extract_features_dinov3.py \
  --img_size 256 --slot 24 --batch_size "$BS" --device cuda:0 \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG_DIR/extract_dinov3_256px_slot24.log"

echo "Done. npy count: $(ls "$OUT"/*.npy 2>/dev/null | wc -l)"

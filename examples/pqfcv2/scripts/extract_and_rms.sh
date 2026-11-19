#!/usr/bin/env bash
# Extract the combined 20k DINOv3 slot24 features, then build the RMS artifact.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export PROJECT_ROOT PYTHONPATH="$SOURCE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FEATURE_ROOT="$PROJECT_ROOT/features/pqfcv2/dinov3"
LOG_DIR="$PROJECT_ROOT/logs/pqfcv2/extract"
mkdir -p "$LOG_DIR" "$FEATURE_ROOT/ade" "$FEATURE_ROOT/nyu"

GPUS=(${GPUS:-1 5 6 7})
NUM_SHARDS="${#GPUS[@]}"

COMMON=(
  --project-root "$PROJECT_ROOT"
  --ade-list "$SOURCE_ROOT/examples/vq_ufc/dinov3/data_lists/ade_1w.txt"
  --nyu-list "$SOURCE_ROOT/examples/vq_ufc/dinov3/data_lists/nyu_train_1w.txt"
  --ade-root "$PROJECT_ROOT/data/ADEChallengeData2016/images/training"
  --nyu-root "$PROJECT_ROOT/data/NYU"
  --ade-out "$FEATURE_ROOT/ade"
  --nyu-out "$FEATURE_ROOT/nyu"
  --num-shards "$NUM_SHARDS"
)

pids=()
for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  log="$LOG_DIR/gpu${gpu}.log"
  echo "launch shard $index on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$SOURCE_ROOT/examples/pqfcv2/extract_features.py" \
    "${COMMON[@]}" --shard-index "$index" --device cuda:0 \
    >"$log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "Feature extraction failed. See $LOG_DIR" >&2
  exit "$status"
fi

"$PYTHON_BIN" -u "$SOURCE_ROOT/examples/pqfcv2/prepare_rms.py" \
  --project-root "$PROJECT_ROOT" \
  --ade-list "$SOURCE_ROOT/examples/vq_ufc/dinov3/data_lists/ade_1w.txt" \
  --nyu-list "$SOURCE_ROOT/examples/vq_ufc/dinov3/data_lists/nyu_train_1w.txt" \
  --ade-root "$PROJECT_ROOT/data/ADEChallengeData2016/images/training" \
  --nyu-root "$PROJECT_ROOT/data/NYU" \
  --feature-root "$FEATURE_ROOT" \
  --ade-out "$FEATURE_ROOT/ade" \
  --nyu-out "$FEATURE_ROOT/nyu" \
  --output "$PROJECT_ROOT/weights/pqfcv2/dinov3/prefix_position_patch_rms_s42.npz" \
  --split-path "$FEATURE_ROOT/split.json" \
  --n-val 50 --seed 42

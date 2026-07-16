#!/bin/bash
# Replay / rate for PQFC DINOv3 (semseg | depth). Forwards args to run_pqfc_dinov3.py.
#
# Usage:
#   bash examples/pqfc/scripts/replay_dinov3.sh replay --task semseg \
#       --ckpt_path weights/pqfc/dinov3_vitl16/<ckpt>.npz
#   bash examples/pqfc/scripts/replay_dinov3.sh replay --task depth \
#       --ckpt_path weights/pqfc/dinov3_vitl16/<ckpt>.npz --gpu 0
#   bash examples/pqfc/scripts/replay_dinov3.sh rate --task semseg \
#       --ckpt_path weights/pqfc/dinov3_vitl16/<ckpt>.npz
#
# Shorthand (defaults to replay):
#   bash examples/pqfc/scripts/replay_dinov3.sh --task semseg --ckpt_path <npz>
#
# Env: PYTHON, GPU, CONFIG, LOG_DIR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
cd "$COFAI_ROOT"
export PROJECT_ROOT="$COFAI_ROOT"
unset PYTHONPATH
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-poetry run python}"
CONFIG="${CONFIG:-examples/pqfc/configs/dinov3_blk23.yaml}"
LOG_DIR="${LOG_DIR:-$PQFC_DIR/logs/replay_dinov3}"
mkdir -p "$LOG_DIR" "$PQFC_DIR/results/dinov3"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${GPU:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

# Default subcommand = replay when first arg is not replay/rate
CMD="replay"
if [ $# -ge 1 ] && { [ "$1" = "replay" ] || [ "$1" = "rate" ]; }; then
  CMD="$1"
  shift
fi

ARGS=(--config "$CONFIG" "$CMD")
HAS_GPU_ARG=0
for a in "$@"; do
  if [ "$a" = "--gpu" ]; then HAS_GPU_ARG=1; break; fi
done
if [ "$HAS_GPU_ARG" -eq 0 ]; then
  ARGS+=(--gpu 0)
fi

TAG="${CMD}_$(date +%Y%m%d_%H%M%S)"
prev=""
for a in "$@"; do
  if [ "$prev" = "--task" ]; then TAG="${CMD}_${a}_$(date +%H%M%S)"; fi
  if [ "$prev" = "--ckpt_path" ]; then
    stem="$(basename "$a")"
    stem="${stem%.npz}"
    TAG="${CMD}_${stem}"
  fi
  prev="$a"
done
LOG="$LOG_DIR/${TAG}.log"

echo "============================================================"
echo "  PQFC DINOv3 $CMD"
echo "  CONFIG: $CONFIG"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  args: ${ARGS[*]} $*"
echo "  log: $LOG"
echo "============================================================"

set -x
$PYTHON "$PQFC_DIR/run_pqfc_dinov3.py" "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"
set +x

echo "Done. Log: $LOG"

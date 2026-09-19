#!/usr/bin/env bash
# Replay / rate for PQFC DINOv3 (semseg | depth).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_env.sh"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SOURCE_ROOT"

CONFIG="${CONFIG:-$SOURCE_ROOT/examples/pqfc/configs/dinov3_blk23.yaml}"
LOG_DIR="${LOG_DIR:-$PQFC_DIR/logs/replay_dinov3}"
mkdir -p "$LOG_DIR"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

CMD="replay"
if [[ $# -ge 1 ]] && { [[ "$1" == "replay" ]] || [[ "$1" == "rate" ]]; }; then
  CMD="$1"
  shift
fi

ARGS=(--config "$CONFIG" "$CMD")
HAS_GPU_ARG=0
for a in "$@"; do
  if [[ "$a" == "--gpu" ]]; then HAS_GPU_ARG=1; break; fi
done
if [[ "$HAS_GPU_ARG" -eq 0 ]]; then
  ARGS+=(--gpu 0)
fi

TAG="${CMD}_$(date +%Y%m%d_%H%M%S)"
prev=""
for a in "$@"; do
  if [[ "$prev" == "--task" ]]; then TAG="${CMD}_${a}_$(date +%H%M%S)"; fi
  if [[ "$prev" == "--ckpt_path" ]]; then
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

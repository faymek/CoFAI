#!/usr/bin/env bash
# Train one PQFC DINOv3 job. All CLI args are forwarded to train_pqfc_dinov3.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/_env.sh"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SOURCE_ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-$SOURCE_ROOT/examples/pqfc/configs/dinov3_blk23.yaml}"
FEAT_DIR="${FEAT_DIR:-}"
WEIGHTS_DIR="${WEIGHTS_DIR:-}"
LOG_DIR="${LOG_DIR:-$PQFC_DIR/logs/train_dinov3}"
mkdir -p "$LOG_DIR"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

ARGS=(--config "$CONFIG")
if [[ -n "$FEAT_DIR" ]]; then
  ARGS+=(--feat_dir "$FEAT_DIR")
fi
if [[ -n "$WEIGHTS_DIR" ]]; then
  ARGS+=(--weights_dir "$WEIGHTS_DIR")
fi
HAS_GPU_ARG=0
for a in "$@"; do
  if [[ "$a" == "--gpu" ]]; then HAS_GPU_ARG=1; break; fi
done
if [[ "$HAS_GPU_ARG" -eq 0 ]]; then
  ARGS+=(--gpu 0)
fi

TAG="train_$(date +%Y%m%d_%H%M%S)"
K_TAG=""; E_TAG=""
prev=""
for a in "$@"; do
  if [[ "$prev" == "--K" ]]; then K_TAG="K${a}"; fi
  if [[ "$prev" == "--embedding_dim" ]]; then E_TAG="emb${a}"; fi
  prev="$a"
done
if [[ -n "$K_TAG$E_TAG" ]]; then
  TAG="${K_TAG}_${E_TAG}_$(date +%H%M%S)"
fi
LOG="$LOG_DIR/${TAG}.log"

echo "============================================================"
echo "  PQFC DINOv3 train"
echo "  SOURCE_ROOT: $SOURCE_ROOT"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  CONFIG: $CONFIG"
echo "  FEAT_DIR: ${FEAT_DIR:-<from config>}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  args: ${ARGS[*]} $*"
echo "  log: $LOG"
echo "============================================================"

set -x
$PYTHON "$PQFC_DIR/offline/train_pqfc_dinov3.py" "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"
set +x

echo "Done. Log: $LOG"

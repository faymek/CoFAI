#!/bin/bash
# DINOv3 ORFC offline replay diagnostic: semantic segmentation + depth.
#
# Formal reference evaluation is driven by ``cofai-eval`` plans. This script
# replays previously extracted features for offline diagnostics only.
#
# Usage:
#   # Evaluate specific checkpoints on both tasks
#   CKPTS="weights/orfc_2446/dinov3_vitl16/a.npz weights/orfc_2446/dinov3_vitl16/b.npz" \
#   TASKS=semseg,depth GPUS=0,1,2,3 \
#   bash examples/orfc_2446/dinov3/scripts/run_offline_replay.sh
#
#   # Single task / single GPU
#   CKPTS=weights/orfc_2446/dinov3_vitl16/foo.npz TASKS=semseg GPUS=0 \
#   bash examples/orfc_2446/dinov3/scripts/run_offline_replay.sh
#
#   # Override norm (default: read from the artifact)
#   NORM=split_reg_cls_patch CKPTS=... bash .../run_offline_replay.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COFAI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$COFAI_ROOT"

CFG="${CFG:-$COFAI_ROOT/examples/orfc_2446/dinov3/configs/dinov3_blk23.yaml}"
REPLAY_PY="$COFAI_ROOT/examples/orfc_2446/dinov3/offline/replay.py"
LOG_DIR="${LOG_DIR:-$COFAI_ROOT/logs/orfc_2446/dinov3/offline_replay}"
RESULTS_DIR="${RESULTS_DIR:-$COFAI_ROOT/logs/orfc_2446/dinov3/offline_replay/results}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

NORM="${NORM:-}"   # empty => auto from ckpt meta/filename
IFS=',' read -r -a TASK_ARR <<< "${TASKS:-semseg,depth}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"

if [[ -z "${CKPTS:-}" ]]; then
  echo "[ERROR] Set CKPTS to one or more ORFC .npz paths." >&2
  exit 1
fi
# shellcheck disable=SC2206
CKPT_ARR=( $CKPTS )

JOBS=()
for ckpt in "${CKPT_ARR[@]}"; do
  if [[ "$ckpt" != /* ]]; then
    ckpt="$COFAI_ROOT/$ckpt"
  fi
  if [[ "$ckpt" != *.npz ]]; then
    echo "[ERROR] offline replay accepts released .npz artifacts only: $ckpt" >&2
    exit 1
  fi
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] missing ckpt: $ckpt" >&2
    exit 1
  fi
  for task in "${TASK_ARR[@]}"; do
    JOBS+=("${task}:${ckpt}")
  done
done

echo "============================================================"
echo "  DINOv3 ORFC offline replay"
echo "  tasks=${TASK_ARR[*]}  ckpts=${#CKPT_ARR[@]}  jobs=${#JOBS[@]}"
echo "  gpus=${GPU_ARR[*]}  norm=${NORM:-auto}"
echo "  Started: $(date)"
echo "============================================================"

pids=()
tags=()
for i in "${!JOBS[@]}"; do
  IFS=':' read -r task ckpt <<< "${JOBS[$i]}"
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  stem="$(basename "${ckpt%.*}")"
  log="$LOG_DIR/${task}_${stem}.log"
  echo "[launch] task=$task gpu=$gpu ckpt=$ckpt → $log"
  NORM_ARGS=()
  if [[ -n "$NORM" ]]; then
    NORM_ARGS=(--norm_mode "$NORM")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  poetry -C "$COFAI_ROOT" run python "$REPLAY_PY" --config "$CFG" replay \
    --task "$task" --mode orfc \
    --ckpt_path "$ckpt" \
    --results_dir "$RESULTS_DIR" \
    "${NORM_ARGS[@]}" --gpu 0 \
    > "$log" 2>&1 &
  pids+=($!)
  tags+=("${task}:${stem}")
done

fail=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then
    echo "[done] ${tags[$k]}"
  else
    echo "[FAIL] ${tags[$k]}  (see $LOG_DIR)"
    fail=$((fail + 1))
  fi
done

echo ""
echo "========== SUMMARY =========="
poetry -C "$COFAI_ROOT" run python - "$RESULTS_DIR" "${CKPT_ARR[@]}" << 'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
stems = {Path(p).stem for p in sys.argv[2:]}
rows = []
for path in sorted(results_dir.glob("*.json")):
    # replay dumps like semseg_orfc_<stem>.json / depth_orfc_<stem>.json
    name = path.name
    hit = None
    for s in stems:
        if s in name:
            hit = s
            break
    if hit is None:
        continue
    with open(path) as f:
        d = json.load(f)
    m = d.get("metrics", {}) or {}
    rate = d.get("rate", {}) or {}
    rows.append((path.name, m, rate, d.get("avg_mse")))

if not rows:
    print("  (no matching result json yet)")
else:
    print(f"{'result':70s}  metric          BPFP")
    print("-" * 100)
    for name, m, rate, mse in rows:
        bpfp = rate.get("bpfp", "?")
        if "mIoU" in m:
            metric = f"mIoU={m['mIoU']:.4f}"
        elif "rmse" in m:
            metric = f"rmse={m['rmse']:.4f}"
        else:
            metric = str(m)
        print(f"{name:70s}  {metric:14s}  {bpfp}")
PY

echo "Logs: $LOG_DIR"
echo "Complete: $(date)"
exit "$fail"

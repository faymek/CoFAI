#!/bin/bash
# Unified DINOv3 SoftPQ offline eval: semantic segmentation + depth.
#
# Parametric entry (env vars). Prefers SoftPQ ``.npz``; falls back to sibling
# ``.pt`` only when ``.npz`` is missing (replay will still require PMF in practice).
#
# Usage:
#   # Evaluate specific checkpoints on both tasks
#   CKPTS="weights/orfc_2446_dinov3/a.npz weights/orfc_2446_dinov3/b.npz" \
#   TASKS=semseg,depth GPUS=0,1,2,3 \
#   bash examples/orfc_2446/dinov3/scripts/run_eval_tasks.sh
#
#   # Single task / single GPU
#   CKPTS=weights/orfc_2446_dinov3/foo.npz TASKS=semseg GPUS=0 \
#   bash examples/orfc_2446/dinov3/scripts/run_eval_tasks.sh
#
#   # Override norm (default: auto from npz meta / filename)
#   NORM=split_reg_cls_patch CKPTS=... bash .../run_eval_tasks.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
unset PYTHONPATH

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
export PROJECT_ROOT="$ROOT"

PYTHON="${PYTHON:-poetry run python}"
CFG="${CFG:-examples/orfc_2446/dinov3/configs/dinov3_blk23.yaml}"
REPLAY_PY="examples/orfc_2446/dinov3/run_orfc_dinov3.py"
LOG_DIR="${LOG_DIR:-examples/orfc_2446/dinov3/logs/eval_tasks}"
RESULTS_DIR="${RESULTS_DIR:-examples/orfc_2446/dinov3/results}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

NORM="${NORM:-}"   # empty => auto from ckpt meta/filename
IFS=',' read -r -a TASK_ARR <<< "${TASKS:-semseg,depth}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"

if [[ -z "${CKPTS:-}" ]]; then
  echo "[ERROR] Set CKPTS to one or more SoftPQ .npz (or .pt) paths." >&2
  exit 1
fi
# shellcheck disable=SC2206
CKPT_ARR=( $CKPTS )

resolve_ckpt() {
  local p=$1
  if [[ -f "$p" ]]; then
    echo "$p"
    return
  fi
  if [[ "$p" == *.pt && -f "${p%.pt}.npz" ]]; then
    echo "${p%.pt}.npz"
    return
  fi
  if [[ "$p" == *.npz && -f "${p%.npz}.pt" ]]; then
    echo "${p%.npz}.pt"
    return
  fi
  echo "$p"
}

JOBS=()
for ckpt in "${CKPT_ARR[@]}"; do
  ckpt="$(resolve_ckpt "$ckpt")"
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] missing ckpt: $ckpt" >&2
    exit 1
  fi
  for task in "${TASK_ARR[@]}"; do
    JOBS+=("${task}:${ckpt}")
  done
done

echo "============================================================"
echo "  DINOv3 SoftPQ task eval"
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
  $PYTHON "$REPLAY_PY" --config "$CFG" replay \
    --task "$task" --mode orfc \
    --ckpt_path "$ckpt" \
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
$PYTHON - "$RESULTS_DIR" "${CKPT_ARR[@]}" << 'PY'
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

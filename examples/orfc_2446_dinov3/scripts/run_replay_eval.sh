#!/bin/bash
# Replay semseg + depth for ep100 DINOv3 ORFC checkpoints (6 configs, GPU 3-5).
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$ROOT"

PYTHON="poetry run python"
REPLAY="$PYTHON examples/orfc_2446_dinov3/run_orfc_dinov3.py replay"
NORM="split_cls_patch"
WD="weights/orfc_2446_dinov3"
LOG_DIR="examples/orfc_2446_dinov3/logs/replay"
mkdir -p "$LOG_DIR"

# ep100 training sweep (e32 × 4 + e16 × 2)
CKPTS=(
  "blk23_K4_emb32_noR_km_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
  "blk23_K16_emb32_noR_km_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
  "blk23_K256_emb32_noR_km_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
  "blk23_K512_emb32_noR_km_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
  "blk23_K64_emb16_noR_km_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
  "blk23_K256_emb16_noR_km_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
)
GPUS=(0 1 2 3 4 5)

run_task() {
  local task=$1 gpu=$2 ckpt=$3
  local tag="${ckpt%.pt}"
  local log="$LOG_DIR/${task}_${tag}.log"
  echo "[launch] task=$task gpu=$gpu ckpt=$ckpt" >&2
  CUDA_VISIBLE_DEVICES=$gpu \
  $REPLAY --task "$task" --mode orfc \
    --ckpt_path "$WD/$ckpt" \
    --norm_mode "$NORM" --gpu 0 \
    > "$log" 2>&1 &
}

run_wave() {
  local task=$1 wave_name=$2 start=$3
  echo ""
  echo "========== $wave_name ($task) =========="
  local pids=()
  local tags=()
  for j in 0 1 2 3 4 5; do
    local i=$((start + j))
    if [ $i -ge ${#CKPTS[@]} ]; then
      break
    fi
    run_task "$task" "${GPUS[$j]}" "${CKPTS[$i]}"
    pids+=($!)
    tags+=("${CKPTS[$i]}")
  done
  local wave_fail=0
  for k in "${!pids[@]}"; do
    if wait "${pids[$k]}"; then
      echo "[done] $task ${tags[$k]}"
    else
      echo "[FAIL] $task ${tags[$k]}"
      wave_fail=$((wave_fail + 1))
    fi
  done
  return $wave_fail
}

TOTAL_FAIL=0
run_wave semseg "Wave1 (GPU 3-5)" 0 || TOTAL_FAIL=$((TOTAL_FAIL + $?))
# run_wave semseg "Wave2 (GPU 3-5)" 3 || TOTAL_FAIL=$((TOTAL_FAIL + $?))
run_wave depth  "Wave1 (GPU 3-5)" 0 || TOTAL_FAIL=$((TOTAL_FAIL + $?))
# run_wave depth  "Wave2 (GPU 3-5)" 3 || TOTAL_FAIL=$((TOTAL_FAIL + $?))

echo ""
echo "========== SUMMARY (ep100) =========="
$PYTHON - << 'PY'
import json
from pathlib import Path

results_dir = Path("examples/orfc_2446_dinov3/results")
for res in sorted(results_dir.glob("*orfc_blk23*ep100*.json")):
    with open(res) as f:
        d = json.load(f)
    m = d.get("metrics", {})
    rate = d.get("rate", {})
    bpfp = rate.get("bpfp", "?")
    mse = d.get("avg_mse", "?")
    if "mIoU" in m:
        print(f"{res.name:75s} mIoU={m['mIoU']:.4f} BPFP={bpfp} MSE={mse}")
    elif "rmse" in m:
        print(f"{res.name:75s} rmse={m['rmse']:.4f} BPFP={bpfp} MSE={mse}")
PY

echo "Logs: $LOG_DIR"
[ $TOTAL_FAIL -eq 0 ] && echo "All passed." || echo "$TOTAL_FAIL job(s) FAILED."
exit $TOTAL_FAIL

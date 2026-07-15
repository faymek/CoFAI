#!/bin/bash
# Replay semseg + depth for DINOv3-L RAEtail+CLS SoftPQ ckpts.
# Norm auto-inferred from ckpt filename/meta (split_reg_cls_patch for current
# training). Override with NORM=... if needed. Requires train-prior .npz.
#
# Usage:
#   bash examples/orfc_2446/dinov3/scripts/run_replay_raetail_cls.sh
#   NORM=split_reg_cls_patch LMBDA=0.5 GPUS="0,1,2,3" bash .../run_replay_raetail_cls.sh

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$ROOT"

PYTHON="${PYTHON:-poetry run python}"
CFG="examples/orfc_2446/dinov3/configs/dinov3_blk23.yaml"
REPLAY="$PYTHON examples/orfc_2446/dinov3/run_orfc_dinov3.py --config $CFG replay"
NORM="${NORM:-}"   # empty => auto from ckpt name/meta
LMBDA="${LMBDA:-0.5}"
WD="weights/orfc_2446_dinov3/dinov3_large_256px"
LOG_DIR="examples/orfc_2446/dinov3/logs/replay_raetail_cls"
RESULTS_DIR="examples/orfc_2446/dinov3/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

if [[ -n "${CKPTS:-}" ]]; then
  read -r -a CKPTS <<< "$CKPTS"
else
  CKPTS=(
    "slot24_K16_e32_lmbda${LMBDA}_ep100_n5000_raetail_decXL_cls1.0_split_reg_cls_patch_wReg.pt"
    "slot24_K256_e32_lmbda${LMBDA}_ep100_n5000_raetail_decXL_cls1.0_split_reg_cls_patch_wReg.pt"
  )
fi

IFS=',' read -r -a GPUS <<< "${GPUS:-0,1,2,3}"

JOBS=()
for ckpt in "${CKPTS[@]}"; do
  JOBS+=("semseg:$ckpt")
  JOBS+=("depth:$ckpt")
done

echo "============================================================"
echo "  RAEtail+CLS replay  norm=${NORM:-auto}  lmbda=${LMBDA}"
echo "  ckpts=${#CKPTS[@]}  jobs=${#JOBS[@]}  gpus=${GPUS[*]}"
echo "  Started: $(date)"
echo "============================================================"

pids=()
tags=()
for i in "${!JOBS[@]}"; do
  IFS=':' read -r task ckpt <<< "${JOBS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  tag="${ckpt%.pt}"
  log="$LOG_DIR/${task}_${tag}.log"
  echo "[launch] task=$task gpu=$gpu ckpt=$ckpt → $log"
  NORM_ARGS=()
  if [[ -n "$NORM" ]]; then
    NORM_ARGS=(--norm_mode "$NORM")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  $REPLAY --task "$task" --mode orfc \
    --ckpt_path "$WD/$ckpt" \
    "${NORM_ARGS[@]}" --require_sidecar --gpu 0 \
    > "$log" 2>&1 &
  pids+=($!)
  tags+=("$task:$tag")
done

fail=0
for k in "${!pids[@]}"; do
  if wait "${pids[$k]}"; then
    echo "[done] ${tags[$k]}"
  else
    echo "[FAIL] ${tags[$k]}"
    fail=$((fail + 1))
  fi
done

echo ""
echo "========== SUMMARY (raetail_cls) =========="
$PYTHON - << 'PY'
import json
from pathlib import Path

results_dir = Path("examples/orfc_2446/dinov3/results")
for res in sorted(results_dir.glob("*orfc_slot24_K*_raetail_decXL_cls1.0*split_reg*")):
    with open(res) as f:
        d = json.load(f)
    m = d.get("metrics", {})
    rate = d.get("rate", {}) or {}
    bpfp = rate.get("bpfp", "?")
    kind = rate.get("rate_kind", "?")
    pmf = rate.get("pmf_source", "?")
    rans = rate.get("rans_bpt", "?")
    mse = d.get("avg_mse", "?")
    print(res.name)
    if "mIoU" in m:
        print(
            f"  mIoU={m['mIoU']:.4f}  BPFP={bpfp}  rans_bpt={rans}  "
            f"rate={kind}  pmf={pmf}  MSE={mse}"
        )
    elif "rmse" in m:
        print(
            f"  rmse={m['rmse']:.4f}  abs_rel={m.get('abs_rel', float('nan')):.4f}  "
            f"a1={m.get('a1', float('nan')):.4f}  BPFP={bpfp}  rans_bpt={rans}  "
            f"rate={kind}  pmf={pmf}  MSE={mse}"
        )
PY

echo "Logs: $LOG_DIR"
echo "Finished: $(date)"
[ "$fail" -eq 0 ] && echo "All passed." || echo "$fail job(s) FAILED."
exit "$fail"

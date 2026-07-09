#!/bin/bash
# Replay semseg + depth for patch-only (_ptpatch) checkpoints.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$ROOT"

PYTHON="poetry run python"
REPLAY="$PYTHON examples/orfc_2446_dinov3/run_orfc_dinov3.py replay"
NORM="${NORM:-split_cls_patch}"
WD="weights/orfc_2446_dinov3"
LOG_DIR="examples/orfc_2446_dinov3/logs/replay_ptpatch"
mkdir -p "$LOG_DIR"

CKPTS=(
  "blk23_K4_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K16_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K256_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K512_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K64_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
)
GPUS=(${GPUS:-3 4 5})

run_task() {
  local task=$1 gpu=$2 ckpt=$3 extra_flags=${4:-}
  local tag="${ckpt%.pt}"
  local log="$LOG_DIR/${task}_${tag}.log"
  echo "[launch] task=$task gpu=$gpu ckpt=$ckpt $extra_flags" >&2
  CUDA_VISIBLE_DEVICES=$gpu \
  $REPLAY --task "$task" --mode orfc \
    --ckpt_path "$WD/$ckpt" \
    --norm_mode "$NORM" --gpu 0 \
    $extra_flags \
    > "$log" 2>&1 &
}

run_wave() {
  local task=$1 wave_name=$2 start=$3 extra_flags=${4:-}
  echo ""
  echo "========== $wave_name ($task) =========="
  local pids=() tags=()
  for j in 0 1 2; do
    local i=$((start + j))
    [ $i -ge ${#CKPTS[@]} ] && break
    run_task "$task" "${GPUS[$j]}" "${CKPTS[$i]}" "$extra_flags"
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
run_wave semseg "Wave1 semseg" 0 "--prefix_bypass" || TOTAL_FAIL=$((TOTAL_FAIL + $?))
run_wave semseg "Wave2 semseg" 3 "--prefix_bypass" || TOTAL_FAIL=$((TOTAL_FAIL + $?))
run_wave depth  "Wave1 depth"  0 "--prefix_bypass" || TOTAL_FAIL=$((TOTAL_FAIL + $?))
run_wave depth  "Wave2 depth"  3 "--prefix_bypass" || TOTAL_FAIL=$((TOTAL_FAIL + $?))

# Ablation: K256 e16 without prefix_bypass
echo ""
echo "========== Ablation K256 e16 no bypass =========="
ABL_CKPT="blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
CUDA_VISIBLE_DEVICES=${GPUS[0]} \
$REPLAY --task semseg --mode orfc \
  --ckpt_path "$WD/$ABL_CKPT" \
  --norm_mode "$NORM" --gpu 0 --no_prefix_bypass \
  > "$LOG_DIR/semseg_${ABL_CKPT%.pt}_no_bypass.log" 2>&1 \
  && echo "[done] ablation no bypass" \
  || TOTAL_FAIL=$((TOTAL_FAIL + 1))

echo ""
echo "========== SUMMARY (ptpatch) =========="
$PYTHON - << 'PY'
import json
from pathlib import Path

results_dir = Path("examples/orfc_2446_dinov3/results")
rows = []
for res in sorted(results_dir.glob("*ptpatch*.json")):
    with open(res) as f:
        d = json.load(f)
    m = d.get("metrics", {})
    rate = d.get("rate", {})
    bpfp = rate.get("bpfp", "?")
    pb = d.get("prefix_bypass", "?")
    if "mIoU" in m:
        rows.append((res.name, m["mIoU"], bpfp, pb))
    elif "rmse" in m:
        rows.append((res.name, m["rmse"], bpfp, pb))

# baseline ep100 (no ptpatch)
for res in sorted(results_dir.glob("*ep100*n1000_s42.json")):
    if "ptpatch" in res.name or "reg_cls" in res.name:
        continue
    with open(res) as f:
        d = json.load(f)
    if d.get("task") != "semseg":
        continue
    m = d.get("metrics", {})
    rate = d.get("rate", {})
    if "mIoU" in m:
        rows.append((res.name + " [baseline]", m["mIoU"], rate.get("bpfp", "?"), "all"))

print(f"{'result':70s}  metric    BPFP     bypass")
print("-" * 100)
for name, metric, bpfp, pb in sorted(rows, key=lambda x: x[0]):
    label = "mIoU" if "depth" not in name else "rmse"
    if "baseline" in name or "semseg" in name:
        label = "mIoU"
    print(f"{name:70s}  {metric:.4f}  {bpfp}  {pb}")
PY

echo "Logs: $LOG_DIR"
[ $TOTAL_FAIL -eq 0 ] && echo "All passed." || echo "$TOTAL_FAIL job(s) FAILED."
exit $TOTAL_FAIL

#!/bin/bash
# Replay semseg + depth for PQFC DINOv3 blk23 checkpoints (8 configs × 2 tasks).
# Parallel on 4 GPUs (default 3,4,5,7).
#
# Usage:
#   bash examples/pqfc/scripts/replay_dinov3.sh
#   PYTHON=.venv/bin/python GPU_IDS=3,4,5,7 \
#       bash examples/pqfc/scripts/replay_dinov3.sh
set -euo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
cd "$COFAI_ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$COFAI_ROOT"

PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-3,4,5,7}"
NORM="${NORM:-split_cls_patch}"
WD="${WD:-weights/pqfc/dinov3_vitl16}"
LOG_DIR="$PQFC_DIR/logs/replay_dinov3"
RESULTS_DIR="$PQFC_DIR/results/dinov3"
mkdir -p "$LOG_DIR"

CKPTS=(
  "blk23_K256_emb32_bt1024_ws_lmbda0.5_split_cls_patch_tau0.5_lr0.0003_ep100_n5000_s42.pt"
  "blk23_K256_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n5000_s42.pt"
  "blk23_K256_emb32_noR_km_lmbda0.5_split_cls_patch_lr0.0003_ep100_n5000_s42.pt"
  "blk23_K256_emb32_noR_km_split_cls_patch_lr0.0003_ep100_n5000_s42.pt"
  "blk23_K256_emb16_bt1024_ws_lmbda0.5_split_cls_patch_tau0.5_lr0.0003_ep100_n5000_s42.pt"
  "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n5000_s42.pt"
  "blk23_K256_emb16_noR_km_lmbda0.5_split_cls_patch_lr0.0003_ep100_n5000_s42.pt"
  "blk23_K256_emb16_noR_km_split_cls_patch_lr0.0003_ep100_n5000_s42.pt"
)

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 1 ]; then
    echo "ERROR: empty GPU_IDS"
    exit 1
fi

# Flat job list: "task ckpt"
JOBS=()
for ckpt in "${CKPTS[@]}"; do
    if [ ! -f "$WD/$ckpt" ]; then
        echo "WARNING: missing ckpt, skip: $WD/$ckpt"
        continue
    fi
    JOBS+=("semseg $ckpt")
    JOBS+=("depth $ckpt")
done

TOTAL=${#JOBS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: no jobs (check $WD)"
    exit 1
fi

echo "============================================================"
echo "  PQFC DINOv3 replay — $TOTAL jobs (semseg+depth × ckpts)"
echo "  GPUs: ${GPU_IDS} ($NUM_GPUS)"
echo "  WD:   $WD"
echo "  Logs: $LOG_DIR"
echo "  Started: $(date)"
echo "============================================================"

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do GPU_PIDS[$g]=0; done
COMPLETED=0
FAILED=0

wait_for_gpu() {
    while true; do
        for ((g=0; g<NUM_GPUS; g++)); do
            local pid=${GPU_PIDS[$g]}
            if [ "$pid" -eq 0 ]; then
                echo $g
                return
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
                GPU_PIDS[$g]=0
                echo $g
                return
            fi
        done
        sleep 2
    done
}

run_job() {
    local gpu_id=$1
    local task=$2
    local ckpt=$3
    local tag="${ckpt%.pt}"
    local log="$LOG_DIR/${task}_${tag}.log"
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$PQFC_DIR/run_pqfc_dinov3.py" replay \
        --task "$task" \
        --mode orfc \
        --ckpt_path "$WD/$ckpt" \
        --norm_mode "$NORM" \
        --gpu 0 \
        > "$log" 2>&1
}

for ((i=0; i<TOTAL; i++)); do
    read -r task ckpt <<< "${JOBS[$i]}"
    gpu_idx=$(wait_for_gpu)
    gpu_id=${GPUS[$gpu_idx]}
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: ${task}/${ckpt}"
    run_job "$gpu_id" "$task" "$ckpt" &
    GPU_PIDS[$gpu_idx]=$!
done

for ((g=0; g<NUM_GPUS; g++)); do
    local_pid=${GPU_PIDS[$g]}
    if [ "$local_pid" -ne 0 ]; then
        wait "$local_pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
        GPU_PIDS[$g]=0
    fi
done

echo ""
echo "============================================================"
echo "  Replay done: ok=$COMPLETED / $TOTAL  fail=$FAILED"
echo "  Logs: $LOG_DIR"
echo "  Results: $RESULTS_DIR"
echo "  Finished: $(date)"
echo "============================================================"

echo ""
echo "  Summary:"
$PYTHON - << 'PY'
import json
from pathlib import Path

results_dir = Path("examples/pqfc/results/dinov3")
if not results_dir.is_dir():
    print("  (no results dir yet)")
    raise SystemExit(0)

rows = []
for res in sorted(results_dir.glob("*.json")):
    try:
        with open(res) as f:
            d = json.load(f)
    except Exception:
        continue
    m = d.get("metrics", {})
    rate = d.get("rate", {}) or {}
    bpfp = rate.get("bpfp", "?")
    mse = d.get("avg_mse", "?")
    if "mIoU" in m:
        rows.append(f"  {res.name}: mIoU={m['mIoU']:.4f}  BPFP={bpfp}  MSE={mse}")
    elif "rmse" in m:
        rows.append(f"  {res.name}: rmse={m['rmse']:.4f}  BPFP={bpfp}  MSE={mse}")

if rows:
    print("\n".join(rows))
else:
    print("  (no matching result json)")
PY

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  Failed logs:"
    for ((i=0; i<TOTAL; i++)); do
        read -r task ckpt <<< "${JOBS[$i]}"
        tag="${ckpt%.pt}"
        log="$LOG_DIR/${task}_${tag}.log"
        if [ -f "$log" ] && ! grep -qE 'Done in|mIoU|rmse|BPFP' "$log" 2>/dev/null; then
            echo "    - ${task}/${tag}  ($log)"
        fi
    done
    exit 1
fi

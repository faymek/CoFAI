#!/bin/bash
# Offline cls + seg evaluation for ViT-L/14 K=4 weights (train_vitl14_k4.sh).
#
# Weights: weights/orfc_2446/dinov2_vitl14/*.npz
# Jobs: 4 blocks x 2 tasks = 8, scheduled on 4 GPUs in parallel.
#
# Usage:
#   bash examples/orfc_2446/dinov2/scripts/test_vitl14_k4_all.sh
#   GPU_IDS=0,1,2,3 PYTHON=.venv/bin/python \
#       bash examples/orfc_2446/dinov2/scripts/test_vitl14_k4_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORFC2446_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$ORFC2446_DIR/offline"
COFAI_ROOT="$(dirname "$(dirname "$(dirname "$ORFC2446_DIR")")")"
FEATCODEC_ROOT="$(dirname "$COFAI_ROOT")"

FEAT_ROOT="${FEAT_ROOT:-$FEATCODEC_ROOT/features}"
SEG_FEAT_ROOT="${SEG_FEAT_ROOT:-$FEATCODEC_ROOT/features/voc2012_100}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/orfc_2446/dinov2_vitl14}"
PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

LOG_DIR="$ORFC2446_DIR/logs/test_vitl14_k4"
mkdir -p "$LOG_DIR/cls" "$LOG_DIR/seg"

BACKBONE="dinov2_vitl14"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

# ================================================================
# Discover checkpoints (blk05/10/15/20, K4)
# Job format: "task layer ckpt_path"
# ================================================================

JOBS=()

if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "ERROR: weights dir not found: $WEIGHTS_DIR"
    exit 1
fi

for ckpt in "$WEIGHTS_DIR"/*.npz; do
    [ -f "$ckpt" ] || continue
    fname=$(basename "$ckpt" .npz)
    layer=$(echo "$fname" | grep -oE '^blk[0-9]+' || true)
    if [ -z "$layer" ]; then
        continue
    fi
    if ! echo "$fname" | grep -q '_K4_emb32_'; then
        continue
    fi
    JOBS+=("cls $layer $ckpt")
    JOBS+=("seg $layer $ckpt")
done

TOTAL=${#JOBS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: no K4 checkpoints found in $WEIGHTS_DIR"
    exit 1
fi

echo "============================================================"
echo "  Soft-PQ Offline Test — ViT-L/14 K=4 (cls + seg)"
echo "  Total jobs: $TOTAL"
echo "  GPUs: ${GPU_IDS} ($NUM_GPUS)"
echo "  FEAT_ROOT: $FEAT_ROOT"
echo "  SEG_FEAT_ROOT: $SEG_FEAT_ROOT"
echo "  WEIGHTS_DIR: $WEIGHTS_DIR"
echo "  Logs: $LOG_DIR"
echo "  Started: $(date)"
echo "============================================================"

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

COMPLETED=0
FAILED=0

run_job() {
    local gpu_id=$1
    local task=$2
    local layer=$3
    local ckpt=$4
    local tag
    tag=$(basename "$ckpt" .npz)
    local log="$LOG_DIR/${task}/${tag}.log"

    if [ "$task" = "cls" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_cls.py" \
            --backbone "$BACKBONE" \
            --layer "$layer" \
            --ckpt_path "$ckpt" \
            --feat_root "$FEAT_ROOT" \
            > "$log" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_seg.py" \
            --backbone "$BACKBONE" \
            --layer "$layer" \
            --ckpt_path "$ckpt" \
            --feat_root "$FEAT_ROOT" \
            --seg_feat_root "$SEG_FEAT_ROOT" \
            > "$log" 2>&1
    fi
}

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
        sleep 1
    done
}

for ((i=0; i<TOTAL; i++)); do
    read -r task layer ckpt <<< "${JOBS[$i]}"
    tag=$(basename "$ckpt" .npz)
    gpu_idx=$(wait_for_gpu)
    gpu_id=${GPUS[$gpu_idx]}

    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: ${task}/${tag}"
    run_job "$gpu_id" "$task" "$layer" "$ckpt" &
    GPU_PIDS[$gpu_idx]=$!
done

for ((g=0; g<NUM_GPUS; g++)); do
    local_pid=${GPU_PIDS[$g]}
    if [ "$local_pid" -ne 0 ]; then
        wait "$local_pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
    fi
done

echo ""
echo "============================================================"
echo "  Offline Test Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Logs: $LOG_DIR"
echo "  Results JSON: $COFAI_ROOT/results/orfc_2446/$BACKBONE/"
echo "  Finished: $(date)"
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  Failed jobs:"
    for ((i=0; i<TOTAL; i++)); do
        read -r task layer ckpt <<< "${JOBS[$i]}"
        tag=$(basename "$ckpt" .npz)
        log="$LOG_DIR/${task}/${tag}.log"
        if [ -f "$log" ] && grep -qiE 'error|traceback|failed' "$log" 2>/dev/null; then
            echo "    - ${task}/${tag}  (see $log)"
        fi
    done
fi

echo ""
echo "  Results summary:"
for ((i=0; i<TOTAL; i++)); do
    read -r task layer ckpt <<< "${JOBS[$i]}"
    tag=$(basename "$ckpt" .npz)
    log="$LOG_DIR/${task}/${tag}.log"
    if [ ! -f "$log" ]; then
        continue
    fi
    if [ "$task" = "cls" ]; then
        acc=$(grep -oE 'Accuracy = [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $3}' || echo "?")
        echo "    [cls] $tag: Acc=${acc}"
    else
        miou=$(grep -oE 'mIoU = [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $3}' || echo "?")
        echo "    [seg] $tag: mIoU=${miou}"
    fi
done

#!/bin/bash
# Run segmentation evaluation for all DINOv2 ORFC configurations.
# Loads pre-trained weights and evaluates VOC2012 mIoU + rate.
# Uses 4 GPUs in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORFC_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$ORFC_DIR/offline"
PYTHON="${PYTHON:-python}"

LOG_DIR="$ORFC_DIR/logs/test_seg"
mkdir -p "$LOG_DIR"

NUM_GPUS="${NUM_GPUS:-4}"

# ================================================================
# Job definitions: "backbone layer K embedding_dim"
# DINOv2 only (segmentation not supported for CLIP)
# ================================================================

JOBS=()

# --- DINOv2 ViT-L/14 ---
for layer in blk05 blk10 blk15; do
    for cfg in "4 32" "8 32" "16 32" "64 32" "256 32" "64 16" "256 16"; do
        JOBS+=("dinov2_vitl14 $layer $cfg")
    done
done
for cfg in "4 32" "8 32" "16 32" "32 32" "64 32" "256 32" "256 16"; do
    JOBS+=("dinov2_vitl14 blk20 $cfg")
done

# --- DINOv2 ViT-G/14 ---
for layer in blk09 blk19 blk29; do
    for cfg in "4 32" "8 32" "16 32" "64 32" "256 32" "64 16"; do
        JOBS+=("dinov2_vitg14 $layer $cfg")
    done
done

TOTAL=${#JOBS[@]}

echo "============================================================"
echo "  ORFC Segmentation Test — DINOv2 Configurations"
echo "  Total jobs: $TOTAL"
echo "  GPUs: $NUM_GPUS"
echo "  Started: $(date)"
echo "============================================================"

# ================================================================
# GPU scheduler
# ================================================================

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

COMPLETED=0
FAILED=0

run_job() {
    local gpu_id=$1
    local backbone=$2
    local layer=$3
    local K=$4
    local emb=$5
    local tag="${backbone}_${layer}_K${K}_e${emb}"
    local log="$LOG_DIR/${tag}.log"

    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_seg.py" \
        --backbone "$backbone" --layer "$layer" \
        --K "$K" --embedding_dim "$emb" \
        > "$log" 2>&1
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
    read -r backbone layer K emb <<< "${JOBS[$i]}"
    tag="${backbone}_${layer}_K${K}_e${emb}"

    gpu_id=$(wait_for_gpu)

    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: $tag"
    run_job "$gpu_id" "$backbone" "$layer" "$K" "$emb" &
    GPU_PIDS[$gpu_id]=$!
done

for ((g=0; g<NUM_GPUS; g++)); do
    local_pid=${GPU_PIDS[$g]}
    if [ "$local_pid" -ne 0 ]; then
        wait "$local_pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
    fi
done

echo ""
echo "============================================================"
echo "  Segmentation Test Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Finished: $(date)"
echo "============================================================"

# Print summary
echo ""
echo "  Results summary:"
for ((i=0; i<TOTAL; i++)); do
    read -r backbone layer K emb <<< "${JOBS[$i]}"
    tag="${backbone}_${layer}_K${K}_e${emb}"
    log="$LOG_DIR/${tag}.log"
    if [ -f "$log" ]; then
        miou=$(grep -oP 'mIoU = \K[\d.]+' "$log" 2>/dev/null | tail -1 || echo "?")
        echo "    $tag: mIoU=${miou}"
    fi
done

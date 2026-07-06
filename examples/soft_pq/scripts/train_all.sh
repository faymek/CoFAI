#!/bin/bash
# Train all Soft-PQ configurations using multiple GPUs in parallel.
# Saves .pt checkpoints to CoFAI/weights/orfc_2446/{backbone}/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOFTPQ_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$SOFTPQ_DIR")")"

FEAT_ROOT="${FEAT_ROOT:-$COFAI_ROOT/features/orfc}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/orfc_2446}"
PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"

LOG_DIR="$SOFTPQ_DIR/logs/train"
mkdir -p "$LOG_DIR"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
if [ ${#GPUS[@]} -lt $NUM_GPUS ]; then
    NUM_GPUS=${#GPUS[@]}
fi

# ================================================================
# Job definitions: "backbone layer K embedding_dim"
# ================================================================
JOBS=()

# DINOv2 ViT-L/14
for layer in blk05 blk10 blk15 blk20; do
    JOBS+=("dinov2_vitl14 $layer 4 32")
    JOBS+=("dinov2_vitl14 $layer 8 32")
    JOBS+=("dinov2_vitl14 $layer 16 32")
    JOBS+=("dinov2_vitl14 $layer 64 32")
    JOBS+=("dinov2_vitl14 $layer 256 32")
    JOBS+=("dinov2_vitl14 $layer 256 16")
done

# DINOv2 ViT-G/14
for layer in blk09 blk19 blk29; do
    JOBS+=("dinov2_vitg14 $layer 4 32")
    JOBS+=("dinov2_vitg14 $layer 8 32")
    JOBS+=("dinov2_vitg14 $layer 16 32")
    JOBS+=("dinov2_vitg14 $layer 64 32")
    JOBS+=("dinov2_vitg14 $layer 256 32")
done

TOTAL=${#JOBS[@]}

echo "============================================================"
echo "  Soft-PQ Training — All Configurations"
echo "  Total jobs: $TOTAL"
echo "  GPUs: ${GPU_IDS} ($NUM_GPUS parallel)"
echo "  Features: $FEAT_ROOT"
echo "  Weights:  $WEIGHTS_DIR"
echo "  Started:  $(date)"
echo "============================================================"

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

COMPLETED=0
FAILED=0

run_job() {
    local gpu_idx=$1
    local backbone=$2
    local layer=$3
    local K=$4
    local emb=$5
    local gpu_id=${GPUS[$gpu_idx]}
    local tag="${backbone}_${layer}_K${K}_emb${emb}"
    local log="$LOG_DIR/${tag}.log"

    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$SOFTPQ_DIR/offline/train_soft_pq.py" \
        --backbone "$backbone" --layer "$layer" \
        --K "$K" --embedding_dim "$emb" \
        --feat_root "$FEAT_ROOT" \
        --weights_dir "$WEIGHTS_DIR" \
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
    tag="${backbone}_${layer}_K${K}_emb${emb}"
    gpu_idx=$(wait_for_gpu)
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU${GPUS[$gpu_idx]}: $tag"
    run_job "$gpu_idx" "$backbone" "$layer" "$K" "$emb" &
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
echo "  Training Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Finished: $(date)"
echo "============================================================"

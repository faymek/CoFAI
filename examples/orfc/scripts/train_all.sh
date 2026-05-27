#!/bin/bash
# Train all ORFC configurations using 8 GPUs in parallel.
# Saves weights (R + codebooks + PMF) to CoFAI/weights/orfc/{backbone}/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORFC_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$ORFC_DIR/offline"
COFAI_ROOT="$(dirname "$(dirname "$ORFC_DIR")")"

FEAT_ROOT="${FEAT_ROOT:-$COFAI_ROOT/features/orfc}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/orfc}"
PYTHON="${PYTHON:-python}"

LOG_DIR="$ORFC_DIR/logs/train"
mkdir -p "$LOG_DIR"

NUM_GPUS="${NUM_GPUS:-4}"

# ================================================================
# Job definitions: "backbone layer K embedding_dim"
# ================================================================

JOBS=()

# --- CLIP ViT-L/14 ---
# blk05
JOBS+=("clip_vitl14 blk05 4 32")
JOBS+=("clip_vitl14 blk05 8 32")
JOBS+=("clip_vitl14 blk05 16 32")
JOBS+=("clip_vitl14 blk05 64 32")
JOBS+=("clip_vitl14 blk05 256 32")
JOBS+=("clip_vitl14 blk05 64 16")
JOBS+=("clip_vitl14 blk05 256 16")
# blk10
JOBS+=("clip_vitl14 blk10 4 32")
JOBS+=("clip_vitl14 blk10 8 32")
JOBS+=("clip_vitl14 blk10 16 32")
JOBS+=("clip_vitl14 blk10 64 32")
JOBS+=("clip_vitl14 blk10 256 32")
JOBS+=("clip_vitl14 blk10 64 16")
JOBS+=("clip_vitl14 blk10 256 16")
# blk15
JOBS+=("clip_vitl14 blk15 4 32")
JOBS+=("clip_vitl14 blk15 8 32")
JOBS+=("clip_vitl14 blk15 16 32")
JOBS+=("clip_vitl14 blk15 64 32")
JOBS+=("clip_vitl14 blk15 512 32")
JOBS+=("clip_vitl14 blk15 256 16")
JOBS+=("clip_vitl14 blk15 512 16")
# blk20
JOBS+=("clip_vitl14 blk20 4 32")
JOBS+=("clip_vitl14 blk20 8 32")
JOBS+=("clip_vitl14 blk20 16 32")
JOBS+=("clip_vitl14 blk20 64 32")
JOBS+=("clip_vitl14 blk20 512 32")
JOBS+=("clip_vitl14 blk20 512 16")
JOBS+=("clip_vitl14 blk20 256 8")

# --- DINOv2 ViT-L/14 ---
# blk05
JOBS+=("dinov2_vitl14 blk05 4 32")
JOBS+=("dinov2_vitl14 blk05 8 32")
JOBS+=("dinov2_vitl14 blk05 16 32")
JOBS+=("dinov2_vitl14 blk05 64 32")
JOBS+=("dinov2_vitl14 blk05 256 32")
JOBS+=("dinov2_vitl14 blk05 64 16")
JOBS+=("dinov2_vitl14 blk05 256 16")
# blk10
JOBS+=("dinov2_vitl14 blk10 4 32")
JOBS+=("dinov2_vitl14 blk10 8 32")
JOBS+=("dinov2_vitl14 blk10 16 32")
JOBS+=("dinov2_vitl14 blk10 64 32")
JOBS+=("dinov2_vitl14 blk10 256 32")
JOBS+=("dinov2_vitl14 blk10 64 16")
JOBS+=("dinov2_vitl14 blk10 256 16")
# blk15
JOBS+=("dinov2_vitl14 blk15 4 32")
JOBS+=("dinov2_vitl14 blk15 8 32")
JOBS+=("dinov2_vitl14 blk15 16 32")
JOBS+=("dinov2_vitl14 blk15 64 32")
JOBS+=("dinov2_vitl14 blk15 256 32")
JOBS+=("dinov2_vitl14 blk15 64 16")
JOBS+=("dinov2_vitl14 blk15 256 16")
# blk20
JOBS+=("dinov2_vitl14 blk20 4 32")
JOBS+=("dinov2_vitl14 blk20 8 32")
JOBS+=("dinov2_vitl14 blk20 16 32")
JOBS+=("dinov2_vitl14 blk20 32 32")
JOBS+=("dinov2_vitl14 blk20 64 32")
JOBS+=("dinov2_vitl14 blk20 256 32")
JOBS+=("dinov2_vitl14 blk20 256 16")

# --- DINOv2 ViT-G/14 ---
# blk09
JOBS+=("dinov2_vitg14 blk09 4 32")
JOBS+=("dinov2_vitg14 blk09 8 32")
JOBS+=("dinov2_vitg14 blk09 16 32")
JOBS+=("dinov2_vitg14 blk09 64 32")
JOBS+=("dinov2_vitg14 blk09 256 32")
JOBS+=("dinov2_vitg14 blk09 64 16")
# blk19
JOBS+=("dinov2_vitg14 blk19 4 32")
JOBS+=("dinov2_vitg14 blk19 8 32")
JOBS+=("dinov2_vitg14 blk19 16 32")
JOBS+=("dinov2_vitg14 blk19 64 32")
JOBS+=("dinov2_vitg14 blk19 256 32")
JOBS+=("dinov2_vitg14 blk19 64 16")
# blk29
JOBS+=("dinov2_vitg14 blk29 4 32")
JOBS+=("dinov2_vitg14 blk29 8 32")
JOBS+=("dinov2_vitg14 blk29 16 32")
JOBS+=("dinov2_vitg14 blk29 64 32")
JOBS+=("dinov2_vitg14 blk29 256 32")
JOBS+=("dinov2_vitg14 blk29 64 16")

TOTAL=${#JOBS[@]}

echo "============================================================"
echo "  ORFC Training — All Configurations (with PMF)"
echo "  Total jobs: $TOTAL"
echo "  GPUs: $NUM_GPUS (parallel)"
echo "  Features: $FEAT_ROOT"
echo "  Weights:  $WEIGHTS_DIR"
echo "  Env:      $CONDA_ENV"
echo "  Started:  $(date)"
echo "============================================================"

# ================================================================
# GPU job scheduler: maintain NUM_GPUS concurrent jobs
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

    cd "$OFFLINE_DIR" && \
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON train_orfc.py \
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
    tag="${backbone}_${layer}_K${K}_e${emb}"

    gpu_id=$(wait_for_gpu)

    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: $tag"
    run_job "$gpu_id" "$backbone" "$layer" "$K" "$emb" &
    GPU_PIDS[$gpu_id]=$!
done

# Wait for remaining jobs
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

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  Failed jobs (check logs in $LOG_DIR):"
    for ((i=0; i<TOTAL; i++)); do
        read -r backbone layer K emb <<< "${JOBS[$i]}"
        tag="${backbone}_${layer}_K${K}_e${emb}"
        log="$LOG_DIR/${tag}.log"
        if [ -f "$log" ] && ! grep -q "Weights saved" "$log" 2>/dev/null; then
            echo "    - $tag"
        fi
    done
fi

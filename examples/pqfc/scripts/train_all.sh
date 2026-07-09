#!/bin/bash
# Train PQFC configurations (DINOv2) with multi-GPU parallel.
# Default: --use_transform (soft PQ, orfc_2446-aligned).
# Pass EXTRA_ARGS='--no_transform' for hard PQ without R.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$PQFC_DIR/offline"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"

FEAT_ROOT="${FEAT_ROOT:-$COFAI_ROOT/features/orfc}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/pqfc}"
PYTHON="${PYTHON:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export PROJECT_ROOT="$COFAI_ROOT"

# Drop unbuilt CompressAI source tree from PYTHONPATH (shadows pip compressai → no _CXX).
if [ -n "${PYTHONPATH:-}" ]; then
    CLEANED=""
    IFS=':' read -ra _PP_PARTS <<< "$PYTHONPATH"
    for _p in "${_PP_PARTS[@]}"; do
        case "$_p" in
            *ORFC/coding/CompressAI*|*coding/CompressAI*) continue ;;
            *) CLEANED="${CLEANED:+${CLEANED}:}${_p}" ;;
        esac
    done
    if [ -n "$CLEANED" ]; then
        export PYTHONPATH="$CLEANED"
    else
        unset PYTHONPATH
    fi
fi

LOG_DIR="$PQFC_DIR/logs/train"
mkdir -p "$LOG_DIR"

NUM_GPUS="${NUM_GPUS:-4}"

# Job: "backbone layer K emb bt lmbda"
JOBS=()

# DINOv2 ViT-L/14
for layer in blk05 blk10 blk15 blk20; do
    JOBS+=("dinov2_vitl14 $layer 4 32 1024 0.5")
    JOBS+=("dinov2_vitl14 $layer 8 32 1024 0.5")
    JOBS+=("dinov2_vitl14 $layer 16 32 1024 0.5")
    JOBS+=("dinov2_vitl14 $layer 64 32 1024 0.5")
    JOBS+=("dinov2_vitl14 $layer 256 32 1024 0.5")
done

# DINOv2 ViT-G/14
for layer in blk09 blk19 blk29; do
    JOBS+=("dinov2_vitg14 $layer 4 32 1536 0.5")
    JOBS+=("dinov2_vitg14 $layer 8 32 1536 0.5")
    JOBS+=("dinov2_vitg14 $layer 16 32 1536 0.5")
    JOBS+=("dinov2_vitg14 $layer 64 32 1536 0.5")
    JOBS+=("dinov2_vitg14 $layer 256 32 1536 0.5")
done

TOTAL=${#JOBS[@]}

echo "============================================================"
echo "  PQFC Training — DINOv2"
echo "  Total jobs: $TOTAL"
echo "  GPUs: $NUM_GPUS"
echo "  Features: $FEAT_ROOT"
echo "  Weights:  $WEIGHTS_DIR"
echo "  EXTRA_ARGS: $EXTRA_ARGS"
echo "  Started:  $(date)"
echo "============================================================"

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
    local bt=$6
    local lmbda=$7
    local tag="${backbone}_${layer}_K${K}_emb${emb}_bt${bt}_lmbda${lmbda}"
    local log="$LOG_DIR/${tag}.log"

    cd "$OFFLINE_DIR" && \
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON train_pqfc.py \
        --backbone "$backbone" --layer "$layer" \
        --K "$K" --embedding_dim "$emb" \
        --bottleneck_dim "$bt" \
        --lmbda "$lmbda" \
        --feat_root "$FEAT_ROOT" \
        --weights_dir "$WEIGHTS_DIR" \
        $EXTRA_ARGS \
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
    read -r backbone layer K emb bt lmbda <<< "${JOBS[$i]}"
    tag="${backbone}_${layer}_K${K}_emb${emb}_bt${bt}_lmbda${lmbda}"
    gpu_id=$(wait_for_gpu)
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: $tag"
    run_job "$gpu_id" "$backbone" "$layer" "$K" "$emb" "$bt" "$lmbda" &
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
echo "  Training Complete: $COMPLETED / $TOTAL  Failed: $FAILED"
echo "  Finished: $(date)"
echo "============================================================"

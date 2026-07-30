#!/bin/bash
# Train Soft-PQ K=4 for all DINOv2 ViT-L/14 blocks (blk05/10/15/20).
#
# Feature layout:
#   ${FEAT_ROOT}/train/dinov2_vitl14/blk{05,10,15,20}/*.npy
# Default FEAT_ROOT points to featcodec/features (not features/orfc).
#
# Hyperparameters match Ours entries in soft_pq_reproduction_results.csv:
#   blk05/blk20: lmbda=0.0, lr=5e-4, ep=300
#   blk10/blk15: lmbda=0.5, lr=3e-4, ep=100
#
# Usage:
#   bash examples/orfc_2446/dinov2/scripts/train_vitl14_k4.sh
#   NUM_GPUS=4 \
#       bash examples/orfc_2446/dinov2/scripts/train_vitl14_k4.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COFAI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

ORFC2446_DIR="$COFAI_ROOT/examples/orfc_2446/dinov2"
OFFLINE_DIR="$ORFC2446_DIR/offline"

FEAT_ROOT="${FEAT_ROOT:-$COFAI_ROOT/features}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/orfc_2446}"
NUM_GPUS="${NUM_GPUS:-4}"

LOG_DIR="${LOG_DIR:-$COFAI_ROOT/logs/orfc_2446/dinov2/train_vitl14_k4}"
mkdir -p "$LOG_DIR"

# Job format: "layer lmbda lr epochs"
JOBS=(
    "blk05 0.0 0.0005 300"
    "blk10 0.5 0.0003 100"
    "blk15 0.5 0.0003 100"
    "blk20 0.0 0.0005 300"
)

BACKBONE="dinov2_vitl14"
K=4
EMB=32
BT=1024
TAU=0.5

TOTAL=${#JOBS[@]}

echo "============================================================"
echo "  Soft-PQ Training — ViT-L/14 K=4 (all blocks)"
echo "  Total jobs: $TOTAL"
echo "  GPUs: $NUM_GPUS"
echo "  FEAT_ROOT: $FEAT_ROOT"
echo "  WEIGHTS_DIR: $WEIGHTS_DIR"
echo "  Started: $(date)"
echo "============================================================"

for layer in blk05 blk10 blk15 blk20; do
    feat_dir="$FEAT_ROOT/train/$BACKBONE/$layer"
    if [ ! -d "$feat_dir" ]; then
        echo "ERROR: feature dir not found: $feat_dir"
        exit 1
    fi
    n_feat=$(find "$feat_dir" -maxdepth 1 -name '*.npy' | wc -l)
    echo "  $layer: $n_feat .npy files in $feat_dir"
done
echo ""

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

COMPLETED=0
FAILED=0

run_job() {
    local gpu_id=$1
    local layer=$2
    local lmbda=$3
    local lr=$4
    local epochs=$5

    local rate_tag=""
    if [ "$lmbda" != "0.0" ]; then
        rate_tag="_lmbda${lmbda}"
    fi
    local tag="${BACKBONE}_${layer}_K${K}_emb${EMB}_bt${BT}_ws${rate_tag}_tau${TAU}_lr${lr}_ep${epochs}"
    local log="$LOG_DIR/${tag}.log"

    CUDA_VISIBLE_DEVICES=$gpu_id poetry -C "$COFAI_ROOT" run python "$OFFLINE_DIR/train_soft_pq.py" \
        --backbone "$BACKBONE" \
        --layer "$layer" \
        --K "$K" \
        --embedding_dim "$EMB" \
        --bottleneck_dim "$BT" \
        --lmbda "$lmbda" \
        --tau_start "$TAU" \
        --lr "$lr" \
        --epochs "$epochs" \
        --warm_start_opq \
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
    read -r layer lmbda lr epochs <<< "${JOBS[$i]}"
    rate_tag=""
    if [ "$lmbda" != "0.0" ]; then
        rate_tag="_lmbda${lmbda}"
    fi
    tag="${BACKBONE}_${layer}_K${K}_emb${EMB}_bt${BT}_ws${rate_tag}_tau${TAU}_lr${lr}_ep${epochs}"

    gpu_id=$(wait_for_gpu)
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: $tag"
    run_job "$gpu_id" "$layer" "$lmbda" "$lr" "$epochs" &
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
echo "  Training Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Logs: $LOG_DIR"
echo "  Weights: $WEIGHTS_DIR/$BACKBONE/"
echo "  Finished: $(date)"
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  Failed jobs:"
    for ((i=0; i<TOTAL; i++)); do
        read -r layer lmbda lr epochs <<< "${JOBS[$i]}"
        rate_tag=""
        if [ "$lmbda" != "0.0" ]; then
            rate_tag="_lmbda${lmbda}"
        fi
        tag="${BACKBONE}_${layer}_K${K}_emb${EMB}_bt${BT}_ws${rate_tag}_tau${TAU}_lr${lr}_ep${epochs}"
        log="$LOG_DIR/${tag}.log"
        if [ -f "$log" ] && ! grep -q "Checkpoint saved" "$log" 2>/dev/null; then
            echo "    - $tag  (see $log)"
        fi
    done
fi

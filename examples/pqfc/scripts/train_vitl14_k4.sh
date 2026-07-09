#!/bin/bash
# Train PQFC K=4 for all DINOv2 ViT-L/14 blocks.
# Default: use_transform + soft PQ. Use EXTRA_ARGS='--no_transform' for hard PQ.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$PQFC_DIR/offline"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
FEATCODEC_ROOT="$(dirname "$COFAI_ROOT")"

FEAT_ROOT="${FEAT_ROOT:-$FEATCODEC_ROOT/features}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/pqfc}"
PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-4}"
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

LOG_DIR="$PQFC_DIR/logs/train_vitl14_k4"
mkdir -p "$LOG_DIR"

JOBS=(
    "blk05 0.5 0.0003 100"
    "blk10 0.5 0.0003 100"
    "blk15 0.5 0.0003 100"
    "blk20 0.5 0.0003 100"
)

BACKBONE="dinov2_vitl14"
K=4
EMB=32
BT=1024
TOTAL=${#JOBS[@]}

echo "============================================================"
echo "  PQFC Training — ViT-L/14 K=4"
echo "  FEAT_ROOT: $FEAT_ROOT"
echo "  WEIGHTS_DIR: $WEIGHTS_DIR"
echo "  EXTRA_ARGS: $EXTRA_ARGS"
echo "============================================================"

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do GPU_PIDS[$g]=0; done
COMPLETED=0
FAILED=0

run_job() {
    local gpu_id=$1 layer=$2 lmbda=$3 lr=$4 epochs=$5
    local tag="${BACKBONE}_${layer}_K${K}_emb${EMB}_lmbda${lmbda}"
    local log="$LOG_DIR/${tag}.log"
    cd "$OFFLINE_DIR" && \
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON train_pqfc.py \
        --backbone "$BACKBONE" --layer "$layer" \
        --K "$K" --embedding_dim "$EMB" --bottleneck_dim "$BT" \
        --lmbda "$lmbda" --lr "$lr" --epochs "$epochs" \
        --feat_root "$FEAT_ROOT" --weights_dir "$WEIGHTS_DIR" \
        $EXTRA_ARGS > "$log" 2>&1
}

wait_for_gpu() {
    while true; do
        for ((g=0; g<NUM_GPUS; g++)); do
            local pid=${GPU_PIDS[$g]}
            if [ "$pid" -eq 0 ]; then echo $g; return; fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
                GPU_PIDS[$g]=0; echo $g; return
            fi
        done
        sleep 1
    done
}

for ((i=0; i<TOTAL; i++)); do
    read -r layer lmbda lr epochs <<< "${JOBS[$i]}"
    gpu_id=$(wait_for_gpu)
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: $layer"
    run_job "$gpu_id" "$layer" "$lmbda" "$lr" "$epochs" &
    GPU_PIDS[$gpu_id]=$!
done

for ((g=0; g<NUM_GPUS; g++)); do
    local_pid=${GPU_PIDS[$g]}
    if [ "$local_pid" -ne 0 ]; then
        wait "$local_pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
    fi
done

echo "Done: $COMPLETED / $TOTAL  Failed: $FAILED"

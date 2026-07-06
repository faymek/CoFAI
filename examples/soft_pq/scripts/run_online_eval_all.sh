#!/bin/bash
# Run online evaluation for all Soft-PQ configurations using engine pipeline.
# Sweeps all backbone/layer/task combinations with multi-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOFTPQ_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$SOFTPQ_DIR")")"

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
OUTPUT_BASE="${OUTPUT_DIR:-$COFAI_ROOT/eval_results/soft_pq_online}"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
if [ ${#GPUS[@]} -lt $NUM_GPUS ]; then
    NUM_GPUS=${#GPUS[@]}
fi

mkdir -p "$OUTPUT_BASE"

# Build job list: "backbone layer task"
JOBS=()
# vitl14
for layer in blk05 blk10 blk15 blk20; do
    JOBS+=("dinov2_vitl14 $layer cls")
    JOBS+=("dinov2_vitl14 $layer seg")
done
# vitg14
for layer in blk09 blk19 blk29; do
    JOBS+=("dinov2_vitg14 $layer cls")
    JOBS+=("dinov2_vitg14 $layer seg")
done

TOTAL=${#JOBS[@]}

echo "============================================================"
echo "  Soft-PQ Online Evaluation — All Configurations"
echo "  Total jobs: $TOTAL"
echo "  GPUs: ${GPU_IDS} ($NUM_GPUS parallel)"
echo "  Output: $OUTPUT_BASE"
echo "  Started: $(date)"
echo "============================================================"

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

COMPLETED=0
FAILED=0

LOG_DIR="$SOFTPQ_DIR/logs/online_eval"
mkdir -p "$LOG_DIR"

run_job() {
    local gpu_idx=$1
    local backbone=$2
    local layer=$3
    local task=$4
    local gpu_id=${GPUS[$gpu_idx]}
    local tag="${backbone}_${layer}_${task}"
    local log="$LOG_DIR/${tag}.log"
    local out_dir="$OUTPUT_BASE/${tag}"

    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$SOFTPQ_DIR/run_eval_soft_pq.py" \
        --backbone "$backbone" --layer "$layer" --task "$task" \
        --multi-run --cuda \
        --output_dir "$out_dir" \
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
    read -r backbone layer task <<< "${JOBS[$i]}"
    tag="${backbone}_${layer}_${task}"
    gpu_idx=$(wait_for_gpu)
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU${GPUS[$gpu_idx]}: $tag"
    run_job "$gpu_idx" "$backbone" "$layer" "$task" &
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
echo "  Online Evaluation Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Finished: $(date)"
echo "============================================================"

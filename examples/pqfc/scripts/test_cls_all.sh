#!/bin/bash
# Batch classification test for all PQFC DINOv2 checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$PQFC_DIR/offline"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
PYTHON="${PYTHON:-python}"

WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/pqfc}"
LOG_DIR="$PQFC_DIR/logs/test_cls"
mkdir -p "$LOG_DIR"
NUM_GPUS="${NUM_GPUS:-4}"

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

JOBS=()
for backbone in dinov2_vitl14 dinov2_vitg14; do
    ckpt_dir="$WEIGHTS_DIR/$backbone"
    [ -d "$ckpt_dir" ] || continue
    for ckpt in "$ckpt_dir"/*.pt; do
        [ -f "$ckpt" ] || continue
        fname=$(basename "$ckpt" .pt)
        layer=$(echo "$fname" | grep -oP '^blk\d+' || true)
        [ -n "$layer" ] && JOBS+=("$backbone $layer $ckpt")
    done
done

TOTAL=${#JOBS[@]}
[ "$TOTAL" -gt 0 ] || { echo "No checkpoints in $WEIGHTS_DIR"; exit 1; }

echo "PQFC Classification Test — $TOTAL jobs, GPUs=$NUM_GPUS"

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do GPU_PIDS[$g]=0; done
COMPLETED=0
FAILED=0

run_job() {
    local gpu_id=$1 backbone=$2 layer=$3 ckpt=$4
    local tag=$(basename "$ckpt" .pt)
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_cls.py" \
        --backbone "$backbone" --layer "$layer" --ckpt_path "$ckpt" \
        > "$LOG_DIR/${tag}.log" 2>&1
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
    read -r backbone layer ckpt <<< "${JOBS[$i]}"
    gpu_id=$(wait_for_gpu)
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: $(basename "$ckpt")"
    run_job "$gpu_id" "$backbone" "$layer" "$ckpt" &
    GPU_PIDS[$gpu_id]=$!
done

for ((g=0; g<NUM_GPUS; g++)); do
    local_pid=${GPU_PIDS[$g]}
    if [ "$local_pid" -ne 0 ]; then
        wait "$local_pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
    fi
done

echo "Done: $COMPLETED / $TOTAL  Failed: $FAILED"

#!/bin/bash
# Run classification evaluation for all Soft-PQ configurations.
# Loads trained codec checkpoints and evaluates accuracy + rate.
# Uses multiple GPUs in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORFC2446_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$ORFC2446_DIR/offline"
COFAI_ROOT="$(dirname "$(dirname "$(dirname "$ORFC2446_DIR")")")"
PYTHON="${PYTHON:-python}"

WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/orfc_2446}"
LOG_DIR="$ORFC2446_DIR/logs/test_cls"
mkdir -p "$LOG_DIR"

NUM_GPUS="${NUM_GPUS:-4}"

# ================================================================
# Auto-discover checkpoints and generate jobs
# Format: "backbone layer ckpt_path"
# ================================================================

JOBS=()

for backbone in dinov2_vitl14 dinov2_vitg14; do
    case "$backbone" in
        dinov2_vitl14) weights_subdir="dinov2_vitl14_ori" ;;
        dinov2_vitg14) weights_subdir="dinov2_vitg14_ori" ;;
        *) weights_subdir="$backbone" ;;
    esac
    ckpt_dir="$WEIGHTS_DIR/$weights_subdir"
    if [ ! -d "$ckpt_dir" ]; then
        continue
    fi
    for ckpt in "$ckpt_dir"/*.npz; do
        [ -f "$ckpt" ] || continue
        fname=$(basename "$ckpt" .npz)
        # Extract layer from filename (first field before _K)
        layer=$(echo "$fname" | grep -oP '^blk\d+')
        if [ -n "$layer" ]; then
            JOBS+=("$backbone $layer $ckpt")
        fi
    done
done

TOTAL=${#JOBS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "No checkpoints found in $WEIGHTS_DIR"
    exit 1
fi

echo "============================================================"
echo "  Soft-PQ Classification Test — All Configurations"
echo "  Total jobs: $TOTAL"
echo "  GPUs: $NUM_GPUS"
echo "  Weights: $WEIGHTS_DIR"
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
    local ckpt=$4
    local tag=$(basename "$ckpt" .npz)
    local log="$LOG_DIR/${tag}.log"

    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_cls.py" \
        --backbone "$backbone" --layer "$layer" \
        --ckpt_path "$ckpt" \
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
    read -r backbone layer ckpt <<< "${JOBS[$i]}"
    tag=$(basename "$ckpt" .npz)

    gpu_id=$(wait_for_gpu)

    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: $tag"
    run_job "$gpu_id" "$backbone" "$layer" "$ckpt" &
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
echo "  Classification Test Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Finished: $(date)"
echo "============================================================"

# Print summary
echo ""
echo "  Results summary:"
for ((i=0; i<TOTAL; i++)); do
    read -r backbone layer ckpt <<< "${JOBS[$i]}"
    tag=$(basename "$ckpt" .npz)
    log="$LOG_DIR/${tag}.log"
    if [ -f "$log" ]; then
        acc=$(grep -oP 'Accuracy = \K[\d.]+' "$log" 2>/dev/null | tail -1 || echo "?")
        echo "    $tag: Acc=${acc}"
    fi
done

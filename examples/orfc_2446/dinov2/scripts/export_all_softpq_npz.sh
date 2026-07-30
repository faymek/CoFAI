#!/bin/bash
# Export ORFC-compatible .npz sidecars (R, codebooks, train PMF) for all Soft-PQ .pt
# weights under dinov2_vitl14_ori and dinov2_vitg14_ori, using multi-GPU parallelism.
#
# Usage:
#   bash examples/orfc_2446/dinov2/scripts/export_all_softpq_npz.sh
#   GPU_IDS=0,1,2,3,4,5 \
#       bash examples/orfc_2446/dinov2/scripts/export_all_softpq_npz.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COFAI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

ORFC2446_DIR="$COFAI_ROOT/examples/orfc_2446/dinov2"
OFFLINE_DIR="$ORFC2446_DIR/offline"

FEAT_ROOT="${FEAT_ROOT:-$COFAI_ROOT/features}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$COFAI_ROOT/weights/orfc_2446}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-5000}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"

LOG_DIR="${LOG_DIR:-$COFAI_ROOT/logs/orfc_2446/dinov2/export_npz}"
mkdir -p "$LOG_DIR"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

declare -a EXPORT_DIRS=(
    "dinov2_vitl14_ori:dinov2_vitl14"
    "dinov2_vitg14_ori:dinov2_vitg14"
)

# Job format: "backbone layer ckpt_path"
JOBS=()

for entry in "${EXPORT_DIRS[@]}"; do
    subdir="${entry%%:*}"
    backbone="${entry##*:}"
    weights_dir="$WEIGHTS_ROOT/$subdir"
    if [ ! -d "$weights_dir" ]; then
        echo "[warn] skip missing dir: $weights_dir"
        continue
    fi
    while IFS= read -r -d '' ckpt; do
        fname=$(basename "$ckpt")
        layer=$(echo "$fname" | grep -oE '^blk[0-9]+' || true)
        if [ -z "$layer" ]; then
            echo "[warn] skip (no layer): $fname"
            continue
        fi
        if [ "$SKIP_EXISTING" = "1" ]; then
            npz="${ckpt%.pt}.npz"
            if [ -f "$npz" ]; then
                continue
            fi
        fi
        JOBS+=("$backbone $layer $ckpt")
    done < <(find -L "$weights_dir" -maxdepth 1 -name '*.pt' -print0 | sort -z)
done

TOTAL=${#JOBS[@]}

echo "============================================================"
echo "  Export Soft-PQ PMF sidecars (.npz)"
echo "  Total jobs: $TOTAL"
echo "  GPUs: ${GPU_IDS} ($NUM_GPUS parallel)"
echo "  FEAT_ROOT: $FEAT_ROOT"
echo "  WEIGHTS_ROOT: $WEIGHTS_ROOT"
echo "  Logs: $LOG_DIR"
echo "  Started: $(date)"
echo "============================================================"

if [ "$TOTAL" -eq 0 ]; then
    echo "Nothing to export (all .npz exist or no .pt found)."
    exit 0
fi

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

COMPLETED=0
FAILED=0

extra_args=()
if [ "$SKIP_EXISTING" = "1" ]; then
    extra_args+=(--skip_existing)
else
    extra_args+=(--force)
fi

run_job() {
    local gpu_id=$1
    local backbone=$2
    local layer=$3
    local ckpt=$4
    local tag
    tag=$(basename "$ckpt" .pt)
    local log="$LOG_DIR/${tag}.log"

    CUDA_VISIBLE_DEVICES=$gpu_id poetry -C "$COFAI_ROOT" run python "$OFFLINE_DIR/export_softpq_npz.py" \
        --ckpt_path "$ckpt" \
        --backbone "$backbone" \
        --layer "$layer" \
        --feat_root "$FEAT_ROOT" \
        --max_train_images "$MAX_TRAIN_IMAGES" \
        "${extra_args[@]}" \
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
    tag=$(basename "$ckpt" .pt)
    gpu_idx=$(wait_for_gpu)
    gpu_id=${GPUS[$gpu_idx]}
    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: $tag"
    run_job "$gpu_id" "$backbone" "$layer" "$ckpt" &
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
echo "  Export Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Logs: $LOG_DIR"
echo "  Finished: $(date)"
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  Failed jobs (see logs):"
    for ((i=0; i<TOTAL; i++)); do
        read -r backbone layer ckpt <<< "${JOBS[$i]}"
        tag=$(basename "$ckpt" .pt)
        log="$LOG_DIR/${tag}.log"
        if [ -f "$log" ] && grep -qiE 'error|traceback|failed|\[error\]' "$log" 2>/dev/null; then
            echo "    - $tag"
        fi
    done
    exit 1
fi

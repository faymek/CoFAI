#!/bin/bash
# Offline cls+seg evaluation for ALL existing DINOv2 ViT-L/14 SoftPQ/PQFC
# checkpoints on blk05/10/15/20 (every codebook config under WEIGHTS_DIR).
#
# Purpose: verify pqfc offline test path matches orfc_2446 behavior when
# checkpoints were trained WITH orthogonal transform (soft PQ).
#
# Default weights: weights/orfc_2446/dinov2_vitl14_ori (existing orfc_2446 ckpts)
# Eval entry:      examples/pqfc/offline/test_{cls,seg}.py
# Parallel:        GPUs 0-3
#
# Usage:
#   bash examples/pqfc/scripts/test_vitl14_all_ckpts.sh
#   GPU_IDS=0,1,2,3 PYTHON=.venv/bin/python \
#       bash examples/pqfc/scripts/test_vitl14_all_ckpts.sh
#   # Or evaluate pqfc-trained weights:
#   WEIGHTS_DIR=weights/pqfc/dinov2_vitl14 bash examples/pqfc/scripts/test_vitl14_all_ckpts.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$PQFC_DIR/offline"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
FEATCODEC_ROOT="$(dirname "$COFAI_ROOT")"

FEAT_ROOT="${FEAT_ROOT:-$FEATCODEC_ROOT/features}"
SEG_FEAT_ROOT="${SEG_FEAT_ROOT:-$FEATCODEC_ROOT/features/voc2012_100}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/orfc_2446/dinov2_vitl14_ori}"
PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TASKS="${TASKS:-cls,seg}"   # comma-separated: cls | seg | cls,seg

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

LOG_DIR="$PQFC_DIR/logs/test_vitl14_all"
mkdir -p "$LOG_DIR/cls" "$LOG_DIR/seg"

BACKBONE="dinov2_vitl14"
LAYERS_RE='^blk(05|10|15|20)_'

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}
IFS=',' read -ra TASK_LIST <<< "$TASKS"

# Job format: "task layer ckpt_path"
JOBS=()

if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "ERROR: weights dir not found: $WEIGHTS_DIR"
    exit 1
fi

for ckpt in "$WEIGHTS_DIR"/*.pt; do
    [ -f "$ckpt" ] || continue
    fname=$(basename "$ckpt" .pt)
    if ! echo "$fname" | grep -qE "$LAYERS_RE"; then
        continue
    fi
    # Skip no-transform / hard-PQ ablations for this consistency check
    if echo "$fname" | grep -qE '_noR_|_noBt_'; then
        continue
    fi
    layer=$(echo "$fname" | grep -oE '^blk[0-9]+')
    for task in "${TASK_LIST[@]}"; do
        task=$(echo "$task" | tr -d ' ')
        [ -n "$task" ] || continue
        JOBS+=("$task $layer $ckpt")
    done
done

TOTAL=${#JOBS[@]}
if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: no blk05/10/15/20 checkpoints found in $WEIGHTS_DIR"
    exit 1
fi

N_CKPT=$(ls "$WEIGHTS_DIR"/*.pt 2>/dev/null | while read -r p; do
    fname=$(basename "$p" .pt)
    echo "$fname" | grep -qE "$LAYERS_RE" || continue
    echo "$fname" | grep -qE '_noR_|_noBt_' && continue
    echo "$fname"
done | wc -l)

echo "============================================================"
echo "  PQFC Offline Test — ViT-L/14 ALL codebook configs"
echo "  (blk05/10/15/20, with-transform ckpts only)"
echo "  Checkpoints: $N_CKPT"
echo "  Total jobs:  $TOTAL  (tasks=$TASKS)"
echo "  GPUs: ${GPU_IDS} ($NUM_GPUS)"
echo "  FEAT_ROOT: $FEAT_ROOT"
echo "  SEG_FEAT_ROOT: $SEG_FEAT_ROOT"
echo "  WEIGHTS_DIR: $WEIGHTS_DIR"
echo "  Logs: $LOG_DIR"
echo "  Started: $(date)"
echo "============================================================"

declare -a GPU_PIDS
for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=0
done

COMPLETED=0
FAILED=0

run_job() {
    local gpu_id=$1
    local task=$2
    local layer=$3
    local ckpt=$4
    local tag
    tag=$(basename "$ckpt" .pt)
    local log="$LOG_DIR/${task}/${tag}.log"

    if [ "$task" = "cls" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_cls.py" \
            --backbone "$BACKBONE" \
            --layer "$layer" \
            --ckpt_path "$ckpt" \
            --feat_root "$FEAT_ROOT" \
            > "$log" 2>&1
    elif [ "$task" = "seg" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_seg.py" \
            --backbone "$BACKBONE" \
            --layer "$layer" \
            --ckpt_path "$ckpt" \
            --feat_root "$FEAT_ROOT" \
            --seg_feat_root "$SEG_FEAT_ROOT" \
            > "$log" 2>&1
    else
        echo "Unknown task: $task" >&2
        return 1
    fi
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
    read -r task layer ckpt <<< "${JOBS[$i]}"
    tag=$(basename "$ckpt" .pt)
    gpu_idx=$(wait_for_gpu)
    gpu_id=${GPUS[$gpu_idx]}

    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU$gpu_id: ${task}/${tag}"
    run_job "$gpu_id" "$task" "$layer" "$ckpt" &
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
echo "  Offline Test Complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Logs: $LOG_DIR"
echo "  Results JSON: $COFAI_ROOT/results/pqfc/$BACKBONE/"
echo "  Finished: $(date)"
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "  Failed jobs:"
    for ((i=0; i<TOTAL; i++)); do
        read -r task layer ckpt <<< "${JOBS[$i]}"
        tag=$(basename "$ckpt" .pt)
        log="$LOG_DIR/${task}/${tag}.log"
        if [ -f "$log" ] && grep -qiE 'error|traceback|failed' "$log" 2>/dev/null; then
            echo "    - ${task}/${tag}  (see $log)"
        fi
    done
fi

echo ""
echo "  Results summary (per layer):"
for layer in blk05 blk10 blk15 blk20; do
    echo "  --- $layer ---"
    for ((i=0; i<TOTAL; i++)); do
        read -r task layer_j ckpt <<< "${JOBS[$i]}"
        [ "$layer_j" = "$layer" ] || continue
        tag=$(basename "$ckpt" .pt)
        log="$LOG_DIR/${task}/${tag}.log"
        [ -f "$log" ] || continue
        if [ "$task" = "cls" ]; then
            acc=$(grep -oE 'Accuracy = [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $3}' || echo "?")
            echo "    [cls] $tag: Acc=${acc}"
        else
            miou=$(grep -oE 'mIoU = [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $3}' || echo "?")
            echo "    [seg] $tag: mIoU=${miou}"
        fi
    done
done

#!/bin/bash
# Train + offline test PQFC K=4 emb=32 for DINOv2 ViT-L/14 blk10/15/20.
#
# WITHOUT orthogonal transform (--no_transform) + hard PQ:
#   no R, tau_start forced to 0, k-means codebook init.
# Hyperparameters aligned with the with-transform K4 script:
#   blk20: lmbda=0.0, lr=5e-4, ep=300
#   blk10/blk15: lmbda=0.5, lr=3e-4, ep=100
#
# After training, runs cls+seg offline tests on the new _noR_ checkpoints.
# Parallel: GPUs 3,4,5 for both train and test phases.
#
# Usage:
#   bash examples/pqfc/scripts/train_test_vitl14_k4_noR.sh
#   PYTHON=.venv/bin/python GPU_IDS=3,4,5 \
#       bash examples/pqfc/scripts/train_test_vitl14_k4_noR.sh
#   SKIP_TRAIN=1  # only test existing K4 noR ckpts under WEIGHTS_DIR
#   SKIP_TEST=1   # only train
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="$PQFC_DIR/offline"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
FEATCODEC_ROOT="$(dirname "$COFAI_ROOT")"

FEAT_ROOT="${FEAT_ROOT:-$FEATCODEC_ROOT/features}"
SEG_FEAT_ROOT="${SEG_FEAT_ROOT:-$FEATCODEC_ROOT/features/voc2012_100}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$COFAI_ROOT/weights/pqfc}"
CKPT_SUBDIR="${CKPT_SUBDIR:-dinov2_vitl14}"
PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-3,4,5}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"

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

LOG_TRAIN="$PQFC_DIR/logs/train_vitl14_k4_noR"
LOG_TEST="$PQFC_DIR/logs/test_vitl14_k4_noR"
mkdir -p "$LOG_TRAIN" "$LOG_TEST/cls" "$LOG_TEST/seg"

BACKBONE="dinov2_vitl14"
K=4
EMB=32
BT=1024

# Job format: "layer lmbda lr epochs"
TRAIN_JOBS=(
    "blk10 0.5 0.0003 100"
    "blk15 0.5 0.0003 100"
    "blk20 0.0 0.0005 300"
)

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

echo "============================================================"
echo "  PQFC Train+Test — ViT-L/14 K=4 emb=32 (no transform / hard)"
echo "  GPUs: ${GPU_IDS} ($NUM_GPUS)"
echo "  FEAT_ROOT: $FEAT_ROOT"
echo "  WEIGHTS_DIR: $WEIGHTS_DIR/$CKPT_SUBDIR"
echo "  SKIP_TRAIN=$SKIP_TRAIN  SKIP_TEST=$SKIP_TEST"
echo "  Started: $(date)"
echo "============================================================"

# -------------------- helpers --------------------
declare -a GPU_PIDS
reset_gpu_pids() {
    for ((g=0; g<NUM_GPUS; g++)); do
        GPU_PIDS[$g]=0
    done
}

COMPLETED=0
FAILED=0

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

wait_remaining() {
    for ((g=0; g<NUM_GPUS; g++)); do
        local_pid=${GPU_PIDS[$g]}
        if [ "$local_pid" -ne 0 ]; then
            wait "$local_pid" 2>/dev/null && COMPLETED=$((COMPLETED+1)) || FAILED=$((FAILED+1))
            GPU_PIDS[$g]=0
        fi
    done
}

# -------------------- train --------------------
if [ "$SKIP_TRAIN" != "1" ]; then
    for ((i=0; i<${#TRAIN_JOBS[@]}; i++)); do
        read -r layer _lmbda _lr _epochs <<< "${TRAIN_JOBS[$i]}"
        feat_dir="$FEAT_ROOT/train/$BACKBONE/$layer"
        if [ ! -d "$feat_dir" ]; then
            echo "ERROR: feature dir not found: $feat_dir"
            exit 1
        fi
        n_feat=$(find "$feat_dir" -maxdepth 1 -name '*.npy' | wc -l)
        echo "  train features $layer: $n_feat .npy"
    done
    echo ""

    TOTAL_TRAIN=${#TRAIN_JOBS[@]}
    COMPLETED=0
    FAILED=0
    reset_gpu_pids

    run_train() {
        local gpu_id=$1
        local layer=$2
        local lmbda=$3
        local lr=$4
        local epochs=$5
        local tag="${BACKBONE}_${layer}_K${K}_emb${EMB}_noR_lmbda${lmbda}_lr${lr}_ep${epochs}"
        local log="$LOG_TRAIN/${tag}.log"

        # --no_transform: hard PQ + k-means (no R, tau_start forced to 0)
        cd "$OFFLINE_DIR" && \
        CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON train_pqfc.py \
            --backbone "$BACKBONE" \
            --layer "$layer" \
            --K "$K" \
            --embedding_dim "$EMB" \
            --bottleneck_dim "$BT" \
            --no_transform \
            --lmbda "$lmbda" \
            --lr "$lr" \
            --epochs "$epochs" \
            --feat_root "$FEAT_ROOT" \
            --weights_dir "$WEIGHTS_DIR" \
            > "$log" 2>&1
    }

    echo "---------- Phase 1: Training ($TOTAL_TRAIN jobs) ----------"
    for ((i=0; i<TOTAL_TRAIN; i++)); do
        read -r layer lmbda lr epochs <<< "${TRAIN_JOBS[$i]}"
        tag="${BACKBONE}_${layer}_K${K}_emb${EMB}_noR_lmbda${lmbda}_lr${lr}_ep${epochs}"
        gpu_idx=$(wait_for_gpu)
        gpu_id=${GPUS[$gpu_idx]}
        echo "[$(date '+%H:%M:%S')] [train $((i+1))/$TOTAL_TRAIN] GPU$gpu_id: $tag"
        run_train "$gpu_id" "$layer" "$lmbda" "$lr" "$epochs" &
        GPU_PIDS[$gpu_idx]=$!
    done
    wait_remaining

    echo "  Train done: ok=$COMPLETED fail=$FAILED"
    if [ "$FAILED" -gt 0 ]; then
        echo "  Failed train logs:"
        for ((i=0; i<TOTAL_TRAIN; i++)); do
            read -r layer lmbda lr epochs <<< "${TRAIN_JOBS[$i]}"
            tag="${BACKBONE}_${layer}_K${K}_emb${EMB}_noR_lmbda${lmbda}_lr${lr}_ep${epochs}"
            log="$LOG_TRAIN/${tag}.log"
            if [ -f "$log" ] && ! grep -q "Checkpoint saved" "$log" 2>/dev/null; then
                echo "    - $tag  ($log)"
            fi
        done
    fi
    echo ""
else
    echo "---------- Phase 1: Training SKIPPED ----------"
fi

# -------------------- test --------------------
if [ "$SKIP_TEST" = "1" ]; then
    echo "---------- Phase 2: Test SKIPPED ----------"
    exit 0
fi

CKPT_DIR="$WEIGHTS_DIR/$CKPT_SUBDIR"
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint dir not found: $CKPT_DIR"
    exit 1
fi

# Discover K4 emb32 no-transform (_noR_) ckpts for blk10/15/20
TEST_JOBS=()
for ckpt in "$CKPT_DIR"/*.pt; do
    [ -f "$ckpt" ] || continue
    fname=$(basename "$ckpt" .pt)
    if ! echo "$fname" | grep -qE '^blk(10|15|20)_'; then
        continue
    fi
    if ! echo "$fname" | grep -q '_K4_emb32_'; then
        continue
    fi
    if ! echo "$fname" | grep -q '_noR_'; then
        continue
    fi
    layer=$(echo "$fname" | grep -oE '^blk[0-9]+')
    TEST_JOBS+=("cls $layer $ckpt")
    TEST_JOBS+=("seg $layer $ckpt")
done

TOTAL_TEST=${#TEST_JOBS[@]}
if [ "$TOTAL_TEST" -eq 0 ]; then
    echo "ERROR: no K4 emb32 noR ckpts in $CKPT_DIR"
    exit 1
fi

COMPLETED=0
FAILED=0
reset_gpu_pids

run_test() {
    local gpu_id=$1
    local task=$2
    local layer=$3
    local ckpt=$4
    local tag
    tag=$(basename "$ckpt" .pt)
    local log="$LOG_TEST/${task}/${tag}.log"

    if [ "$task" = "cls" ]; then
        CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_cls.py" \
            --backbone "$BACKBONE" \
            --layer "$layer" \
            --ckpt_path "$ckpt" \
            --feat_root "$FEAT_ROOT" \
            > "$log" 2>&1
    else
        CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$OFFLINE_DIR/test_seg.py" \
            --backbone "$BACKBONE" \
            --layer "$layer" \
            --ckpt_path "$ckpt" \
            --feat_root "$FEAT_ROOT" \
            --seg_feat_root "$SEG_FEAT_ROOT" \
            > "$log" 2>&1
    fi
}

echo "---------- Phase 2: Offline Test ($TOTAL_TEST jobs) ----------"
for ((i=0; i<TOTAL_TEST; i++)); do
    read -r task layer ckpt <<< "${TEST_JOBS[$i]}"
    tag=$(basename "$ckpt" .pt)
    gpu_idx=$(wait_for_gpu)
    gpu_id=${GPUS[$gpu_idx]}
    echo "[$(date '+%H:%M:%S')] [test $((i+1))/$TOTAL_TEST] GPU$gpu_id: ${task}/${tag}"
    run_test "$gpu_id" "$task" "$layer" "$ckpt" &
    GPU_PIDS[$gpu_idx]=$!
done
wait_remaining

echo ""
echo "============================================================"
echo "  Train+Test Complete (no transform / hard PQ)"
echo "  Test completed: $COMPLETED / $TOTAL_TEST  Failed: $FAILED"
echo "  Train logs: $LOG_TRAIN"
echo "  Test logs:  $LOG_TEST"
echo "  Weights:    $CKPT_DIR"
echo "  Results:    $COFAI_ROOT/results/pqfc/$BACKBONE/"
echo "  Finished: $(date)"
echo "============================================================"

echo ""
echo "  Test summary:"
for ((i=0; i<TOTAL_TEST; i++)); do
    read -r task layer ckpt <<< "${TEST_JOBS[$i]}"
    tag=$(basename "$ckpt" .pt)
    log="$LOG_TEST/${task}/${tag}.log"
    [ -f "$log" ] || continue
    if [ "$task" = "cls" ]; then
        acc=$(grep -oE 'Accuracy = [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $3}' || echo "?")
        echo "    [cls] $tag: Acc=${acc}"
    else
        miou=$(grep -oE 'mIoU = [0-9.]+' "$log" 2>/dev/null | tail -1 | awk '{print $3}' || echo "?")
        echo "    [seg] $tag: mIoU=${miou}"
    fi
done

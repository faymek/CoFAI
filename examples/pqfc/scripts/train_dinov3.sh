#!/bin/bash
# Train DINOv3 CTC PQFC (blk23): K256 e32 / K256 e16 × with/without transform × λ∈{0.5,0.0}
# = 8 configs, epochs=100.
#
# Default train features: ImageNet short-seq (T≈201) under
#   ORFC/features/train/dinov3_vitl16/blk23
# (COCO T≈5445 is too heavy for SoftPQ on 24GB.)
#
# Schedule (one codebook size per round, 4 GPUs in parallel):
#   Round 1: K256 emb=32  → 4 jobs on GPUs 3,4,5,7
#   Round 2: K256 emb=16  → 4 jobs on GPUs 3,4,5,7
#
# Usage:
#   bash examples/pqfc/scripts/train_dinov3.sh
#   PYTHON=.venv/bin/python GPU_IDS=3,4,5,7 \
#       bash examples/pqfc/scripts/train_dinov3.sh
#   FEAT_DIR=/path/to/dinov3_vitl16_coco/blk23  # optional override
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PQFC_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$PQFC_DIR")")"
FEATCODEC_ROOT="$(dirname "$COFAI_ROOT")"
PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-3,4,5,7}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.0003}"
# ImageNet 224 features (T=201); override for COCO long features if needed.
FEAT_DIR="${FEAT_DIR:-$FEATCODEC_ROOT/ORFC/features/train/dinov3_vitl16/blk23}"

export PROJECT_ROOT="$COFAI_ROOT"
unset PYTHONPATH
# Reduce CUDA allocator fragmentation after OPQ / k-means peaks on 24GB cards.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG_DIR="$PQFC_DIR/logs/train_dinov3"
mkdir -p "$LOG_DIR"

if [ ! -d "$FEAT_DIR" ]; then
    echo "ERROR: FEAT_DIR not found: $FEAT_DIR"
    exit 1
fi
n_feat=$(find "$FEAT_DIR" -maxdepth 1 -name '*.npy' | wc -l)
echo "  FEAT_DIR: $FEAT_DIR  ($n_feat .npy)"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
NUM_GPUS=${#GPUS[@]}
if [ "$NUM_GPUS" -lt 4 ]; then
    echo "ERROR: need 4 GPUs for one round (got GPU_IDS=$GPU_IDS)"
    exit 1
fi

# Per round: 4 jobs = (use_transform|no_transform) × (lmbda 0.5|0.0)
# Format: "transform_flag lmbda tag_suffix"
#   transform_flag: use | no
ROUND_VARIANTS=(
    "use 0.5 ws"
    "use 0.0 ws"
    "no  0.5 noR"
    "no  0.0 noR"
)

# Round codebook sizes: emb only (K fixed at 256)
ROUNDS=(
    "32"
    "16"
)

echo "============================================================"
echo "  PQFC DINOv3 blk23 train — 8 configs (2 rounds × 4)"
echo "  K=256  emb={32,16}  transform×λ  epochs=$EPOCHS"
echo "  FEAT_DIR: $FEAT_DIR"
echo "  GPUs: ${GPU_IDS}"
echo "  Started: $(date)"
echo "============================================================"

run_one() {
    local gpu_id=$1
    local emb=$2
    local transform_flag=$3
    local lmbda=$4
    local mode_tag=$5

    local tf_args=()
    if [ "$transform_flag" = "use" ]; then
        tf_args=(--use_transform --warm_start_opq)
    else
        tf_args=(--no_transform)
    fi

    local tag="blk23_K256_emb${emb}_${mode_tag}_lmbda${lmbda}_ep${EPOCHS}"
    local log="$LOG_DIR/${tag}.log"

    echo "[$(date '+%H:%M:%S')] GPU$gpu_id → $tag"
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON "$PQFC_DIR/offline/train_pqfc_dinov3.py" \
        --K 256 \
        --embedding_dim "$emb" \
        --batch_size 32 \
        --max_train 5000 \
        --lmbda "$lmbda" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --feat_dir "$FEAT_DIR" \
        --gpu 0 \
        "${tf_args[@]}" \
        > "$log" 2>&1
}

round_idx=0
for emb in "${ROUNDS[@]}"; do
    round_idx=$((round_idx + 1))
    echo ""
    echo "---------- Round $round_idx/2: K256 emb=$emb (4 jobs) ----------"

    pids=()
    tags=()
    for ((i=0; i<4; i++)); do
        read -r transform_flag lmbda mode_tag <<< "${ROUND_VARIANTS[$i]}"
        gpu_id=${GPUS[$i]}
        tag="blk23_K256_emb${emb}_${mode_tag}_lmbda${lmbda}_ep${EPOCHS}"
        tags+=("$tag")
        run_one "$gpu_id" "$emb" "$transform_flag" "$lmbda" "$mode_tag" &
        pids+=($!)
    done

    fail=0
    for ((i=0; i<4; i++)); do
        if wait "${pids[$i]}"; then
            echo "  OK   ${tags[$i]}"
        else
            echo "  FAIL ${tags[$i]}  (log: $LOG_DIR/${tags[$i]}.log)"
            fail=$((fail + 1))
        fi
    done

    if [ "$fail" -gt 0 ]; then
        echo "ERROR: round $round_idx had $fail failure(s); aborting"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "  All 8 DINOv3 train jobs finished"
echo "  Logs: $LOG_DIR"
echo "  Weights: $COFAI_ROOT/weights/pqfc/dinov3_vitl16"
echo "  Finished: $(date)"
echo "============================================================"

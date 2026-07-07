#!/bin/bash
# 重跑 result_subdir 碰撞的 16 个 job，写入独立目录 SoftPQ_by_ckpt/<stem>_<task>
#
# Usage:
#   GPU_ID=4 bash examples/orfc_2446/scripts/rerun_collision_eval.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORFC2446_DIR="$(dirname "$SCRIPT_DIR")"
COFAI_ROOT="$(dirname "$(dirname "$ORFC2446_DIR")")"

PYTHON="${PYTHON:-$COFAI_ROOT/.venv/bin/python}"
GPU_ID="${GPU_ID:-4}"
OUTPUT_BASE="${OUTPUT_BASE:-$COFAI_ROOT/eval_results}"

export PROJECT_ROOT="$COFAI_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ -n "${PYTHONPATH:-}" ]; then
    CLEANED=""
    IFS=':' read -ra _PP_PARTS <<< "$PYTHONPATH"
    for _p in "${_PP_PARTS[@]}"; do
        case "$_p" in
            *ORFC/coding/CompressAI*|*coding/CompressAI*) continue ;;
            *) CLEANED="${CLEANED:+${CLEANED}:}${_p}" ;;
        esac
    done
    if [ -n "$CLEANED" ]; then export PYTHONPATH="$CLEANED"; else unset PYTHONPATH; fi
fi

W_L="weights/orfc_2446/dinov2_vitl14_ori"

# backbone layer task checkpoint
JOBS=(
    "dinov2_vitl14 blk05 cls ${W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk05 cls ${W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk10 cls ${W_L}/blk10_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk10 cls ${W_L}/blk10_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk15 cls ${W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk15 cls ${W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk20 cls ${W_L}/blk20_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk20 cls ${W_L}/blk20_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk05 seg ${W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk05 seg ${W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk10 seg ${W_L}/blk10_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk10 seg ${W_L}/blk10_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk15 seg ${W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk15 seg ${W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"
    "dinov2_vitl14 blk20 seg ${W_L}/blk20_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt"
)

TOTAL=${#JOBS[@]}
LOG_DIR="$OUTPUT_BASE/collision_rerun_logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Collision rerun: $TOTAL jobs on GPU $GPU_ID"
echo "  Output: $OUTPUT_BASE/SoftPQ_by_ckpt/<stem>_<task>"
echo "  Started: $(date)"
echo "============================================================"

FAILED=0
COMPLETED=0

for ((i=0; i<TOTAL; i++)); do
    read -r backbone layer task ckpt <<< "${JOBS[$i]}"
    stem=$(basename "$ckpt" .pt)
    result_subdir="${OUTPUT_BASE}/SoftPQ_by_ckpt/${stem}_${task}"
    log="${LOG_DIR}/${stem}_${task}.log"

    if [ ! -f "$COFAI_ROOT/$ckpt" ]; then
        echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] MISSING ckpt: $ckpt"
        FAILED=$((FAILED+1))
        continue
    fi
    npz="${COFAI_ROOT}/${ckpt%.pt}.npz"
    if [ ! -f "$npz" ]; then
        echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] MISSING npz: ${ckpt%.pt}.npz"
        FAILED=$((FAILED+1))
        continue
    fi

    echo "[$(date '+%H:%M:%S')] [$((i+1))/$TOTAL] GPU${GPU_ID}: ${layer}/${task} ${stem:0:48}..."
    if CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON" "$ORFC2446_DIR/run_eval_orfc_2446.py" \
        --backbone "$backbone" \
        --layer "$layer" \
        --task "$task" \
        --ckpt_path "$ckpt" \
        --cuda \
        --output_dir "$OUTPUT_BASE" \
        --result_subdir "$result_subdir" \
        > "$log" 2>&1; then
        COMPLETED=$((COMPLETED+1))
    else
        echo "  FAILED — see $log"
        FAILED=$((FAILED+1))
    fi
done

echo ""
echo "============================================================"
echo "  Collision rerun complete"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed: $FAILED"
echo "  Logs: $LOG_DIR"
echo "  Finished: $(date)"
echo "============================================================"

exit $(( FAILED > 0 ? 1 : 0 ))

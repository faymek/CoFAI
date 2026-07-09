#!/bin/bash
# Patch-only OPQ + SoftPQ training (6 configs, train_tokens=patch).
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$ROOT"

PYTHON="poetry run python"
TRAIN_SCRIPT="examples/orfc_2446_dinov3/offline/train_soft_pq_dinov3.py"
LOG_DIR="examples/orfc_2446_dinov3/logs/train_ptpatch"
mkdir -p "$LOG_DIR"

EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
TAU_S="${TAU_S:-0.5}"
TAU_E="${TAU_E:-0.005}"
BS="${BS:-8}"
TRAIN="${TRAIN:-1000}"
NVAL="${NVAL:-50}"
NORM="${NORM:-split_cls_patch}"
LMBDA="${LMBDA:-0.0}"
TRAIN_TOKENS="${TRAIN_TOKENS:-patch}"

echo ""
echo "========================================"
echo "  Patch-only training (OPQ + SoftPQ)"
echo "  train_tokens=${TRAIN_TOKENS}  norm=${NORM}"
echo "  ep=${EPOCHS}  train=${TRAIN}  batch=${BS}"
echo "========================================"

run_wave() {
    local wave_name=$1
    shift
    local cfgs=("$@")
    echo ""
    echo "========== $wave_name (${#cfgs[@]} jobs) =========="
    local pids=()
    local tags=()
    for cfg in "${cfgs[@]}"; do
        read -r tag K emb gpu <<< "$cfg"
        tags+=("$tag")
        local log="$LOG_DIR/${tag}.log"
        echo "[launch] $tag  GPU=$gpu  K=$K  emb=$emb"
        CUDA_VISIBLE_DEVICES=$gpu \
        $PYTHON "$TRAIN_SCRIPT" \
            --K "$K" --embedding_dim "$emb" \
            --norm_mode "$NORM" \
            --train_tokens "$TRAIN_TOKENS" \
            --max_train "$TRAIN" --epochs "$EPOCHS" --batch_size "$BS" --n_val "$NVAL" \
            --lr "$LR" --tau_start "$TAU_S" --tau_end "$TAU_E" \
            --lmbda "$LMBDA" \
            --gpu 0 \
            > "$log" 2>&1 &
        pids+=($!)
    done
    local fail=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "[done]  ${tags[$i]}  OK"
        else
            echo "[FAIL]  ${tags[$i]}"
            fail=$((fail + 1))
        fi
    done
    return $fail
}

WAVE_E32=(
    "blk23_K4_e32_ptpatch     4    32  ${GPU0:-3}"
    "blk23_K16_e32_ptpatch    16   32  ${GPU1:-4}"
    "blk23_K256_e32_ptpatch   256  32  ${GPU2:-5}"
    "blk23_K512_e32_ptpatch   512  32  ${GPU3:-6}"
)

WAVE_E16=(
    "blk23_K64_e16_ptpatch    64   16  ${GPU0:-3}"
    "blk23_K256_e16_ptpatch   256  16  ${GPU1:-4}"
)

TOTAL_FAIL=0
run_wave "Wave1 e32 ptpatch" "${WAVE_E32[@]}" || TOTAL_FAIL=$((TOTAL_FAIL + $?))
run_wave "Wave2 e16 ptpatch" "${WAVE_E16[@]}" || TOTAL_FAIL=$((TOTAL_FAIL + $?))

echo ""
echo "Logs: $LOG_DIR"
[ $TOTAL_FAIL -eq 0 ] && echo "All passed." || echo "$TOTAL_FAIL job(s) FAILED."
exit $TOTAL_FAIL

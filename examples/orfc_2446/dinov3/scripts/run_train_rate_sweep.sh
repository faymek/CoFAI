#!/bin/bash
# High-capacity rate sweep: K256/K512/K1024 e32 (standard all-token training).
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$ROOT"

PYTHON="poetry run python"
TRAIN_SCRIPT="examples/orfc_2446/dinov3/offline/train_soft_pq_dinov3.py"
WD="weights/orfc_2446_dinov3"
LOG_DIR="examples/orfc_2446/dinov3/logs/train_rate_sweep"
mkdir -p "$LOG_DIR"

EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
TAU_S="${TAU_S:-0.5}"
TAU_E="${TAU_E:-0.005}"
BS="${BS:-8}"
TRAIN="${TRAIN:-1000}"
NVAL="${NVAL:-50}"
NORM="${NORM:-split_cls_patch}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

echo ""
echo "========================================"
echo "  Rate sweep (all tokens)  K256/K512/K1024 e32"
echo "  SKIP_EXISTING=${SKIP_EXISTING}"
echo "========================================"

CFGS=(
  "blk23_K256_e32_rs  256  32  ${GPU0:-3}"
  "blk23_K512_e32_rs  512  32  ${GPU1:-4}"
  "blk23_K1024_e32_rs 1024 32  ${GPU2:-5}"
)

pids=() tags=()
for cfg in "${CFGS[@]}"; do
    read -r tag K emb gpu <<< "$cfg"
    ckpt_name="blk23_K${K}_emb${emb}_bt1024_ws_${NORM}_tau${TAU_S}_lr${LR}_ep${EPOCHS}_n${TRAIN}_s42.pt"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$WD/$ckpt_name" ]; then
        echo "[skip] $tag  exists: $ckpt_name"
        continue
    fi
    tags+=("$tag")
    log="$LOG_DIR/${tag}.log"
    echo "[launch] $tag  GPU=$gpu  K=$K"
    CUDA_VISIBLE_DEVICES=$gpu \
    $PYTHON "$TRAIN_SCRIPT" \
        --K "$K" --embedding_dim "$emb" \
        --norm_mode "$NORM" --train_tokens all \
        --max_train "$TRAIN" --epochs "$EPOCHS" --batch_size "$BS" --n_val "$NVAL" \
        --lr "$LR" --tau_start "$TAU_S" --tau_end "$TAU_E" \
        --gpu 0 \
        > "$log" 2>&1 &
    pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[done] ${tags[$i]} OK"
    else
        echo "[FAIL] ${tags[$i]}"
        fail=$((fail + 1))
    fi
done

echo ""
echo "Logs: $LOG_DIR"
[ $fail -eq 0 ] && echo "All passed." || echo "$fail job(s) FAILED."
exit $fail

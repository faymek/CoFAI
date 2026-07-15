#!/bin/bash
# DINOv3 blk23 SoftPQ training — parallel K sweep (COCO features)
#
# Usage:
#   unset PYTHONPATH
#   export PROJECT_ROOT=/data4/workspace/zlt/featcodec/CoFAI
#   bash examples/orfc_2446/dinov3/scripts/run_train_pipeline.sh

set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH

PYTHON="poetry run python"
TRAIN_SCRIPT="examples/orfc_2446/dinov3/offline/train_soft_pq_dinov3.py"
LOG_DIR="examples/orfc_2446/dinov3/logs/train"
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
USE_TRANSFORM="${USE_TRANSFORM:-1}"
TRANSFORM_FLAG=""
if [ "$USE_TRANSFORM" = "0" ]; then
    TRANSFORM_FLAG="--no_transform"
fi

echo ""
echo "========================================"
echo "  DINOv3 ORFC-2446  COCO blk23 features"
echo "  tau=${TAU_S}->${TAU_E}  lr=${LR}  ep=${EPOCHS}"
echo "  train=${TRAIN}  batch=${BS}  n_val=${NVAL}"
echo "  norm=${NORM}  lambda=${LMBDA}  use_transform=${USE_TRANSFORM}"
echo "  6 configs, GPU 0-3, 2 waves (e32×4 then e16×2)"
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
            --max_train "$TRAIN" --epochs "$EPOCHS" --batch_size "$BS" --n_val "$NVAL" \
            --lr "$LR" --tau_start "$TAU_S" --tau_end "$TAU_E" \
            --lmbda "$LMBDA" \
            $TRANSFORM_FLAG \
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
    "blk23_K4_e32     4    32  3"
    "blk23_K16_e32    16   32  4"
    "blk23_K256_e32   256  32  5"
    "blk23_K512_e32   512  32  6"
)

WAVE_E16=(
    "blk23_K64_e16    64   16  3"
    "blk23_K256_e16   256  16  4"
)

TOTAL_FAIL=0
run_wave "Wave1 e32 (GPU 0-3)" "${WAVE_E32[@]}" || TOTAL_FAIL=$((TOTAL_FAIL + $?))
run_wave "Wave2 e16 (GPU 0-1)" "${WAVE_E16[@]}" || TOTAL_FAIL=$((TOTAL_FAIL + $?))

echo ""
echo "=========================================="
echo "  RESULTS SUMMARY"
echo "=========================================="

$PYTHON - "$LOG_DIR" << 'PYSCRIPT'
import sys, os, re, json
from pathlib import Path

log_dir = Path(sys.argv[1])
rows = []
for path in sorted(log_dir.glob("*.log")):
    tag = path.stem
    d = {"tag": tag}
    with open(path) as f:
        text = f.read()
    m = re.search(r"Val raw MSE = ([\d.eE+-]+)", text)
    if m:
        d["val_raw_mse"] = m.group(1)
    m = re.search(r"Val post-LN patch MSE \(delta_L_ref\) = ([\d.eE+-]+)", text)
    if m:
        d["val_postln_patch_mse"] = m.group(1)
    m = re.search(r"Val MSE = ([\d.eE+-]+)", text)
    if m:
        d["val_mse"] = m.group(1)
    m = re.search(r"Checkpoint → (.+\.pt)", text)
    if m:
        d["ckpt"] = m.group(1).strip()
    if d.get("val_mse") or d.get("val_postln_patch_mse"):
        rows.append(d)

if not rows:
    print("  (No completed results)")
else:
    print(f"{'Config':>16} {'postLN':>10} {'raw_MSE':>12}  checkpoint")
    print("-" * 72)
    for d in rows:
        postln = d.get('val_postln_patch_mse', d.get('val_mse', '-'))
        raw = d.get('val_raw_mse', d.get('val_mse', '-'))
        print(f"{d['tag']:>16} {postln:>10} {raw:>12}  {d.get('ckpt','')}")
PYSCRIPT

echo ""
echo "Logs: $LOG_DIR"
[ $TOTAL_FAIL -eq 0 ] && echo "All passed." || echo "$TOTAL_FAIL job(s) FAILED."

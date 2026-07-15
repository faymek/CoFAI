#!/bin/bash
# Step 1+2: fair baseline vs ptpatch MSE + ptpatch token rel-MSE (6 configs, GPU 0-5).
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$ROOT"

PYTHON="poetry run python"
LOG_DIR="examples/orfc_2446/dinov3/logs/analyze"
mkdir -p "$LOG_DIR"

TAGS=(K4_e32 K16_e32 K256_e32 K512_e32 K64_e16 K256_e16)
GPUS=(0 1 2 3 4 5)
WD="weights/orfc_2446_dinov3"

echo "========== Step 1: fair baseline vs ptpatch MSE (per config) =========="
pids=()
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  gpu="${GPUS[$i]}"
  log="$LOG_DIR/compare_fair_mse_${tag}.log"
  echo "[launch] compare $tag GPU=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
  $PYTHON examples/orfc_2446/dinov3/scripts/compare_ptpatch_mse.py \
    --tag "$tag" --task semseg --gpu 0 \
  > "$log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] compare ${TAGS[$i]}"
  else
    echo "[FAIL] compare ${TAGS[$i]}  (see $LOG_DIR/compare_fair_mse_${TAGS[$i]}.log)"
    fail=$((fail + 1))
  fi
done

echo ""
echo "========== Step 2: ptpatch token rel-MSE (post-LN patch) =========="
pids=()
CKPTS=(
  "blk23_K4_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K16_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K256_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K512_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K64_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
  "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
)
for i in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$i]}"
  tag="${TAGS[$i]}"
  gpu="${GPUS[$i]}"
  log="$LOG_DIR/token_relmse_ptpatch_${tag}.log"
  echo "[launch] relmse $tag GPU=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu \
  $PYTHON examples/orfc_2446/dinov3/scripts/analyze_token_rmse.py \
    --ckpt_path "$WD/$ckpt" --task semseg --gpu 0 --prefix_bypass \
  > "$log" 2>&1 &
  pids+=($!)
done
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] relmse ${TAGS[$i]}"
  else
    echo "[FAIL] relmse ${TAGS[$i]}  (see $LOG_DIR/token_relmse_ptpatch_${TAGS[$i]}.log)"
    fail=$((fail + 1))
  fi
done

echo ""
echo "========== SUMMARY =========="
$PYTHON - << 'PY'
import json
from pathlib import Path

results = Path("examples/orfc_2446/dinov3/results")
baseline_path = results / (
    "token_relmse_semseg_blk23_K256_emb16_bt1024_ws_split_cls_patch_"
    "tau0.5_lr0.0003_ep100_n1000_s42.json"
)
baseline_patch_post = None
if baseline_path.is_file():
    with open(baseline_path) as f:
        baseline_patch_post = json.load(f)["post_ln"]["patch"]["pooled_rel_mse"]

print(f"{'tag':>10}  {'Δall_mse':>12}  {'Δpatch_mse':>12}  "
      f"{'Δpost_ln_patch':>14}  ptpatch_post_ln")
print("-" * 72)
for path in sorted(results.glob("ptpatch_fair_mse_semseg_*.json")):
    if path.name.endswith("_all.json"):
        continue
    with open(path) as f:
        d = json.load(f)
    row = d["pairs"][0]
    tag = row["tag"]
    delta = row["delta"]
    ptp = row["ptpatch"]["patch_pooled_rel_post_ln"]
    print(
        f"{tag:>10}  {delta['all_mse']:+12.1f}  {delta['patch_mse']:+12.2f}  "
        f"{delta['patch_pooled_rel_post_ln']:+14.4f}  {ptp:.4f}"
    )

if baseline_patch_post is not None:
    ptpatch_k256 = results / "ptpatch_fair_mse_semseg_K256_e16.json"
    if ptpatch_k256.is_file():
        with open(ptpatch_k256) as f:
            ptp = json.load(f)["pairs"][0]["ptpatch"]["patch_pooled_rel_post_ln"]
        print()
        print(f"K256 e16 baseline post-LN patch rel: {baseline_patch_post:.4f}")
        print(f"K256 e16 ptpatch   post-LN patch rel: {ptp:.4f}")
        print(f"Δ (ptpatch - baseline): {ptp - baseline_patch_post:+.4f}")
PY

[ $fail -eq 0 ] && echo "All passed." || { echo "$fail job(s) FAILED."; exit 1; }

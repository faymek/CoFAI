#!/bin/bash
# Semseg replay for rate sweep + baseline reference points.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT="${PROJECT_ROOT:-/data4/workspace/zlt/featcodec/CoFAI}"
cd "$ROOT"
unset PYTHONPATH
export PROJECT_ROOT="$ROOT"

PYTHON="poetry run python"
REPLAY="$PYTHON examples/orfc_2446_dinov3/run_orfc_dinov3.py replay"
NORM="${NORM:-split_cls_patch}"
WD="weights/orfc_2446_dinov3"
LOG_DIR="examples/orfc_2446_dinov3/logs/replay_rate_sweep"
RESULTS="examples/orfc_2446_dinov3/results/rate_sweep_summary.json"
mkdir -p "$LOG_DIR"

GPU="${GPU:-3}"

echo ""
echo "========== bypass baseline =========="
CUDA_VISIBLE_DEVICES=$GPU \
$REPLAY --task semseg --mode bypass --norm_mode "$NORM" --gpu 0 \
  > "$LOG_DIR/semseg_bypass.log" 2>&1
echo "[done] bypass"

CKPTS=(
  "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
  "blk23_K256_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
  "blk23_K512_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
  "blk23_K1024_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
)

for ckpt in "${CKPTS[@]}"; do
  if [ ! -f "$WD/$ckpt" ]; then
    echo "[skip] missing $ckpt"
    continue
  fi
  echo "[launch] $ckpt"
  CUDA_VISIBLE_DEVICES=$GPU \
  $REPLAY --task semseg --mode orfc \
    --ckpt_path "$WD/$ckpt" \
    --norm_mode "$NORM" --gpu 0 \
    > "$LOG_DIR/semseg_${ckpt%.pt}.log" 2>&1 \
    && echo "[done] $ckpt" \
    || echo "[FAIL] $ckpt"
done

echo ""
echo "========== SUMMARY =========="
$PYTHON - "$RESULTS" << 'PY'
import json, sys
from pathlib import Path

out_path = Path(sys.argv[1])
results_dir = Path("examples/orfc_2446_dinov3/results")
rows = []

bypass = results_dir / "semseg_bypass.json"
if bypass.is_file():
    d = json.loads(bypass.read_text())
    rows.append({
        "config": "bypass",
        "mIoU": d["metrics"]["mIoU"],
        "bpfp": 0.0,
        "ckpt": None,
    })

targets = [
    ("K256_e16", "K256_emb16"),
    ("K256_e32", "K256_emb32"),
    ("K512_e32", "K512_emb32"),
    ("K1024_e32", "K1024_emb32"),
]
for label, needle in targets:
    matches = [p for p in results_dir.glob("semseg_orfc_blk23_*ep100*.json") if needle in p.name and "ptpatch" not in p.name]
    if not matches:
        continue
    p = sorted(matches)[-1]
    d = json.loads(p.read_text())
    rows.append({
        "config": label,
        "mIoU": d["metrics"]["mIoU"],
        "bpfp": d.get("rate", {}).get("bpfp"),
        "ckpt": d.get("ckpt_path"),
        "result_json": str(p),
    })

bypass_miou = rows[0]["mIoU"] if rows and rows[0]["config"] == "bypass" else None
print(f"{'Config':12s}  {'BPFP':>8s}  {'mIoU':>8s}  {'d_vs_bypass':>12s}")
print("-" * 50)
for r in rows:
    d = ""
    if bypass_miou is not None and r["config"] != "bypass":
        d = f"{r['mIoU'] - bypass_miou:+.4f}"
    bpfp = r.get("bpfp")
    bpfp_s = f"{bpfp:.4f}" if isinstance(bpfp, (int, float)) else "?"
    print(f"{r['config']:12s}  {bpfp_s:>8s}  {r['mIoU']:.4f}  {d:>12s}")

out_path.write_text(json.dumps({"rows": rows, "bypass_mIoU": bypass_miou}, indent=2))
print(f"\nWrote {out_path}")
PY

echo "Logs: $LOG_DIR"

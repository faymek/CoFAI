#!/usr/bin/env python3
"""Per-group relative MSE: decode vs original, pre/post LN (cls / reg / patch)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_COFAI_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.config_utils import load_config, resolve_project_root, task_feat_dir
from lib.orfc_codec import resolve_n_prefix
from lib.token_mse_metrics import GROUPS, analyze_codec_mse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt_path",
        default=(
            "weights/orfc_2446_dinov3/"
            "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
        ),
    )
    ap.add_argument("--norm_mode", default="split_cls_patch")
    ap.add_argument("--n_prefix", type=int, default=0)
    ap.add_argument("--task", default="semseg", choices=["semseg", "depth"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--prefix_bypass", action="store_true", default=None,
                    help="Skip PQ on prefix (default: on for ptpatch ckpts)")
    ap.add_argument("--no_prefix_bypass", dest="prefix_bypass", action="store_false")
    args = ap.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    cfg = load_config()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    n_prefix = resolve_n_prefix(args.norm_mode, args.n_prefix, cfg.get("n_prefix", 5))
    feat_dir = task_feat_dir(cfg, args.task)
    token_dir = feat_dir / "tokens"
    meta_dir = feat_dir / "meta"

    if args.subset:
        with open(args.subset) as f:
            stems = sorted({ln.strip() for ln in f if ln.strip()})
    else:
        stems = sorted(p.stem for p in token_dir.glob("*.npy"))
    if args.max_samples and args.max_samples > 0:
        stems = stems[: args.max_samples]

    print(f"[analyze] ckpt={args.ckpt_path}")
    print(f"[analyze] task={args.task}  n={len(stems)}  n_prefix={n_prefix}  gpu={args.gpu}")

    t0 = time.time()
    metrics = analyze_codec_mse(
        args.ckpt_path,
        token_dir=token_dir,
        meta_dir=meta_dir,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
        device=device,
        stems=stems,
        prefix_bypass=args.prefix_bypass,
    )
    out = {
        **metrics,
        "task": args.task,
        "definition": "rel_mse = mean((hat-ref)^2) / mean(ref^2); "
        "mean_rel = average of per-image rel_mse; "
        "pooled_rel = mean(mse_i) / mean(energy_i)",
    }

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"token_relmse_{args.task}_{Path(args.ckpt_path).stem}"
    out_path = results_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    def _row(stage: str, g: str) -> str:
        d = out[stage][g]
        return (
            f"  {g:6s}  mean_rel={d['mean_rel_mse']:.6f}  "
            f"pooled_rel={d['pooled_rel_mse']:.6f}  "
            f"abs_mse={d['mean_abs_mse']:.4f}  energy={d['mean_energy']:.4f}"
        )

    print(f"  prefix_bypass={out['prefix_bypass']}")
    print("\n===== Pre-LN (codec I/O features) =====")
    for g in GROUPS:
        print(_row("pre_ln", g))
    print("\n===== Post-LN (model.norm, before task head) =====")
    for g in GROUPS:
        print(_row("post_ln", g))
    print(f"\n[analyze] saved -> {out_path}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

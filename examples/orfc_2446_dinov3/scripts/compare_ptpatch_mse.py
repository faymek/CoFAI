#!/usr/bin/env python3
"""Fair baseline vs ptpatch MSE comparison (patch-only + post-LN metrics)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_COFAI_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.config_utils import load_config, resolve_project_root, task_feat_dir
from lib.orfc_codec import resolve_n_prefix
from lib.token_mse_metrics import analyze_codec_mse

PTPATCH_PAIRS = [
    ("K4_e32", "blk23_K4_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"),
    ("K16_e32", "blk23_K16_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"),
    ("K256_e32", "blk23_K256_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"),
    ("K512_e32", "blk23_K512_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"),
    ("K64_e16", "blk23_K64_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"),
    ("K256_e16", "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"),
]


def _fair_summary(metrics: dict) -> dict:
    return {
        "prefix_bypass": metrics["prefix_bypass"],
        "all_mse": metrics["pre_ln"]["all"]["mean_abs_mse"],
        "prefix_mse": metrics["pre_ln"]["prefix"]["mean_abs_mse"],
        "patch_mse": metrics["pre_ln"]["patch"]["mean_abs_mse"],
        "patch_pooled_rel_pre_ln": metrics["pre_ln"]["patch"]["pooled_rel_mse"],
        "patch_pooled_rel_post_ln": metrics["post_ln"]["patch"]["pooled_rel_mse"],
        "patch_mean_rel_post_ln": metrics["post_ln"]["patch"]["mean_rel_mse"],
        "post_ln_all_pooled_rel": metrics["post_ln"]["all"]["pooled_rel_mse"],
    }


def _delta(baseline: dict, ptpatch: dict) -> dict:
    out = {}
    for k in baseline:
        if k == "prefix_bypass":
            continue
        out[k] = ptpatch[k] - baseline[k]
    return out


def compare_pair(
    tag: str,
    baseline_ckpt: Path,
    *,
    weights_dir: Path,
    token_dir: Path,
    meta_dir: Path,
    norm_mode: str,
    n_prefix: int,
    device: torch.device,
    stems: list[str],
) -> dict:
    ptpatch_ckpt = weights_dir / baseline_ckpt.name.replace(".pt", "_ptpatch.pt")
    if not ptpatch_ckpt.is_file():
        raise FileNotFoundError(f"Missing ptpatch ckpt: {ptpatch_ckpt}")

    baseline = analyze_codec_mse(
        baseline_ckpt,
        token_dir=token_dir,
        meta_dir=meta_dir,
        norm_mode=norm_mode,
        n_prefix=n_prefix,
        device=device,
        stems=stems,
        prefix_bypass=False,
        verbose=True,
    )
    ptpatch = analyze_codec_mse(
        ptpatch_ckpt,
        token_dir=token_dir,
        meta_dir=meta_dir,
        norm_mode=norm_mode,
        n_prefix=n_prefix,
        device=device,
        stems=stems,
        prefix_bypass=True,
        verbose=True,
    )
    b = _fair_summary(baseline)
    p = _fair_summary(ptpatch)
    return {
        "tag": tag,
        "baseline_ckpt": str(baseline_ckpt),
        "ptpatch_ckpt": str(ptpatch_ckpt),
        "baseline": b,
        "ptpatch": p,
        "delta_ptpatch_minus_baseline": _delta(b, p),
        "baseline_full": baseline,
        "ptpatch_full": ptpatch,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default=None,
                    help="Single pair tag, e.g. K256_e16 (default: all pairs)")
    ap.add_argument("--weights_dir", default="weights/orfc_2446_dinov3")
    ap.add_argument("--norm_mode", default="split_cls_patch")
    ap.add_argument("--n_prefix", type=int, default=0)
    ap.add_argument("--task", default="semseg", choices=["semseg", "depth"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
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
    weights_dir = Path(args.weights_dir)

    if args.subset:
        with open(args.subset) as f:
            stems = sorted({ln.strip() for ln in f if ln.strip()})
    else:
        stems = sorted(p.stem for p in token_dir.glob("*.npy"))
    if args.max_samples > 0:
        stems = stems[: args.max_samples]

    pairs = PTPATCH_PAIRS
    if args.tag:
        pairs = [p for p in PTPATCH_PAIRS if p[0] == args.tag]
        if not pairs:
            raise SystemExit(f"Unknown tag: {args.tag}")

    results = []
    for tag, ckpt_name in pairs:
        print(f"\n{'=' * 70}\n  [{tag}] baseline vs ptpatch  n={len(stems)}\n{'=' * 70}")
        row = compare_pair(
            tag,
            weights_dir / ckpt_name,
            weights_dir=weights_dir,
            token_dir=token_dir,
            meta_dir=meta_dir,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
            device=device,
            stems=stems,
        )
        results.append(row)
        b, p, d = row["baseline"], row["ptpatch"], row["delta_ptpatch_minus_baseline"]
        print(f"\n  metric                    baseline        ptpatch         Δ")
        print(f"  {'-' * 62}")
        for key in (
            "all_mse", "prefix_mse", "patch_mse",
            "patch_pooled_rel_pre_ln", "patch_pooled_rel_post_ln",
        ):
            print(f"  {key:26s}  {b[key]:14.4f}  {p[key]:14.4f}  {d[key]:+14.4f}")

    out = {
        "task": args.task,
        "norm_mode": args.norm_mode,
        "n_prefix": n_prefix,
        "n_samples": len(stems),
        "pairs": [
            {
                "tag": r["tag"],
                "baseline_ckpt": r["baseline_ckpt"],
                "ptpatch_ckpt": r["ptpatch_ckpt"],
                "baseline": r["baseline"],
                "ptpatch": r["ptpatch"],
                "delta": r["delta_ptpatch_minus_baseline"],
            }
            for r in results
        ],
    }
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    elif args.tag:
        out_path = results_dir / f"ptpatch_fair_mse_{args.task}_{args.tag}.json"
    else:
        out_path = results_dir / f"ptpatch_fair_mse_{args.task}_all.json"

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[compare] saved -> {out_path}")


if __name__ == "__main__":
    main()

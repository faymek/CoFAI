#!/usr/bin/env python3
"""Offline PQFC replay on DINOv3 slot24 features (CTC: semseg + depth)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_COFAI_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLE_DIR = Path(__file__).resolve().parent
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.entropy_models.soft_pq import load_codec, load_codec_meta

from lib.config_utils import load_config, resolve_project_root, task_feat_dir
from lib.orfc_codec import (
    NORM_MODE_CHOICES,
    encode_decode_single,
    norm_sideinfo_bits,
    resolve_n_prefix,
)
from lib.rate_eval import evaluate_rate

from lib.dataset_utils import (  # noqa: E402
    build_backbone,
    build_dataset,
    build_head,
    build_meter,
    load_subset,
)


def _fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def _build_stem_index(dataset, task: str, cfg: dict) -> dict[str, int]:
    from lib.dataset_utils import sample_stem

    nyu_root = cfg["datasets"]["depth"]["root"] if task == "depth" else None
    index: dict[str, int] = {}
    for idx in tqdm(range(len(dataset)), desc=f"index-{task}", leave=False):
        sample = dataset[idx]
        stem = sample_stem(sample, task, nyu_root=nyu_root)
        if stem in index:
            _fail(f"Duplicate stem={stem} at idx={idx} and idx={index[stem]}")
        index[stem] = idx
    return index


def _prime_backbone_rope(backbone, meta: dict, patch_size: int, device: torch.device) -> None:
    h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
    img = torch.zeros(1, 3, h_p * patch_size, w_p * patch_size, device=device)
    with torch.no_grad():
        backbone.encode(img)


@torch.inference_mode()
def _replay_one(
    tokens: np.ndarray,
    meta: dict,
    *,
    task: str,
    backbone,
    head,
    device: torch.device,
    patch_size: int,
    sample: dict | None = None,
):
    h_hat = torch.from_numpy(tokens).unsqueeze(0).to(device=device, dtype=torch.float32)
    h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
    token_res = (h_p, w_p)
    _prime_backbone_rope(backbone, meta, patch_size, device)

    if task == "semseg":
        feat = backbone.decode_seg(h_hat, token_res)
        logits = head.predict(feat, scale=int(patch_size))
        pred = torch.argmax(logits, dim=1)
        gt_t = sample["semseg"].squeeze().long()
        return pred, gt_t

    feat = backbone.decode_depth(h_hat, token_res)
    img_h, img_w = int(meta["img_hw"][0]), int(meta["img_hw"][1])
    depth = head.predict(feat, size=(img_h, img_w))
    return depth, sample["depth"] if sample is not None else None


def cmd_replay(args) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    task = args.task
    mode = args.mode
    subset = load_subset(args.subset)
    dataset = build_dataset(task, cfg)
    backbone = build_backbone(cfg, device)
    head = build_head(task, cfg).to(device).eval()
    feat_dir = task_feat_dir(cfg, task)
    token_dir = feat_dir / "tokens"
    meta_dir = feat_dir / "meta"
    stem_index = _build_stem_index(dataset, task, cfg)

    n_prefix = resolve_n_prefix(args.norm_mode, args.n_prefix, cfg.get("n_prefix", 5))
    sideinfo_bpi = norm_sideinfo_bits(args.norm_mode, n_prefix, 0)

    codec = None
    train_tokens = "all"
    if mode == "orfc":
        if not args.ckpt_path:
            _fail("--ckpt_path required for mode=orfc")
        ckpt_meta = load_codec_meta(args.ckpt_path)
        train_tokens = ckpt_meta.get("train_tokens", "all")
        codec = load_codec(args.ckpt_path, device=device)
        print(f"[replay] loaded codec: {args.ckpt_path}")

    prefix_bypass = args.prefix_bypass
    if prefix_bypass is None:
        prefix_bypass = train_tokens == "patch"

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    stems = subset
    if stems is None:
        stems = {p.stem for p in token_dir.glob("*.npy")}

    meter = build_meter(task, cfg)
    total_mse = 0.0
    n_mse = 0
    rate_features: list[np.ndarray] = []

    tag = f"{task}_{mode}"
    if mode == "orfc" and args.ckpt_path:
        ckpt_stem = Path(args.ckpt_path).stem
        tag = f"{task}_orfc_{ckpt_stem}"

    for stem in tqdm(sorted(stems), desc=f"replay-{tag}"):
        tok_path = token_dir / f"{stem}.npy"
        meta_path = meta_dir / f"{stem}.npz"
        if not tok_path.is_file() or not meta_path.is_file():
            _fail(f"Missing token/meta for stem={stem}")

        tokens = np.load(tok_path)
        meta = dict(np.load(meta_path, allow_pickle=True))

        if mode == "orfc":
            recon = encode_decode_single(
                tokens, codec, args.norm_mode, device,
                n_prefix=n_prefix, prefix_bypass=prefix_bypass,
            )
            total_mse += float(np.mean((tokens - recon) ** 2))
            n_mse += 1
            rate_features.append(tokens)
            replay_tokens = recon
        else:
            replay_tokens = tokens

        idx = stem_index.get(stem)
        if idx is None:
            _fail(f"Could not find dataset sample for stem={stem}")
        sample = dataset[idx]

        pred, gt = _replay_one(
            replay_tokens,
            meta,
            task=task,
            backbone=backbone,
            head=head,
            device=device,
            patch_size=cfg["patch_size"],
            sample=sample,
        )
        if task == "semseg":
            meter.update(pred.squeeze(0).cpu().numpy(), gt.squeeze().cpu().numpy())
        else:
            meter.update(pred, gt)

    metrics = meter.compute()
    use_transform = True
    if mode == "orfc" and args.ckpt_path:
        use_transform = ckpt_meta.get(
            "use_transform", ckpt_meta.get("has_transform", True)
        )
    out = {
        "task": task,
        "mode": mode,
        "ckpt_path": args.ckpt_path,
        "norm_mode": args.norm_mode,
        "n_prefix": n_prefix,
        "n_samples": len(stems),
        "train_tokens": train_tokens,
        "prefix_bypass": prefix_bypass,
        "use_transform": use_transform,
        "metrics": metrics,
    }
    if n_mse > 0:
        out["avg_mse"] = total_mse / n_mse

    if mode == "orfc" and codec is not None and rate_features:
        rate = evaluate_rate(
            rate_features,
            codec,
            ckpt_path=args.ckpt_path,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
            embed_dim=cfg["embed_dim"],
            device=device,
            sideinfo_bpi=sideinfo_bpi,
            prefix_bypass=prefix_bypass,
        )
        out["rate"] = rate
        print(f"[replay] BPFP={rate['bpfp']:.4f}  MSE={out.get('avg_mse', 0):.6f}")

    out_path = results_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[replay] {tag}: {metrics} -> {out_path}")


def cmd_rate(args) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if not args.ckpt_path:
        _fail("--ckpt_path required")

    task = args.task
    subset = load_subset(args.subset)
    feat_dir = task_feat_dir(cfg, task)
    token_dir = feat_dir / "tokens"

    n_prefix = resolve_n_prefix(args.norm_mode, args.n_prefix, cfg.get("n_prefix", 5))
    sideinfo_bpi = norm_sideinfo_bits(args.norm_mode, n_prefix, 0)

    codec = load_codec(args.ckpt_path, device=device)
    stems = subset or {p.stem for p in token_dir.glob("*.npy")}
    features = [np.load(token_dir / f"{s}.npy") for s in sorted(stems)]

    rate = evaluate_rate(
        features,
        codec,
        ckpt_path=args.ckpt_path,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
        embed_dim=cfg["embed_dim"],
        device=device,
        sideinfo_bpi=sideinfo_bpi,
    )

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{task}_rate_{Path(args.ckpt_path).stem}"
    out = {"task": task, "ckpt_path": args.ckpt_path, "n_samples": len(features), "rate": rate}
    out_path = results_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[rate] BPFP={rate['bpfp']:.4f} -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="DINOv3 PQFC offline replay/rate")
    ap.add_argument("--config", type=str, default=None)
    sub = ap.add_subparsers(dest="command", required=True)

    rp = sub.add_parser("replay")
    rp.add_argument("--task", choices=["semseg", "depth"], required=True)
    rp.add_argument("--mode", choices=["bypass", "orfc"], default="orfc")
    rp.add_argument("--ckpt_path", type=str, default=None)
    rp.add_argument("--subset", type=str, default=None)
    rp.add_argument("--norm_mode", choices=NORM_MODE_CHOICES,
                    default="split_cls_patch")
    rp.add_argument("--n_prefix", type=int, default=0)
    rp.add_argument("--prefix_bypass", action="store_true", default=None,
                    help="Skip PQ on prefix tokens (default: on for ptpatch ckpts)")
    rp.add_argument("--no_prefix_bypass", dest="prefix_bypass", action="store_false")
    rp.add_argument("--gpu", type=int, default=0)

    rt = sub.add_parser("rate")
    rt.add_argument("--task", choices=["semseg", "depth"], required=True)
    rt.add_argument("--ckpt_path", type=str, required=True)
    rt.add_argument("--subset", type=str, default=None)
    rt.add_argument("--norm_mode", choices=NORM_MODE_CHOICES,
                    default="split_cls_patch")
    rt.add_argument("--n_prefix", type=int, default=0)
    rt.add_argument("--gpu", type=int, default=0)

    args = ap.parse_args()
    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    t0 = time.time()
    if args.command == "replay":
        cmd_replay(args)
    else:
        cmd_rate(args)
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

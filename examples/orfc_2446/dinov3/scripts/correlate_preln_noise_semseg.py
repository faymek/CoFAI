#!/usr/bin/env python3
"""Correlate pre-LN (raw feature) patch MSE with semseg mIoU via Gaussian noise.

Mirrors correlate_postln_noise_semseg.py, but noise is added in pre-LN feature
space (the space OPQ warm-start optimizes after split_cls_patch normalize +
inv_norm round-trip ≈ raw tokens). mIoU uses backbone.decode_seg (applies LN).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

_COFAI_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.entropy_models.soft_pq import load_codec

from lib.config_utils import load_config, resolve_project_root, task_feat_dir
from lib.dataset_utils import (
    build_backbone,
    build_dataset,
    build_head,
    build_meter,
    sample_stem,
)
from lib.orfc_codec import encode_decode_single, resolve_n_prefix
from lib.token_mse_metrics import split_groups


def _build_stem_index(dataset) -> dict[str, int]:
    index: dict[str, int] = {}
    for idx in range(len(dataset)):
        index[sample_stem(dataset[idx], "semseg")] = idx
    return index


def _pre_ln_patch_mse(ref: np.ndarray, noisy: np.ndarray, n_prefix: int) -> float:
    ref_p = split_groups(ref, n_prefix)["patch"]
    noisy_p = split_groups(noisy, n_prefix)["patch"]
    return float(np.mean((noisy_p - ref_p) ** 2))


@torch.inference_mode()
def _miou_from_pre_ln(
    tokens_td: torch.Tensor,
    *,
    meta: dict,
    backbone,
    head,
    device: torch.device,
    patch_size: int,
    sample: dict,
    meter,
) -> None:
    h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
    img = torch.zeros(1, 3, h_p * patch_size, w_p * patch_size, device=device)
    backbone.encode(img)
    feat = backbone.decode_seg(tokens_td, (h_p, w_p))
    logits = head.predict(feat, scale=int(patch_size), token_hw=(h_p, w_p))
    pred = torch.argmax(logits, dim=1)
    gt_t = sample["semseg"].squeeze().long()
    meter.update(pred.squeeze(0).cpu().numpy(), gt_t.squeeze().cpu().numpy())


@torch.inference_mode()
def evaluate_sigma(
    sigma: float,
    samples: list[tuple],
    *,
    n_prefix: int,
    backbone,
    head,
    device: torch.device,
    patch_size: int,
    seed: int,
) -> dict:
    meter = build_meter("semseg", load_config())
    mse_list = []
    for img_idx, (tokens, meta, sample) in enumerate(
        tqdm(samples, desc=f"sigma={sigma:.4f}", leave=False)
    ):
        x = torch.from_numpy(tokens).float().to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        n_patch = int(meta["token_hw"][0]) * int(meta["token_hw"][1])
        if sigma == 0.0:
            noisy = x
        else:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + img_idx)
            noise = torch.randn(
                (x.shape[0], n_patch, x.shape[-1]),
                generator=gen,
                device=device,
                dtype=x.dtype,
            ) * sigma
            noisy = x.clone()
            noisy[:, n_prefix:n_prefix + n_patch, :] += noise

        mse_list.append(
            _pre_ln_patch_mse(
                x.squeeze(0).cpu().numpy(),
                noisy.squeeze(0).cpu().numpy(),
                n_prefix,
            )
        )
        _miou_from_pre_ln(
            noisy, meta=meta, backbone=backbone, head=head,
            device=device, patch_size=patch_size, sample=sample, meter=meter,
        )

    return {
        "sigma": sigma,
        "pre_ln_patch_mse": float(np.mean(mse_list)),
        "pre_ln_patch_rmse": float(np.sqrt(np.mean(mse_list))),
        "mIoU": float(meter.compute()["mIoU"]),
    }


@torch.inference_mode()
def evaluate_codec(
    ckpt_path: str,
    samples: list[tuple],
    *,
    norm_mode: str,
    n_prefix: int,
    prefix_bypass: bool,
    backbone,
    head,
    device: torch.device,
    patch_size: int,
) -> dict:
    codec = load_codec(ckpt_path, device=device)
    meter = build_meter("semseg", load_config())
    mse_list = []
    for tokens, meta, sample in tqdm(samples, desc=Path(ckpt_path).stem, leave=False):
        recon = encode_decode_single(
            tokens, codec, norm_mode, device,
            n_prefix=n_prefix, prefix_bypass=prefix_bypass,
        )
        mse_list.append(_pre_ln_patch_mse(tokens, recon, n_prefix))
        _miou_from_pre_ln(
            torch.from_numpy(recon).unsqueeze(0).to(device=device, dtype=torch.float32),
            meta=meta, backbone=backbone, head=head,
            device=device, patch_size=patch_size, sample=sample, meter=meter,
        )

    del codec
    torch.cuda.empty_cache()
    return {
        "label": Path(ckpt_path).stem,
        "pre_ln_patch_mse": float(np.mean(mse_list)),
        "pre_ln_patch_rmse": float(np.sqrt(np.mean(mse_list))),
        "mIoU": float(meter.compute()["mIoU"]),
        "prefix_bypass": prefix_bypass,
    }


def _corr(xs: np.ndarray, ys: np.ndarray) -> dict:
    if len(xs) < 3:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr = stats.pearsonr(xs, ys)
    sr = stats.spearmanr(xs, ys)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm_mode", default="split_cls_patch")
    ap.add_argument("--n_prefix", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sigmas",
        type=str,
        default="",
        help="Comma-separated pre-LN noise std; empty = auto from patch std",
    )
    ap.add_argument(
        "--baseline_ckpt",
        default=(
            "weights/orfc_2446_dinov3/"
            "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt"
        ),
    )
    ap.add_argument(
        "--ptpatch_ckpt",
        default=(
            "weights/orfc_2446_dinov3/"
            "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt"
        ),
    )
    args = ap.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    cfg = load_config()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    n_prefix = resolve_n_prefix(args.norm_mode, args.n_prefix, cfg.get("n_prefix", 5))
    feat_dir = task_feat_dir(cfg, "semseg")
    token_dir = feat_dir / "tokens"
    meta_dir = feat_dir / "meta"
    dataset = build_dataset("semseg", cfg)
    stem_index = _build_stem_index(dataset)
    stems = sorted(p.stem for p in token_dir.glob("*.npy"))[: args.max_samples]

    samples = []
    for stem in stems:
        tokens = np.load(token_dir / f"{stem}.npy")
        meta = dict(np.load(meta_dir / f"{stem}.npz", allow_pickle=True))
        samples.append((tokens, meta, dataset[stem_index[stem]]))

    backbone = build_backbone(cfg, device)
    head = build_head("semseg", cfg).to(device).eval()
    patch_size = cfg["patch_size"]

    patch_vals = []
    for tokens, _, _ in samples[: min(20, len(samples))]:
        patch_vals.append(tokens[n_prefix:, :].reshape(-1))
    patch_std = float(np.std(np.concatenate(patch_vals)))
    patch_rms = float(np.sqrt(np.mean(np.concatenate(patch_vals) ** 2)))

    if args.sigmas.strip():
        sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
    else:
        # Relative levels similar to post-LN sweep (σ/std ≈ 0..~1.8)
        fracs = [0.0, 0.036, 0.072, 0.11, 0.18, 0.25, 0.36, 0.54, 0.72, 1.1, 1.8]
        sigmas = [round(f * patch_std, 4) for f in fracs]

    noise_rows = [
        evaluate_sigma(
            s, samples,
            n_prefix=n_prefix, backbone=backbone, head=head,
            device=device, patch_size=patch_size, seed=args.seed,
        )
        for s in sigmas
    ]

    codec_rows = [
        evaluate_codec(
            args.baseline_ckpt, samples,
            norm_mode=args.norm_mode, n_prefix=n_prefix, prefix_bypass=False,
            backbone=backbone, head=head, device=device, patch_size=patch_size,
        ),
        evaluate_codec(
            args.ptpatch_ckpt, samples,
            norm_mode=args.norm_mode, n_prefix=n_prefix, prefix_bypass=True,
            backbone=backbone, head=head, device=device, patch_size=patch_size,
        ),
    ]

    bypass_miou = noise_rows[0]["mIoU"]
    xs = np.array([r["pre_ln_patch_mse"] for r in noise_rows])
    ys = np.array([r["mIoU"] for r in noise_rows])
    corr_noise = _corr(xs, ys)

    baseline_mse = codec_rows[0]["pre_ln_patch_mse"]
    nearest_sigma = min(
        sigmas, key=lambda s: abs(s ** 2 - baseline_mse) if s > 0 else 1e18,
    )
    per_image_mse = []
    per_image_miou_drop = []
    for img_idx, (tokens, meta, sample) in enumerate(samples):
        x = torch.from_numpy(tokens).float().to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        n_patch = int(meta["token_hw"][0]) * int(meta["token_hw"][1])
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + img_idx)
        noise = torch.randn(
            (x.shape[0], n_patch, x.shape[-1]),
            generator=gen, device=device, dtype=x.dtype,
        ) * nearest_sigma
        noisy = x.clone()
        noisy[:, n_prefix:n_prefix + n_patch, :] += noise
        mse = _pre_ln_patch_mse(
            x.squeeze(0).cpu().numpy(), noisy.squeeze(0).cpu().numpy(), n_prefix,
        )
        meter = build_meter("semseg", load_config())
        _miou_from_pre_ln(
            noisy, meta=meta, backbone=backbone, head=head,
            device=device, patch_size=patch_size, sample=sample, meter=meter,
        )
        clean_meter = build_meter("semseg", load_config())
        _miou_from_pre_ln(
            x, meta=meta, backbone=backbone, head=head,
            device=device, patch_size=patch_size, sample=sample, meter=clean_meter,
        )
        per_image_mse.append(mse)
        per_image_miou_drop.append(
            clean_meter.compute()["mIoU"] - meter.compute()["mIoU"]
        )
    corr_per_image = _corr(np.array(per_image_mse), np.array(per_image_miou_drop))

    out = {
        "n_samples": len(samples),
        "norm_mode": args.norm_mode,
        "n_prefix": n_prefix,
        "pre_ln_patch_std": patch_std,
        "pre_ln_patch_rms": patch_rms,
        "bypass_miou": bypass_miou,
        "noise_sweep": noise_rows,
        "codec_points": codec_rows,
        "correlation": {
            "noise_sweep_mse_vs_miou": corr_noise,
            "per_image_at_sigma": nearest_sigma,
            "per_image_mse_vs_miou_drop": corr_per_image,
        },
        "notes": {
            "opq_objective": (
                "OPQ warm-start minimizes PQ MSE on batch_normalize_gpu(Y) "
                "(split_cls_patch normalized features = pre-final-LN space). "
                "Orthogonal R ⇒ same as ||Y-Yhat||^2. SoftPQ then switches to "
                "post-LN delta_L_ref for blk23 (norm-only tail)."
            ),
        },
    }

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"preln_noise_corr_semseg_n{len(samples)}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\npre-LN patch token std ≈ {patch_std:.4f}  rms ≈ {patch_rms:.4f}")
    print(f"bypass mIoU = {bypass_miou:.4f}")
    print("\nnoise sweep:")
    print(f"{'sigma':>10}  {'patch_mse':>14}  {'mIoU':>8}")
    for r in noise_rows:
        print(f"{r['sigma']:10.4f}  {r['pre_ln_patch_mse']:14.4f}  {r['mIoU']:8.4f}")
    print("\ncodec points:")
    for r in codec_rows:
        print(
            f"  {r['label'][:50]:50s}  mse={r['pre_ln_patch_mse']:.4f}  "
            f"mIoU={r['mIoU']:.4f}"
        )
    c = corr_noise
    print(
        f"\n[sweep] Pearson(mse,mIoU)={c['pearson_r']:.4f} (p={c['pearson_p']:.2e})  "
        f"Spearman={c['spearman_r']:.4f}"
    )
    c2 = corr_per_image
    print(
        f"[per-image @ sigma={nearest_sigma:.4f}] Pearson(mse,ΔmIoU)="
        f"{c2['pearson_r']:.4f} (p={c2['pearson_p']:.2e})  "
        f"Spearman={c2['spearman_r']:.4f}"
    )
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()

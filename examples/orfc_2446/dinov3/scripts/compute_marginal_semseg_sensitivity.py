#!/usr/bin/env python3
"""Per-PQ-group marginal semseg distortion (restore-one-group + noise control).

Mirrors ORFC compute_sensitivity.py ablation logic, but task metric is ADE20K mIoU:
  - Recon path: fully decoded features as baseline; restore one codec group to original.
  - Noise path: original features as baseline; add Gaussian noise to one group (σ matched
    to per-group quantization RMS in rotated latent space).

Sensitivity[g] = mIoU_ablated - mIoU_baseline  (positive ⇒ group g matters for the task).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from scipy import stats as sp_stats
from tqdm import tqdm

_COFAI_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.entropy_models.orfc_model import batch_inv_normalize_gpu, batch_normalize_gpu
from cofai.entropy_models.soft_pq import load_codec

from lib.config_utils import load_config, resolve_project_root, task_feat_dir
from lib.dataset_utils import build_backbone, build_dataset, build_head, build_meter
from lib.orfc_codec import resolve_n_prefix


@torch.inference_mode()
def _norm_tokens(x: torch.Tensor, backbone, meta: dict, device: torch.device) -> torch.Tensor:
    h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
    img = torch.zeros(1, 3, h_p * 16, w_p * 16, device=device)
    backbone.encode(img)
    h = x if x.dim() == 3 else x.unsqueeze(0)
    return backbone.model.norm(h)


@torch.inference_mode()
def _miou_from_ln_patch(
    ln: torch.Tensor,
    *,
    n_prefix: int,
    token_hw: tuple[int, int],
    head,
    device: torch.device,
    patch_size: int,
    sample: dict,
    meter,
) -> None:
    h_p, w_p = token_hw
    n_patch = h_p * w_p
    patch = ln[:, n_prefix:n_prefix + n_patch, :]
    feat = rearrange(patch, "b (h w) c -> b c h w", h=h_p, w=w_p)
    logits = head.predict([feat], scale=int(patch_size), token_hw=token_hw)
    pred = torch.argmax(logits, dim=1)
    gt_t = sample["semseg"].squeeze().long()
    meter.update(pred.squeeze(0).cpu().numpy(), gt_t.squeeze().cpu().numpy())


def _corr(xs: np.ndarray, ys: np.ndarray) -> dict:
    if len(xs) < 3:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr = sp_stats.pearsonr(xs, ys)
    sr = sp_stats.spearmanr(xs, ys)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def gini(arr: np.ndarray) -> float:
    a = np.abs(np.sort(arr))
    n = len(a)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * a) / (n * np.sum(a) + 1e-30)) - (n + 1) / n)


def summarize_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "cv": float(np.std(arr) / (np.mean(arr) + 1e-10)),
        "gini": gini(arr),
        "values": arr.tolist(),
    }


def _latent_encode(codec, y: torch.Tensor) -> torch.Tensor:
    """Y [B,T,D] normalized -> Z_flat [B*T, D]."""
    b, t, d = y.shape
    flat = y.reshape(b * t, d)
    if codec.transform is not None and hasattr(codec.transform, "get_rotation"):
        return flat @ codec.transform.get_rotation()
    if codec.transform is not None:
        return codec.transform.encode(flat)
    return flat


def _latent_decode(codec, z_flat: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    """Z_flat [B*T, D] -> Y_hat [B,T,D]."""
    b, t, d = shape
    if codec.transform is not None and hasattr(codec.transform, "get_rotation"):
        y_hat = z_flat @ codec.transform.get_rotation().t()
    elif codec.transform is not None:
        y_hat = codec.transform.decode(z_flat)
    else:
        y_hat = z_flat
    return y_hat.reshape(b, t, d)


def _to_feature_space(
    codec,
    z_flat: torch.Tensor,
    shape: tuple[int, int, int],
    mu: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    y_hat = _latent_decode(codec, z_flat, shape)
    return batch_inv_normalize_gpu(y_hat, mu, std)


@torch.inference_mode()
def _eval_semseg_miou(
    tokens_td: torch.Tensor,
    meta: dict,
    *,
    backbone,
    head,
    meter,
    device: torch.device,
    patch_size: int,
    sample: dict,
) -> None:
    h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
    token_res = (h_p, w_p)
    img = torch.zeros(1, 3, h_p * patch_size, w_p * patch_size, device=device)
    backbone.encode(img)
    feat = backbone.decode_seg(tokens_td, token_res)
    logits = head.predict(feat, scale=int(patch_size))
    pred = torch.argmax(logits, dim=1)
    gt_t = sample["semseg"].squeeze().long()
    meter.update(pred.squeeze(0).cpu().numpy(), gt_t.squeeze().cpu().numpy())


def _build_stem_index(dataset) -> dict[str, int]:
    from lib.dataset_utils import sample_stem

    index: dict[str, int] = {}
    for idx in range(len(dataset)):
        stem = sample_stem(dataset[idx], "semseg")
        index[stem] = idx
    return index


@torch.inference_mode()
def run_codec_sensitivity(
    codec,
    samples: list[tuple[str, np.ndarray, dict, dict]],
    *,
    norm_mode: str,
    n_prefix: int,
    backbone,
    head,
    device: torch.device,
    patch_size: int,
    seed: int,
    noise_modes: tuple[str, ...] = ("latent", "postln"),
) -> dict:
    pq = codec.pq
    g, d = pq.G, pq.d
    codec.eval()

    meters = {
        "bypass": build_meter("semseg", load_config()),
        "recon_base": build_meter("semseg", load_config()),
        **{f"restore_{i}": build_meter("semseg", load_config()) for i in range(g)},
    }
    if "latent" in noise_modes:
        meters.update({f"latent_noise_{i}": build_meter("semseg", load_config()) for i in range(g)})
    if "postln" in noise_modes:
        meters["bypass_postln"] = build_meter("semseg", load_config())
        meters.update({f"postln_noise_{i}": build_meter("semseg", load_config()) for i in range(g)})
    if "preln" in noise_modes:
        meters.update({f"preln_noise_{i}": build_meter("semseg", load_config()) for i in range(g)})

    per_group_sqerr = np.zeros(g, dtype=np.float64)
    per_group_postln_sqerr = np.zeros(g, dtype=np.float64)
    per_group_preln_sqerr = np.zeros(g, dtype=np.float64)
    n_tokens = 0
    n_postln_elems = 0
    n_preln_elems = 0
    z_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int], dict, dict]] = []

    for stem, tokens, meta, sample in tqdm(samples, desc="encode-cache", leave=False):
        x = torch.from_numpy(tokens).unsqueeze(0).to(device=device, dtype=torch.float32)
        shape = tuple(x.shape)
        y, mu, std = batch_normalize_gpu(x, mode=norm_mode, n_prefix=n_prefix)
        z = _latent_encode(codec, y)
        z_hat, _ = pq._quantise(z)

        z_g = z.reshape(-1, g, d)
        z_hat_g = z_hat.reshape(-1, g, d)
        err = (z_g - z_hat_g) ** 2
        per_group_sqerr += err.mean(dim=0).sum(dim=1).detach().double().cpu().numpy()
        n_tokens += z_g.shape[0]

        need_recon = "postln" in noise_modes or "preln" in noise_modes
        if need_recon:
            x_recon = _to_feature_space(codec, z_hat, shape, mu, std)
            h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
            n_patch = h_p * w_p
            if "postln" in noise_modes:
                ln_recon = _norm_tokens(x_recon, backbone, meta, device)
            for gi in range(g):
                oracle = z_hat_g.clone()
                oracle[:, gi, :] = z_g[:, gi, :]
                x_fix = _to_feature_space(
                    codec, oracle.reshape(-1, g * d), shape, mu, std,
                )
                if "postln" in noise_modes:
                    ln_fix = _norm_tokens(x_fix, backbone, meta, device)
                    delta = (
                        ln_fix[:, n_prefix:n_prefix + n_patch, :]
                        - ln_recon[:, n_prefix:n_prefix + n_patch, :]
                    )
                    per_group_postln_sqerr[gi] += float((delta ** 2).sum().item())
                    n_postln_elems += delta.numel()
                if "preln" in noise_modes:
                    delta_pre = (
                        x_fix[:, n_prefix:n_prefix + n_patch, :]
                        - x_recon[:, n_prefix:n_prefix + n_patch, :]
                    )
                    per_group_preln_sqerr[gi] += float((delta_pre ** 2).sum().item())
                    n_preln_elems += delta_pre.numel()

        z_cache.append((x, y, mu, std, z, z_hat, shape, meta, sample))

    noise_sigma_latent = np.sqrt(per_group_sqerr / max(n_tokens, 1))
    noise_sigma_postln = np.sqrt(per_group_postln_sqerr / max(n_postln_elems, 1))
    noise_sigma_preln = np.sqrt(per_group_preln_sqerr / max(n_preln_elems, 1))

    for img_idx, (x, y, mu, std, z, z_hat, shape, meta, sample) in enumerate(
        tqdm(z_cache, desc="ablate-miou", leave=False)
    ):
        b, t, c = shape
        g_count, d_dim = pq.G, pq.d
        z_g = z.reshape(-1, g_count, d_dim)
        z_hat_g = z_hat.reshape(-1, g_count, d_dim)

        variants: list[tuple[str, torch.Tensor]] = []
        variants.append(("bypass", x))
        x_recon = _to_feature_space(codec, z_hat, shape, mu, std)
        variants.append(("recon_base", x_recon))

        for gi in range(g_count):
            oracle = z_hat_g.clone()
            oracle[:, gi, :] = z_g[:, gi, :]
            x_oracle = _to_feature_space(codec, oracle.reshape(-1, g_count * d_dim), shape, mu, std)
            variants.append((f"restore_{gi}", x_oracle))

        if "latent" in noise_modes:
            for gi in range(g_count):
                z_noisy = z.reshape(-1, g_count, d_dim).clone()
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + img_idx * g_count + gi)
                noise = torch.randn(
                    z_noisy[:, gi, :].shape,
                    generator=gen,
                    device=device,
                    dtype=z_noisy.dtype,
                ) * float(noise_sigma_latent[gi])
                z_noisy[:, gi, :] = z_noisy[:, gi, :] + noise
                x_noisy = _to_feature_space(
                    codec, z_noisy.reshape(-1, g_count * d_dim), shape, mu, std,
                )
                variants.append((f"latent_noise_{gi}", x_noisy))

        for key, feat in variants:
            _eval_semseg_miou(
                feat, meta,
                backbone=backbone, head=head, meter=meters[key],
                device=device, patch_size=patch_size, sample=sample,
            )

        if "postln" in noise_modes:
            ln_ref = _norm_tokens(x, backbone, meta, device)
            _miou_from_ln_patch(
                ln_ref,
                n_prefix=n_prefix,
                token_hw=(int(meta["token_hw"][0]), int(meta["token_hw"][1])),
                head=head,
                device=device,
                patch_size=patch_size,
                sample=sample,
                meter=meters["bypass_postln"],
            )
            h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
            n_patch = h_p * w_p
            for gi in range(g_count):
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + 100_000 + img_idx * g_count + gi)
                noise = torch.randn(
                    (ln_ref.shape[0], n_patch, ln_ref.shape[-1]),
                    generator=gen,
                    device=device,
                    dtype=ln_ref.dtype,
                ) * float(noise_sigma_postln[gi])
                ln_noisy = ln_ref.clone()
                ln_noisy[:, n_prefix:n_prefix + n_patch, :] += noise
                _miou_from_ln_patch(
                    ln_noisy,
                    n_prefix=n_prefix,
                    token_hw=(h_p, w_p),
                    head=head,
                    device=device,
                    patch_size=patch_size,
                    sample=sample,
                    meter=meters[f"postln_noise_{gi}"],
                )

        if "preln" in noise_modes:
            h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
            n_patch = h_p * w_p
            for gi in range(g_count):
                gen = torch.Generator(device=device)
                gen.manual_seed(seed + 200_000 + img_idx * g_count + gi)
                noise = torch.randn(
                    (x.shape[0], n_patch, x.shape[-1]),
                    generator=gen,
                    device=device,
                    dtype=x.dtype,
                ) * float(noise_sigma_preln[gi])
                x_noisy = x.clone()
                x_noisy[:, n_prefix:n_prefix + n_patch, :] += noise
                _eval_semseg_miou(
                    x_noisy, meta,
                    backbone=backbone, head=head, meter=meters[f"preln_noise_{gi}"],
                    device=device, patch_size=patch_size, sample=sample,
                )

    bypass_miou = meters["bypass"].compute()["mIoU"]
    recon_miou = meters["recon_base"].compute()["mIoU"]
    restore_miou = np.array([meters[f"restore_{i}"].compute()["mIoU"] for i in range(g)])

    restore_sens = np.maximum(restore_miou - recon_miou, 0.0)

    out: dict = {
        "G": g,
        "d": d,
        "mIoU": {
            "bypass": float(bypass_miou),
            "recon_base": float(recon_miou),
            "restore_per_group": restore_miou.tolist(),
        },
        "marginal_distortion": {
            "restore_one_group": summarize_stats(restore_sens),
        },
        "restore_sensitivity": restore_sens.tolist(),
    }

    if "latent" in noise_modes:
        latent_miou = np.array([
            meters[f"latent_noise_{i}"].compute()["mIoU"] for i in range(g)
        ])
        latent_sens = np.maximum(bypass_miou - latent_miou, 0.0)
        out["noise_sigma_latent"] = noise_sigma_latent.tolist()
        out["mIoU"]["latent_noise_per_group"] = latent_miou.tolist()
        out["marginal_distortion"]["latent_noise_one_group"] = summarize_stats(latent_sens)
        out["latent_noise_sensitivity"] = latent_sens.tolist()
        out["correlation_latent_sigma_vs_sens"] = _corr(
            noise_sigma_latent ** 2, latent_sens,
        )

    if "postln" in noise_modes:
        bypass_postln_miou = meters["bypass_postln"].compute()["mIoU"]
        postln_miou = np.array([
            meters[f"postln_noise_{i}"].compute()["mIoU"] for i in range(g)
        ])
        postln_sens = np.maximum(bypass_postln_miou - postln_miou, 0.0)
        out["noise_sigma_postln"] = noise_sigma_postln.tolist()
        out["noise_sigma_postln_sq"] = (noise_sigma_postln ** 2).tolist()
        out["mIoU"]["bypass_postln"] = float(bypass_postln_miou)
        out["mIoU"]["postln_noise_per_group"] = postln_miou.tolist()
        out["marginal_distortion"]["postln_noise_one_group"] = summarize_stats(postln_sens)
        out["postln_noise_sensitivity"] = postln_sens.tolist()
        out["correlation_postln_mse_vs_sens"] = _corr(
            noise_sigma_postln ** 2, postln_sens,
        )

    if "preln" in noise_modes:
        preln_miou = np.array([
            meters[f"preln_noise_{i}"].compute()["mIoU"] for i in range(g)
        ])
        preln_sens = np.maximum(bypass_miou - preln_miou, 0.0)
        out["noise_sigma_preln"] = noise_sigma_preln.tolist()
        out["noise_sigma_preln_sq"] = (noise_sigma_preln ** 2).tolist()
        out["mIoU"]["preln_noise_per_group"] = preln_miou.tolist()
        out["marginal_distortion"]["preln_noise_one_group"] = summarize_stats(preln_sens)
        out["preln_noise_sensitivity"] = preln_sens.tolist()
        out["correlation_preln_mse_vs_sens"] = _corr(
            noise_sigma_preln ** 2, preln_sens,
        )

    # Backward-compatible aliases (latent path)
    if "latent" in noise_modes:
        out["noise_sigma"] = out["noise_sigma_latent"]
        out["noise_sensitivity"] = out["latent_noise_sensitivity"]
        out["mIoU"]["noise_per_group"] = out["mIoU"]["latent_noise_per_group"]
        out["marginal_distortion"]["noise_one_group"] = out["marginal_distortion"][
            "latent_noise_one_group"
        ]

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-group marginal semseg mIoU sensitivity")
    ap.add_argument(
        "--ckpts",
        nargs="+",
        default=[
            "weights/orfc_2446_dinov3/blk23_K64_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt",
            "weights/orfc_2446_dinov3/blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt",
        ],
    )
    ap.add_argument("--norm_mode", default="split_cls_patch")
    ap.add_argument("--n_prefix", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--noise_modes",
        default="latent,postln",
        help="Comma-separated: latent, postln, preln (pre-LN raw feature group noise)",
    )
    args = ap.parse_args()
    noise_modes = tuple(m.strip() for m in args.noise_modes.split(",") if m.strip())

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
    stems = sorted(p.stem for p in token_dir.glob("*.npy"))[: args.n_samples]

    samples: list[tuple[str, np.ndarray, dict, dict]] = []
    for stem in stems:
        tokens = np.load(token_dir / f"{stem}.npy")
        meta = dict(np.load(meta_dir / f"{stem}.npz", allow_pickle=True))
        idx = stem_index[stem]
        samples.append((stem, tokens, meta, dataset[idx]))

    backbone = build_backbone(cfg, device)
    head = build_head("semseg", cfg).to(device).eval()

    print(f"[sensitivity] n={len(samples)}  n_prefix={n_prefix}  gpu={args.gpu}")

    all_results: dict = {
        "config": {
            "n_samples": len(samples),
            "stems": stems,
            "norm_mode": args.norm_mode,
            "n_prefix": n_prefix,
            "seed": args.seed,
            "definition": {
                "restore_sensitivity": "max(0, mIoU(recon with group g restored) - mIoU(full recon))",
                "latent_noise_sensitivity": (
                    "max(0, mIoU(bypass decode_seg) - mIoU(decode after Z-group noise))"
                ),
                "postln_noise_sensitivity": (
                    "max(0, mIoU(clean post-LN patch) - mIoU(post-LN patch + group-g noise))"
                ),
                "preln_noise_sensitivity": (
                    "max(0, mIoU(bypass decode_seg) - mIoU(decode after pre-LN patch group-g noise))"
                ),
                "noise_sigma_latent": "per-group RMS of (Z - Z_hat) in rotated latent space",
                "noise_sigma_postln": (
                    "per-group RMS of isolated post-LN patch error from restoring group g"
                ),
                "noise_sigma_preln": (
                    "per-group RMS of isolated pre-LN (raw) patch error from restoring group g"
                ),
            },
            "noise_modes": list(noise_modes),
        },
        "codecs": {},
    }

    t0 = time.time()
    for ckpt_path in args.ckpts:
        tag = Path(ckpt_path).stem
        print(f"\n{'=' * 60}\n  codec: {tag}")
        codec = load_codec(ckpt_path, device=device)
        res = run_codec_sensitivity(
            codec,
            samples,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
            backbone=backbone,
            head=head,
            device=device,
            patch_size=cfg["patch_size"],
            seed=args.seed,
            noise_modes=noise_modes,
        )
        res["ckpt_path"] = ckpt_path
        all_results["codecs"][tag] = res

        rs = res["marginal_distortion"]["restore_one_group"]
        print(
            f"  mIoU bypass={res['mIoU']['bypass']:.4f}  "
            f"recon={res['mIoU']['recon_base']:.4f}"
        )
        print(
            f"  restore sens: mean={rs['mean']:.6f}  CV={rs['cv']:.4f}  Gini={rs['gini']:.4f}"
        )
        if "latent_noise_one_group" in res["marginal_distortion"]:
            ns = res["marginal_distortion"]["latent_noise_one_group"]
            c = res.get("correlation_latent_sigma_vs_sens", {})
            print(
                f"  latent noise: mean={ns['mean']:.6f}  CV={ns['cv']:.4f}  "
                f"Gini={ns['gini']:.4f}  corr(sigma²,sens)={c.get('pearson_r')}"
            )
        if "postln_noise_one_group" in res["marginal_distortion"]:
            ps = res["marginal_distortion"]["postln_noise_one_group"]
            c = res.get("correlation_postln_mse_vs_sens", {})
            print(
                f"  postln noise: bypass_ln={res['mIoU']['bypass_postln']:.4f}  "
                f"mean={ps['mean']:.6f}  CV={ps['cv']:.4f}  Gini={ps['gini']:.4f}  "
                f"corr(mse,sens)={c.get('pearson_r')}"
            )
        if "preln_noise_one_group" in res["marginal_distortion"]:
            pr = res["marginal_distortion"]["preln_noise_one_group"]
            c = res.get("correlation_preln_mse_vs_sens", {})
            print(
                f"  preln noise: mean={pr['mean']:.6f}  CV={pr['cv']:.4f}  "
                f"Gini={pr['gini']:.4f}  corr(mse,sens)={c.get('pearson_r')}"
            )
        del codec
        torch.cuda.empty_cache()

    # Cross-codec imbalance comparison
    rows = []
    for tag, res in all_results["codecs"].items():
        rows.append({
            "codec": tag,
            "restore_cv": res["marginal_distortion"]["restore_one_group"]["cv"],
            "restore_gini": res["marginal_distortion"]["restore_one_group"]["gini"],
            "latent_noise_cv": res["marginal_distortion"].get(
                "latent_noise_one_group", {}
            ).get("cv"),
            "latent_noise_gini": res["marginal_distortion"].get(
                "latent_noise_one_group", {}
            ).get("gini"),
            "postln_noise_cv": res["marginal_distortion"].get(
                "postln_noise_one_group", {}
            ).get("cv"),
            "postln_noise_gini": res["marginal_distortion"].get(
                "postln_noise_one_group", {}
            ).get("gini"),
            "preln_noise_cv": res["marginal_distortion"].get(
                "preln_noise_one_group", {}
            ).get("cv"),
            "preln_noise_gini": res["marginal_distortion"].get(
                "preln_noise_one_group", {}
            ).get("gini"),
        })
    all_results["imbalance_summary"] = rows

    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_suffix = "_preln" if set(noise_modes) == {"preln"} else ""
    out_path = out_dir / f"marginal_semseg_sensitivity{out_suffix}_n{len(samples)}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(
        f"{'codec':<45} {'R_CV':>7} {'R_Gini':>7} {'L_CV':>7} {'L_Gini':>7} "
        f"{'P_CV':>7} {'P_Gini':>7} {'Pre_CV':>7} {'Pre_G':>7}"
    )
    for row in rows:
        print(
            f"{row['codec']:<45} {row['restore_cv']:>7.4f} {row['restore_gini']:>7.4f} "
            f"{(row['latent_noise_cv'] or 0):>7.4f} {(row['latent_noise_gini'] or 0):>7.4f} "
            f"{(row['postln_noise_cv'] or 0):>7.4f} {(row['postln_noise_gini'] or 0):>7.4f} "
            f"{(row['preln_noise_cv'] or 0):>7.4f} {(row['preln_noise_gini'] or 0):>7.4f}"
        )
    print(f"\nSaved -> {out_path}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline SoftPQ training on DINOv3 ViT-L/16 blk23 features (FrozenTail).

Supports large feature corpora via path-mode lazy loading (no full RAM stack).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_COFAI_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.backbone import Dinov3TimmBackbone
from cofai.ops.orfc import (
    batched_assign,
    learn_orfc_rotation,
)
from examples.orfc_2446.offline.soft_pq import (
    OrthogonalTransform,
    save_codec,
    train_soft_pq,
)
from examples.orfc_2446.offline.artifacts import (
    compute_histogram_pmf,
    save_codec_npz,
)
from cofai.latent_codecs.orfc_normalization import normalize_orfc_features

from lib.config_utils import load_config, resolve_project_root
from lib.distortion_metrics import eval_val_distortion
from lib.orfc_codec import NORM_MODE_CHOICES, resolve_norm_settings

from examples.orfc.offline.utils import preload_features, set_seed  # noqa: E402


def _split_train_val_paths(
    feat_dir: Path,
    max_train: int,
    n_val: int,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Shuffle file paths only — do not preload train features into RAM."""
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {feat_dir}")

    rng = np.random.RandomState(seed)
    n_total = len(files)
    n_need = min(n_total, max_train + n_val) if max_train > 0 else n_total
    if n_need < n_total:
        pick = rng.choice(n_total, n_need, replace=False)
        files = [files[i] for i in sorted(pick)]

    perm = rng.permutation(len(files))
    n_val = min(n_val, len(files))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if max_train > 0:
        train_idx = train_idx[:max_train]
    train_paths = [files[i] for i in train_idx]
    val_paths = [files[i] for i in val_idx]
    return train_paths, val_paths


def _parse_token_hw(s: str | None, default: tuple[int, int]) -> tuple[int, int]:
    if not s:
        return default
    parts = [int(x) for x in s.replace("x", ",").split(",")]
    if len(parts) != 2:
        raise ValueError(f"--token_hw expects H,W got {s!r}")
    return parts[0], parts[1]


def _infer_token_hw_from_feat(path: Path, n_prefix: int) -> tuple[int, int] | None:
    """Factor T - n_prefix into (H, W); prefer known ADE/COCO layouts."""
    arr = np.load(path, mmap_mode="r")
    n_patch = int(arr.shape[0]) - n_prefix
    if n_patch <= 0:
        return None
    known = {(32, 43), (43, 32), (64, 85), (85, 64), (16, 16), (32, 32), (14, 14)}
    for h, w in known:
        if h * w == n_patch:
            return (h, w)
    # Nearest-to-square factorization
    best = None
    best_score = 1e18
    for h in range(1, int(n_patch**0.5) + 1):
        if n_patch % h == 0:
            w = n_patch // h
            score = abs(h - w)
            if score < best_score:
                best_score = score
                best = (h, w)
    return best


def _ckpt_name(args, bt_dim: int, D: int) -> str:
    if args.use_transform:
        bt_tag = f"bt{bt_dim}"
        ws_tag = "ws" if args.warm_start_opq else "km"
    else:
        bt_tag = "noR"
        ws_tag = "km"
    mse_tag = "_mse" if args.mse_loss else ""
    rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    norm_tag = f"_{args.norm_mode}" if args.norm_mode != "per_image" else ""
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    return (
        f"{args.layer}_K{args.K}_emb{args.embedding_dim}_{bt_tag}_{ws_tag}"
        f"{mse_tag}{rate_tag}{norm_tag}{tau_tag}"
        f"_lr{args.lr}_ep{args.epochs}_n{args.max_train}_s{args.seed}.pt"
    )


def _norm_vectors_from_batch(Y: torch.Tensor, D: int) -> np.ndarray:
    return Y.reshape(-1, D).cpu().numpy()


def _sample_opq_vectors(
    train_paths: list[Path],
    *,
    max_flat: int,
    norm_mode: str,
    n_prefix: int,
    D: int,
    seed: int,
    device: torch.device,
    chunk_images: int = 32,
) -> np.ndarray:
    """Stream-normalize images until ``max_flat`` token vectors are collected."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train_paths))
    chunks: list[np.ndarray] = []
    n_tok = 0
    n_img = 0
    for start in range(0, len(order), chunk_images):
        if n_tok >= max_flat:
            break
        batch_paths = [train_paths[int(i)] for i in order[start : start + chunk_images]]
        batch = np.stack([np.load(p).astype(np.float32) for p in batch_paths])
        X = torch.from_numpy(batch).float().to(device)
        with torch.no_grad():
            Y, _, _ = normalize_orfc_features(X, mode=norm_mode, n_prefix=n_prefix)
        vecs = _norm_vectors_from_batch(Y, D)
        chunks.append(vecs)
        n_tok += vecs.shape[0]
        n_img += len(batch_paths)
        del X, Y, batch
    full = np.concatenate(chunks, axis=0)
    del chunks
    if full.shape[0] > max_flat:
        full = full[rng.choice(full.shape[0], max_flat, replace=False)]
    print(f"  OPQ sample: {n_img} images → {full.shape[0]} token vectors " f"(cap={max_flat})")
    return full


def _iter_path_batches(paths: list[Path], batch_size: int):
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        yield np.stack([np.load(p).astype(np.float32) for p in batch_paths])


def train_and_save(args) -> str:
    cfg = load_config(Path(args.config) if args.config else None)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])
    norm_mode, n_prefix, _ = resolve_norm_settings(
        norm_mode=args.norm_mode,
        n_prefix=args.n_prefix,
        default_n_prefix=cfg.get("n_prefix", 5),
        fallback_norm=cfg.get("train_defaults", {}).get("norm_mode", "split_cls_patch"),
    )
    args.norm_mode = norm_mode
    default_hw = tuple(cfg.get("train_token_hw", [32, 43]))
    D = cfg["embed_dim"]
    G = D // args.embedding_dim

    feat_dir = Path(args.feat_dir or cfg["paths"]["train_feat_dir"])
    weights_dir = Path(args.weights_dir or cfg["paths"]["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# DINOv3 SoftPQ Training  layer={args.layer}  K={args.K}  d={args.embedding_dim}")
    print(f"# norm={args.norm_mode}  n_prefix={n_prefix}  λ={args.lmbda}")
    print(f"# use_transform={args.use_transform}")
    print(f"# feat_dir={feat_dir}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    train_paths, val_paths = _split_train_val_paths(
        feat_dir,
        args.max_train,
        args.n_val,
        args.seed,
    )
    sample = np.load(train_paths[0])
    feat_shape = sample.shape
    del sample
    token_hw = _parse_token_hw(args.token_hw, default_hw)
    if args.token_hw is None:
        inferred = _infer_token_hw_from_feat(train_paths[0], n_prefix)
        if inferred is not None:
            token_hw = inferred
    print(
        f"  train={len(train_paths)}  val={len(val_paths)}  "
        f"shape={feat_shape}  token_hw={token_hw}  (path-mode lazy)"
    )

    # Val is small — preload for metrics / SoftPQ val loop.
    val_feats, _ = preload_features(val_paths, num_workers=min(8, max(1, len(val_paths))))

    # ---- Backbone + frozen tail ----
    tail = None
    if not args.mse_loss:
        print(f"\nLoading DINOv3 backbone for frozen tail (layer_idx={layer_idx})...")
        backbone = Dinov3TimmBackbone(
            model_size="large",
            img_size=512,
            patch_size=cfg["patch_size"],
            dynamic_size=True,
            slot=cfg["slot"],
            n_last_blocks=1,
            cast_dtype="float32",
            pretrained=False,
            ckpt_path=cfg["paths"]["backbone"],
            device=str(device),
        ).eval()
        tail = backbone.build_frozen_tail(layer_idx, token_hw, device)
        n_tail = len(getattr(tail, "blocks", []))
        precompute_teacher = n_tail > 0
        print(f"  tail blocks={n_tail} (+ norm)")
        if not precompute_teacher:
            print("  teacher: on-the-fly norm (skip CPU precompute)")
        del backbone
        torch.cuda.empty_cache()
    else:
        precompute_teacher = True

    opq_usage = None
    if args.use_transform:
        print(f"\n{'=' * 60}\n  [OPQ warm-start]\n{'=' * 60}")
        t0 = time.time()
        # Keep historical cap: kmeans_max_samples // G full-dim vectors.
        max_flat = max(args.kmeans_max_samples // G, 1)
        full_vecs = _sample_opq_vectors(
            train_paths,
            max_flat=max_flat,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
            D=D,
            seed=args.seed,
            device=device,
        )
        R_opq, cb_opq, _ = learn_orfc_rotation(
            full_vecs,
            G,
            args.embedding_dim,
            args.K,
            max_iter_orfc=20,
            max_iter_kmeans=100,
            device=device,
            verbose=False,
        )
        del full_vecs
        torch.cuda.empty_cache()
        print(f"  OPQ done ({time.time() - t0:.1f}s)")

        if args.lmbda > 0 and args.warm_start_opq:
            R_t = torch.from_numpy(R_opq).float().to(device)
            cb_t = torch.from_numpy(np.stack(cb_opq)).float().to(device)
            opq_usage = np.zeros((G, args.K), dtype=np.float64)
            for batch in _iter_path_batches(train_paths, 32):
                X = torch.from_numpy(batch).float().to(device)
                with torch.no_grad():
                    Y, _, _ = normalize_orfc_features(
                        X,
                        mode=args.norm_mode,
                        n_prefix=n_prefix,
                    )
                    flat = Y.reshape(-1, D) @ R_t
                    sub = flat.reshape(-1, G, args.embedding_dim).permute(1, 0, 2).contiguous()
                    lbl = batched_assign(sub, cb_t, device=device)[1].cpu().numpy()
                    for g in range(G):
                        np.add.at(opq_usage[g], lbl[g], 1)
                del X, Y, batch
            del R_t, cb_t
            torch.cuda.empty_cache()
    else:
        if args.warm_start_opq:
            print("  [init] --no_transform: ignoring --warm_start_opq (k-means in train_soft_pq)")
        print(f"\n{'=' * 60}\n  [init] no orthogonal R; k-means in train_soft_pq\n{'=' * 60}")
        R_opq = cb_opq = None

    # ---- SoftPQ training (path list → lazy FeatureDataset) ----
    if args.use_transform:
        R_ws = R_opq.copy()
        C_ws = [c.copy() for c in cb_opq]
        if np.linalg.det(R_ws) < 0:
            R_ws[:, -1] *= -1
            C_ws[-1][:, -1] *= -1
        transform = OrthogonalTransform(D)
    else:
        R_ws = C_ws = None
        transform = None
    loss_name = "MSE" if args.mse_loss else "delta_L_ref"
    print(f"\n{'=' * 60}\n  [SoftPQ] {loss_name}  epochs={args.epochs}\n{'=' * 60}")

    train_path_strs = [str(p) for p in train_paths]
    t_spq = time.time()
    codec, history = train_soft_pq(
        features_train=train_path_strs,
        tail=tail,
        G=G,
        K=args.K,
        d=args.embedding_dim,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        val_features=val_feats,
        transform=transform,
        R_init=R_ws,
        codebooks_init=C_ws,
        use_mse_loss=args.mse_loss,
        lmbda=args.lmbda,
        prior_init_counts=opq_usage,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
        kmeans_max_samples=args.kmeans_max_samples,
        grad_clip=args.grad_clip,
        prior_floor=args.prior_floor,
        precompute_teacher=precompute_teacher,
    )
    print(f"  Training done ({time.time() - t_spq:.1f}s)")

    ckpt_path = weights_dir / _ckpt_name(args, D, D)
    if getattr(args, "keep_pt", False):
        save_codec(
            codec,
            str(ckpt_path),
            use_transform=args.use_transform,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
        )
        print(f"  Resume PT → {ckpt_path}")

    hist_path = ckpt_path.with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    val_metrics = eval_val_distortion(
        val_feats,
        codec,
        tail,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
        device=device,
        batch_size=args.batch_size,
    )
    val_metrics_path = ckpt_path.with_suffix(".val_metrics.json")
    with open(val_metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)

    if tail is not None:
        tail.to("cpu")
    del tail
    torch.cuda.empty_cache()

    pmf = compute_histogram_pmf(
        codec,
        train_path_strs,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
        device=device,
        batch_size=args.batch_size,
    )
    npz_path = save_codec_npz(
        codec,
        ckpt_path.with_suffix(".npz"),
        pmf,
        source_pt=str(ckpt_path) if getattr(args, "keep_pt", False) else None,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
        metadata={
            "proposal": "ORFC-2446",
            "backbone": "dinov3_vitl16",
            "layer": args.layer,
            "slot": layer_idx + 1,
        },
    )
    print(f"  Official NPZ → {npz_path}")
    print(f"  meta: norm_mode={args.norm_mode} n_prefix={n_prefix}")
    torch.cuda.empty_cache()

    print(f"  Val raw MSE = {val_metrics['raw_mse']:.6f}")
    if val_metrics.get("post_ln_patch_mse") is not None:
        print(f"  Val post-LN patch MSE (delta_L_ref) = {val_metrics['post_ln_patch_mse']:.6f}")
        last_val = history[-1].get("val_loss") if history else None
        if last_val is not None:
            n_tok = val_feats[0].shape[0]
            implied_per_elem = last_val / max(n_tok * D, 1)
            print(f"  (train val_loss ep_last={last_val:.1f}  " f"≈{implied_per_elem:.6f} per-element)")
    print(f"  Val metrics → {val_metrics_path}")
    print(f"  Val MSE = {val_metrics['raw_mse']:.6f}")
    return str(npz_path)


def main():
    defaults = load_config().get("train_defaults", {})
    p = argparse.ArgumentParser(description="DINOv3 blk23 SoftPQ offline training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--layer", type=str, default="blk23")
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--embedding_dim", type=int, default=defaults.get("embedding_dim", 32))
    p.add_argument("--norm_mode", choices=NORM_MODE_CHOICES, default=defaults.get("norm_mode", "split_cls_patch"))
    p.add_argument("--n_prefix", type=int, default=0)
    p.add_argument(
        "--token_hw",
        type=str,
        default=None,
        help="Patch grid H,W (e.g. 32,43 for ADE pad688x512). " "Auto-inferred from first feature when omitted.",
    )
    p.add_argument("--epochs", type=int, default=defaults.get("epochs", 30))
    p.add_argument("--lr", type=float, default=defaults.get("lr", 3e-4))
    p.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 4))
    p.add_argument("--lmbda", type=float, default=defaults.get("lmbda", 0.0))
    p.add_argument("--tau_start", type=float, default=defaults.get("tau_start", 0.5))
    p.add_argument("--tau_end", type=float, default=defaults.get("tau_end", 0.005))
    p.add_argument("--tau_schedule", default="exponential")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--prior_floor", type=float, default=0.0)
    p.add_argument("--warm_start_opq", action="store_true", default=True)
    p.add_argument("--no_warm_start", dest="warm_start_opq", action="store_false")
    p.add_argument("--mse_loss", action="store_true")
    p.add_argument("--max_train", type=int, default=defaults.get("max_train", 1000))
    p.add_argument("--n_val", type=int, default=defaults.get("n_val", 50))
    p.add_argument("--kmeans_max_samples", type=int, default=defaults.get("kmeans_max_samples", 2_000_000))
    p.add_argument("--use_transform", action="store_true", default=True, help="Learn orthogonal R + PQ (default)")
    p.add_argument(
        "--no_transform",
        dest="use_transform",
        action="store_false",
        help="No orthogonal R; k-means init + train codebooks only",
    )
    p.add_argument("--feat_dir", type=str, default=None)
    p.add_argument("--weights_dir", type=str, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    p.add_argument(
        "--keep_pt",
        action="store_true",
        help="Also save intermediate .pt for resume (eval uses .npz only)",
    )
    args = p.parse_args()
    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    train_and_save(args)


if __name__ == "__main__":
    main()

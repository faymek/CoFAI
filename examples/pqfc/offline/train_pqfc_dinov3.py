#!/usr/bin/env python3
"""Offline PQFC training on DINOv3 ViT-L/16 features (CTC DINOv3).

Orthogonal transform switch binds soft/hard assignment:
  --use_transform (default): OrthogonalTransform + OPQ warm-start + soft PQ
  --no_transform:            no R; k-means init; hard PQ (tau_start=0)

Loss: J = R + λ · D, D = frozen-tail ΔL_ref (not feature-space MSE).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_COFAI_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.backbone import Dinov3TimmBackbone
from cofai.entropy_models.orfc_model import (
    batched_assign,
    batch_normalize_gpu,
    learn_orfc_rotation,
)
from cofai.entropy_models.soft_pq import (
    OrthogonalTransform,
    save_codec,
    train_soft_pq,
)
from cofai.entropy_models.soft_pq_export import (
    compute_histogram_pmf,
    save_codec_npz,
)

from lib.config_utils import load_config, resolve_project_root
from lib.dinov3_frozen_tail import build_dinov3_tail
from lib.distortion_metrics import eval_val_distortion
from lib.orfc_codec import NORM_MODE_CHOICES, resolve_n_prefix

_PQFC_OFFLINE = _EXAMPLE_DIR / "offline"
if str(_PQFC_OFFLINE) not in sys.path:
    sys.path.insert(0, str(_PQFC_OFFLINE))
from utils import preload_features, set_seed  # noqa: E402


def _apply_transform_tau_binding(args):
    """Bind soft/hard to orthogonal transform switch."""
    if args.use_transform:
        if args.tau_start <= 0:
            args.tau_start = 0.5
            args.tau_end = 0.005
            print("  [bind] use_transform: soft PQ tau_start=0.5, tau_end=0.005")
    else:
        if args.tau_start != 0.0:
            print(f"  [bind] --no_transform: forcing tau_start=0 (was {args.tau_start})")
        args.tau_start = 0.0
        args.warm_start_opq = False


def _load_train_val_features(feat_dir: Path, max_train: int, n_val: int, seed: int):
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {feat_dir}")

    rng = np.random.RandomState(seed)
    n_total = len(files)
    n_need = min(n_total, max_train + n_val) if max_train > 0 else n_total
    if n_need < n_total:
        pick = rng.choice(n_total, n_need, replace=False)
        files = [files[i] for i in sorted(pick)]

    features, _ = preload_features(files, num_workers=8)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(features))
    n_val = min(n_val, len(features))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if max_train > 0:
        train_idx = train_idx[:max_train]
    train_feats = [features[i] for i in train_idx]
    val_feats = [features[i] for i in val_idx]
    return train_feats, val_feats


def _ckpt_name(args, bt_dim: int, D: int) -> str:
    if args.use_transform:
        bt_tag = f"bt{bt_dim}"
        ws_tag = "ws" if args.warm_start_opq else "km"
    else:
        bt_tag = "noR"
        ws_tag = "km"
    rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    norm_tag = f"_{args.norm_mode}" if args.norm_mode != "per_image" else ""
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    patch_tag = "_ptpatch" if args.train_tokens == "patch" else ""
    return (
        f"{args.layer}_K{args.K}_emb{args.embedding_dim}_{bt_tag}_{ws_tag}"
        f"{rate_tag}{norm_tag}{tau_tag}"
        f"_lr{args.lr}_ep{args.epochs}_n{args.max_train}_s{args.seed}{patch_tag}.pt"
    )


def _norm_vectors_from_batch(Y: torch.Tensor, train_tokens: str, n_prefix: int, D: int) -> np.ndarray:
    if train_tokens == "patch" and n_prefix > 0:
        return Y[:, n_prefix:, :].reshape(-1, D).cpu().numpy()
    return Y.reshape(-1, D).cpu().numpy()


def _cuda_release():
    """Aggressively free CUDA caching allocator after large init phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_and_save(args) -> str:
    # Must be set before first CUDA allocation in this process.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _apply_transform_tau_binding(args)

    cfg = load_config(Path(args.config) if args.config else None)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])
    n_prefix = resolve_n_prefix(args.norm_mode, args.n_prefix, cfg.get("n_prefix", 5))
    token_hw = tuple(cfg.get("token_hw")
                     or cfg.get("imagenet_token_hw")
                     or cfg.get("coco_token_hw", [14, 14]))
    D = cfg["embed_dim"]
    G = D // args.embedding_dim

    feat_dir = Path(args.feat_dir or cfg["paths"]["train_feat_dir"])
    weights_dir = Path(args.weights_dir or cfg["paths"]["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "OrthogonalTransform + soft PQ" if args.use_transform else "no R + hard PQ"
    print(f"\n{'#' * 70}")
    print(f"# DINOv3 PQFC Training ({mode_str})")
    print(f"# layer={args.layer}  K={args.K}  d={args.embedding_dim}")
    print(f"# norm={args.norm_mode}  n_prefix={n_prefix}  λ={args.lmbda}")
    print(f"# train_tokens={args.train_tokens}  use_transform={args.use_transform}")
    print(f"# loss = R + λ·D  (D = ΔL_ref)")
    print(f"# feat_dir={feat_dir}")
    print(f"# token_hw={token_hw}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    train_feats, val_feats = _load_train_val_features(
        feat_dir, args.max_train, args.n_val, args.seed,
    )
    train_array = np.stack(train_feats)
    n_train = train_array.shape[0]
    T = train_array.shape[1]
    print(f"  train={n_train}  val={len(val_feats)}  shape={train_array.shape[1:]}")
    if T > 1000:
        print(
            f"  WARNING: long token sequence T={T} (COCO-scale). "
            f"Prefer ImageNet features (T≈201) for SoftPQ on 24GB GPUs."
        )

    # ---- Backbone + frozen tail (always ΔL_ref) ----
    print(f"\nLoading DINOv3 backbone for frozen tail (layer_idx={layer_idx})...")
    bb_path = Path(cfg["paths"]["backbone"])
    if not bb_path.is_file():
        alt = bb_path.with_suffix(".safetensors" if bb_path.suffix == ".pth" else ".pth")
        if alt.is_file():
            bb_path = alt
            print(f"  backbone fallback → {bb_path}")
    backbone = Dinov3TimmBackbone(
        model_size="large",
        img_size=512,
        patch_size=cfg["patch_size"],
        dynamic_size=True,
        slot=cfg["slot"],
        n_last_blocks=1,
        cast_dtype="float32",
        pretrained=False,
        ckpt_path=str(bb_path),
        device=str(device),
    ).eval()
    tail = build_dinov3_tail(backbone, layer_idx, token_hw, device)
    n_tail = len(getattr(tail, "blocks", []))
    precompute_teacher = n_tail > 0
    print(f"  tail blocks={n_tail} (+ norm)")
    if not precompute_teacher:
        print("  teacher: on-the-fly norm (skip 22GB CPU precompute)")
    del backbone
    _cuda_release()

    opq_usage = None
    R_opq = cb_opq = None
    if args.use_transform:
        opq_scope = "patch" if args.train_tokens == "patch" else "all"
        print(f"\n{'=' * 60}\n  [OPQ warm-start]  tokens={opq_scope}\n{'=' * 60}")
        t0 = time.time()
        all_vectors = []
        for start in range(0, n_train, 200):
            end = min(start + 200, n_train)
            X = torch.from_numpy(train_array[start:end]).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode, n_prefix=n_prefix)
            all_vectors.append(
                _norm_vectors_from_batch(Y, args.train_tokens, n_prefix, D)
            )
            del X, Y
        full_vecs = np.concatenate(all_vectors, axis=0)
        del all_vectors

        max_flat = args.kmeans_max_samples // G
        if full_vecs.shape[0] > max_flat:
            rng = np.random.RandomState(args.seed)
            full_vecs = full_vecs[rng.choice(full_vecs.shape[0], max_flat, replace=False)]

        R_opq, cb_opq, _ = learn_orfc_rotation(
            full_vecs, G, args.embedding_dim, args.K,
            max_iter_orfc=20, max_iter_kmeans=100, device=device, verbose=False,
        )
        del full_vecs
        _cuda_release()
        print(f"  OPQ done ({time.time() - t0:.1f}s)")

        if args.lmbda > 0 and args.warm_start_opq:
            R_t = torch.from_numpy(R_opq).float().to(device)
            cb_t = torch.from_numpy(np.stack(cb_opq)).float().to(device)
            opq_usage = np.zeros((G, args.K), dtype=np.float64)
            for start in range(0, n_train, 200):
                end = min(start + 200, n_train)
                X = torch.from_numpy(train_array[start:end]).float().to(device)
                with torch.no_grad():
                    Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode, n_prefix=n_prefix)
                    if args.train_tokens == "patch" and n_prefix > 0:
                        flat = Y[:, n_prefix:, :].reshape(-1, D) @ R_t
                    else:
                        flat = Y.reshape(-1, D) @ R_t
                    sub = flat.reshape(-1, G, args.embedding_dim).permute(1, 0, 2).contiguous()
                    lbl = batched_assign(sub, cb_t, device=device)[1].cpu().numpy()
                    for g in range(G):
                        np.add.at(opq_usage[g], lbl[g], 1)
                del X, Y
            del R_t, cb_t
            _cuda_release()
    else:
        print(f"\n{'=' * 60}\n  [init] no orthogonal R; k-means in train_soft_pq\n{'=' * 60}")

    del train_feats
    _cuda_release()

    # ---- PQFC training ----
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

    tau_str = f"τ={args.tau_start}→{args.tau_end}" if args.tau_start > 0 else "hard PQ"
    print(f"\n{'=' * 60}\n  [PQFC] ΔL_ref  {tau_str}  epochs={args.epochs}\n{'=' * 60}")

    t_spq = time.time()
    codec, history = train_soft_pq(
        features_train=train_array,
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
        use_mse_loss=False,
        lmbda=args.lmbda,
        prior_init_counts=opq_usage,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
        kmeans_max_samples=args.kmeans_max_samples,
        grad_clip=args.grad_clip,
        prior_floor=args.prior_floor,
        precompute_teacher=precompute_teacher,
        train_tokens=args.train_tokens,
    )
    print(f"  Training done ({time.time() - t_spq:.1f}s)")

    ckpt_path = weights_dir / _ckpt_name(args, D, D)
    if getattr(args, "keep_pt", False):
        save_codec(
            codec, str(ckpt_path),
            train_tokens=args.train_tokens,
            use_transform=args.use_transform,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
        )
        print(f"  Resume PT → {ckpt_path}")

    hist_path = ckpt_path.with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    prefix_bypass = args.train_tokens == "patch" and n_prefix > 0
    val_metrics = eval_val_distortion(
        val_feats, codec, tail,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
        prefix_bypass=prefix_bypass,
        device=device,
        batch_size=args.batch_size,
    )
    val_metrics_path = ckpt_path.with_suffix(".val_metrics.json")
    with open(val_metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)

    if tail is not None:
        tail.to("cpu")
    del tail
    _cuda_release()

    token_slice = "patch" if args.train_tokens == "patch" else "all"
    pmf = compute_histogram_pmf(
        codec,
        train_array,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
        device=device,
        batch_size=args.batch_size,
        token_slice=token_slice,
    )
    npz_path = save_codec_npz(
        codec,
        ckpt_path.with_suffix(".npz"),
        pmf,
        source_pt=str(ckpt_path) if getattr(args, "keep_pt", False) else None,
        norm_mode=args.norm_mode,
        n_prefix=n_prefix,
    )
    print(f"  Official NPZ → {npz_path}")
    print(f"  meta: norm_mode={args.norm_mode} n_prefix={n_prefix} "
          f"train_tokens={args.train_tokens}")
    _cuda_release()

    print(f"  Val raw MSE = {val_metrics['raw_mse']:.6f}")
    if val_metrics.get("post_ln_patch_mse") is not None:
        print(f"  Val post-LN patch MSE (delta_L_ref) = {val_metrics['post_ln_patch_mse']:.6f}")
    print(f"  Val metrics → {val_metrics_path}")
    print(f"  Val MSE = {val_metrics['raw_mse']:.6f}")
    return str(npz_path)


def main():
    defaults = load_config().get("train_defaults", {})
    p = argparse.ArgumentParser(description="DINOv3 blk23 PQFC offline training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--layer", type=str, default="blk23")
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--embedding_dim", type=int, default=defaults.get("embedding_dim", 32))
    p.add_argument("--norm_mode", choices=NORM_MODE_CHOICES,
                   default=defaults.get("norm_mode", "per_image"))
    p.add_argument("--n_prefix", type=int, default=0)
    p.add_argument("--epochs", type=int, default=defaults.get("epochs", 100))
    p.add_argument("--lr", type=float, default=defaults.get("lr", 3e-4))
    p.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 64))
    p.add_argument("--lmbda", type=float, default=defaults.get("lmbda", 0.5),
                   help="J = R + λ·D (D = ΔL_ref)")
    p.add_argument("--tau_start", type=float, default=defaults.get("tau_start", 0.5),
                   help="Overridden by transform binding")
    p.add_argument("--tau_end", type=float, default=defaults.get("tau_end", 0.005))
    p.add_argument("--tau_schedule", default="exponential")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--prior_floor", type=float, default=0.0)
    p.add_argument("--warm_start_opq", action="store_true", default=True)
    p.add_argument("--no_warm_start", dest="warm_start_opq", action="store_false")
    p.add_argument("--max_train", type=int, default=defaults.get("max_train", 5000))
    p.add_argument("--n_val", type=int, default=defaults.get("n_val", 50))
    p.add_argument("--kmeans_max_samples", type=int, default=defaults.get("kmeans_max_samples", 2_000_000))
    p.add_argument("--train_tokens", choices=("all", "patch"), default="all")
    p.add_argument("--use_transform", action="store_true", default=True,
                   help="Orthogonal R + soft PQ (default)")
    p.add_argument("--no_transform", dest="use_transform", action="store_false",
                   help="No R; hard PQ (tau_start forced to 0)")
    p.add_argument("--feat_dir", type=str, default=None)
    p.add_argument("--weights_dir", type=str, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    p.add_argument(
        "--keep_pt", action="store_true",
        help="Also save intermediate .pt for resume (eval uses .npz only)",
    )
    args = p.parse_args()
    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    train_and_save(args)


if __name__ == "__main__":
    main()

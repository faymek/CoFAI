#!/usr/bin/env python
"""
Soft-PQ Training — learn OrthogonalTransform + differentiable PQ codebooks.

Trains codec via frozen-tail consistency loss (ΔL_ref) with optional rate loss.
Requires pre-extracted features AND a backbone model (for building the frozen tail).

Usage:
    python train_soft_pq.py --backbone dinov2_vitl14 --layer blk10 \
        --K 64 --embedding_dim 32 --epochs 100 --warm_start_opq

    python train_soft_pq.py --backbone dinov2_vitg14 --layer blk19 \
        --K 8 --embedding_dim 32 --lmbda 0.5 --tau_start 0.5 --lr 0.0003
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
)

OFFLINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OFFLINE_DIR)
sys.path.insert(0, PROJECT_ROOT)

from cofai.entropy_models.soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureTransform, FeatureCodec,
    FrozenTail, CLIPFrozenTail,
    train_soft_pq, save_codec,
)
from cofai.entropy_models.soft_pq_export import (
    compute_histogram_pmf, save_codec_npz, npz_path_for_codec,
)
from cofai.entropy_models.orfc_model import (
    batch_normalize_gpu, batch_inv_normalize_gpu,
    learn_orfc_rotation, batched_assign,
)
from utils import set_seed, preload_features
from weights_paths import codec_weights_dir
from backbone.wrapper import Dinov2Wrapper, ClipWrapper

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

try:
    import logging
    from mmcv.utils import get_logger
    logger = get_logger('mmcv')
    logger.setLevel(logging.WARNING)
except ImportError:
    pass


def train_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    freeze_parts = []
    if args.freeze_transform:
        freeze_parts.append("freeze_R")
    if args.freeze_codebooks:
        freeze_parts.append("freeze_C")
    freeze_str = ", ".join(freeze_parts) if freeze_parts else "all trainable"
    print(f"# Soft-PQ Training (OrthogonalTransform + PQ)")
    print(f"# backbone={args.backbone}, layer={args.layer} (idx={layer_idx})")
    print(f"# K={args.K}, emb={args.embedding_dim}, bt={args.bottleneck_dim}")
    print(f"# transform: {freeze_str}")
    tau_info = f", τ={args.tau_start}→{args.tau_end}" if args.tau_start > 0 else ""
    print(f"# epochs={args.epochs}, lr={args.lr}, λ={args.lmbda}{tau_info}")
    print(f"# max_train={args.max_train_images}, batch={args.batch_size}")
    print(f"# device={device}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load features ----
    train_dir = Path(args.feat_root) / "train" / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))

    print(f"\nTrain dir: {train_dir}")
    print(f"  Found {len(train_files)} feature files")

    if len(train_files) == 0:
        raise RuntimeError(f"No .npy files in {train_dir}")

    features_train, _ = preload_features(train_files, num_workers=8)

    D = features_train[0].shape[1]
    T = features_train[0].shape[0]
    bt_dim = args.bottleneck_dim if args.bottleneck_dim > 0 else D
    Dp = bt_dim
    num_groups = Dp // args.embedding_dim

    print(f"  D={D}, D'={Dp}, T={T}, G={num_groups}, d={args.embedding_dim}")

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train_sub = [features_train[i] for i in idx]
        print(f"  Subsampled to {len(features_train_sub)} images")
    else:
        features_train_sub = features_train

    # Validation set
    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]

    # ---- Load backbone (for frozen tail) ----
    is_clip = args.backbone.startswith("clip")
    if is_clip:
        print(f"\n  Loading CLIP ViT-L/14...")
        wrapper = ClipWrapper(args.classnames, device=device)
    else:
        print(f"\n  Loading DINOv2 ({args.backbone})...")
        wrapper = Dinov2Wrapper(
            head_layers=1, model_name=args.backbone,
            weights_root=args.weights_root, device=device,
        )

    n_blocks = len(wrapper.backbone.blocks)
    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    norm_ref = wrapper.backbone.norm
    n_tail = len(tail_blocks)
    print(f"  Tail: {n_tail} blocks (blk{layer_idx+1}..blk{layer_idx+n_tail}) + norm")

    # ---- OPQ Baseline (for warm-start) ----
    print(f"\n{'=' * 60}")
    print(f"  [OPQ baseline] for warm-start initialization")
    print(f"{'=' * 60}")

    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    t0 = time.time()
    all_vectors = []
    for start in range(0, len(features_train_sub), 200):
        end = min(start + 200, len(features_train_sub))
        X = torch.from_numpy(
            np.stack(features_train_sub[start:end])
        ).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode)
        all_vectors.append(Y.reshape(-1, D).cpu().numpy())
        del X, Y
    full_vectors = np.concatenate(all_vectors, axis=0)
    del all_vectors

    opq_groups = D // args.embedding_dim
    max_flat = args.kmeans_max_samples // opq_groups
    if full_vectors.shape[0] > max_flat:
        rng2 = np.random.RandomState(args.seed)
        idx2 = rng2.choice(full_vectors.shape[0], max_flat, replace=False)
        full_vectors = full_vectors[idx2]

    R_std, codebooks_std, hist_std = learn_orfc_rotation(
        full_vectors, opq_groups, args.embedding_dim, args.K,
        max_iter_orfc=20, max_iter_kmeans=100,
        device=device, verbose=False,
    )
    std_time = time.time() - t0
    print(f"    OPQ done: MSE={hist_std[-1][0]:.8f} ({std_time:.1f}s)")

    # OPQ usage counts for prior init
    opq_usage_counts = None
    if args.lmbda > 0 and args.warm_start_opq:
        R_t = torch.from_numpy(R_std).float().to(device)
        cb_t = torch.from_numpy(np.stack(codebooks_std)).float().to(device)
        opq_usage_counts = np.zeros((opq_groups, args.K), dtype=np.float64)
        for start in range(0, len(features_train_sub), 200):
            end = min(start + 200, len(features_train_sub))
            X = torch.from_numpy(
                np.stack(features_train_sub[start:end])
            ).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode)
                flat = Y.reshape(-1, D) @ R_t
                sub = flat.reshape(-1, opq_groups, args.embedding_dim) \
                      .permute(1, 0, 2).contiguous()
                dists = torch.cdist(sub, cb_t)
                labels = dists.argmin(dim=-1)
                for g in range(opq_groups):
                    for k in labels[g].cpu().numpy():
                        opq_usage_counts[g, k] += 1
            del X, Y, flat, sub, dists, labels
        del R_t, cb_t
        torch.cuda.empty_cache()

    del full_vectors
    torch.cuda.empty_cache()

    # ---- Build frozen tail ----
    wrapper.backbone.to(device)
    if wrapper.head is not None:
        wrapper.head.to(device)

    tail_blocks_ref = list(wrapper.backbone.blocks[layer_idx + 1:])
    norm_ref = wrapper.backbone.norm
    if is_clip:
        tail = CLIPFrozenTail(tail_blocks_ref, norm_ref, device=device)
    else:
        tail = FrozenTail(tail_blocks_ref, norm_ref, device=device)

    # Move early blocks off GPU
    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    # ---- Build transform ----
    transform = None
    if bt_dim > 0 and bt_dim == D:
        transform = OrthogonalTransform(D)
    elif bt_dim > 0:
        transform = FeatureTransform(D, bt_dim)

    # ---- Warm-start setup ----
    R_ws = None
    C_ws = None
    if args.warm_start_opq and bt_dim == D:
        R_ws = R_std.copy()
        C_ws = [c.copy() for c in codebooks_std]
        if np.linalg.det(R_ws) < 0:
            R_ws[:, -1] *= -1
            C_ws[-1][:, -1] *= -1
            print(f"    det(R_opq)<0: flipped last col to SO(D)")

    # ---- Train codec ----
    print(f"\n{'=' * 60}")
    tau_str = f", τ={args.tau_start}→{args.tau_end}" if args.tau_start > 0 else ""
    ws_str = "warm-start" if args.warm_start_opq else "k-means"
    print(f"  [Codec Training] K={args.K}, λ={args.lmbda}, {ws_str}{tau_str}")
    print(f"{'=' * 60}")

    t0 = time.time()
    codec, history = train_soft_pq(
        features_train=features_train_sub,
        tail=tail,
        G=num_groups,
        K=args.K,
        d=args.embedding_dim,
        norm_mode=args.norm_mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        val_features=val_features,
        verbose=True,
        transform=transform,
        R_init=R_ws,
        codebooks_init=C_ws,
        kmeans_max_samples=args.kmeans_max_samples,
        use_mse_loss=args.mse_loss,
        lmbda=args.lmbda,
        prior_init_counts=opq_usage_counts,
        grad_clip=args.grad_clip,
        freeze_transform=args.freeze_transform,
        freeze_codebooks=args.freeze_codebooks,
        prior_floor=args.prior_floor,
        tau_start=args.tau_start,
        tau_end=args.tau_end,
        tau_schedule=args.tau_schedule,
    )
    train_time = time.time() - t0
    print(f"\n  Codec training done ({train_time:.1f}s)")

    # ---- Save checkpoint ----
    out_dir = codec_weights_dir(args.weights_dir, args.backbone)
    os.makedirs(out_dir, exist_ok=True)

    bt_tag = f"bt{bt_dim}" if bt_dim > 0 else "noBt"
    ws_tag = "ws" if args.warm_start_opq else "km"
    mse_tag = "_mse" if args.mse_loss else ""
    rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    fz_tag = ""
    if args.freeze_transform:
        fz_tag += "_fzR"
    if args.freeze_codebooks:
        fz_tag += "_fzC"
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    ckpt_name = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
                 f"_{bt_tag}_{ws_tag}{mse_tag}{rate_tag}"
                 f"{fz_tag}{tau_tag}"
                 f"_lr{args.lr}_ep{args.epochs}"
                 f"_n{args.max_train_images}_s{args.seed}")
    ckpt_path = os.path.join(out_dir, f"{ckpt_name}.pt")
    save_codec(codec, ckpt_path)

    print(f"\n  Computing PMF from training features...")
    pmf = compute_histogram_pmf(
        codec, features_train_sub, norm_mode=args.norm_mode, device=device,
    )
    npz_path = save_codec_npz(codec, npz_path_for_codec(ckpt_path), pmf,
                              source_pt=ckpt_path)
    print(f"  Sidecar NPZ saved: {npz_path}")
    print(f"    PMF entropy: {-(pmf * np.log2(pmf + 1e-30)).sum(axis=1).mean():.4f} bits/group")

    print(f"\n  Checkpoint saved: {ckpt_path}")
    print(f"    file size: {os.path.getsize(ckpt_path) / 1024:.1f} KB")

    return ckpt_path


def main():
    parser = argparse.ArgumentParser(
        description="Soft-PQ training — save codec checkpoint (.pt)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14", "clip_vitl14"])
    parser.add_argument("--layer", type=str, required=True,
                        help="Layer name (e.g. blk05, blk10)")
    parser.add_argument("--K", type=int, required=True,
                        help="Codebook size")
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="Per-group embedding dimension")
    parser.add_argument("--bottleneck_dim", type=int, default=0,
                        help="Transform bottleneck dim (0=auto=D, use OrthogonalTransform)")
    parser.add_argument("--norm_mode", type=str, default="per_image")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lmbda", type=float, default=0.5,
                        help="R-D Lagrange multiplier (0=no rate loss)")
    parser.add_argument("--tau_start", type=float, default=0.5,
                        help="Initial temperature for soft PQ")
    parser.add_argument("--tau_end", type=float, default=0.005,
                        help="Final temperature (annealed)")
    parser.add_argument("--tau_schedule", type=str, default="exponential",
                        choices=["exponential", "linear"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--prior_floor", type=float, default=0.0)

    parser.add_argument("--warm_start_opq", action="store_true", default=True,
                        help="Warm-start from OPQ solution")
    parser.add_argument("--no_warm_start", dest="warm_start_opq",
                        action="store_false")
    parser.add_argument("--freeze_transform", action="store_true")
    parser.add_argument("--freeze_codebooks", action="store_true")
    parser.add_argument("--mse_loss", action="store_true",
                        help="Use MSE loss instead of ΔL_ref")

    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "orfc"))
    parser.add_argument("--weights_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "orfc_2446"))
    parser.add_argument("--weights_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "pretrained"),
                        help="Pretrained backbone weights root")
    parser.add_argument("--classnames", type=str,
                        default=os.path.join(OFFLINE_DIR, "cfg", "classnames.txt"))

    args = parser.parse_args()
    train_and_save(args)


if __name__ == '__main__':
    main()

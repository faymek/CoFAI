#!/usr/bin/env python3
"""
Soft-PQ Training — learn differentiable PQ codec, save to .pt.

Only requires pre-extracted features (no backbone needed at training time).
Trained weights are shared between cls and seg tasks.

Usage:
    python train_soft_pq.py --backbone dinov2_vitl14 --layer blk20 \
        --K 16 --embedding_dim 32 --lmbda 0.5 --epochs 100

    python train_soft_pq.py --backbone dinov2_vitg14 --layer blk29 \
        --K 64 --embedding_dim 32 --lmbda 0.5 --epochs 100
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from cofai.entropy_models.soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureCodec,
    train_soft_pq, save_codec, load_codec,
)
from cofai.entropy_models.orfc_model import batch_normalize_gpu


def preload_features(feat_dir, max_images=5000, seed=42):
    """Load .npy feature files from directory."""
    feat_files = sorted(Path(feat_dir).glob("*.npy"))
    if not feat_files:
        raise RuntimeError(f"No .npy files in {feat_dir}")

    if max_images > 0 and len(feat_files) > max_images:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(feat_files), max_images, replace=False)
        feat_files = [feat_files[i] for i in idx]

    features = []
    for f in feat_files:
        features.append(np.load(f))
    return features


def main():
    parser = argparse.ArgumentParser(description="Soft-PQ Training")
    parser.add_argument("--backbone", required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--layer", required=True)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--lmbda", type=float, default=0.5,
                        help="Rate-distortion tradeoff (0 = distortion only)")
    parser.add_argument("--tau", type=float, default=0.5,
                        help="Straight-through softmax temperature")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feat_root", type=str,
                        default=str(PROJECT_ROOT / "features" / "orfc"))
    parser.add_argument("--weights_dir", type=str,
                        default=str(PROJECT_ROOT / "weights" / "orfc_2446"))
    parser.add_argument("--init_from_orfc", type=str, default="",
                        help="Path to ORFC .npz for warm-start (rotation + codebooks)")
    parser.add_argument("--warm_start", action="store_true", default=True,
                        help="Warm-start from k-means (default)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'#' * 70}")
    print(f"# Soft-PQ Training")
    print(f"# backbone={args.backbone}, layer={args.layer}")
    print(f"# K={args.K}, emb={args.embedding_dim}, lmbda={args.lmbda}")
    print(f"# tau={args.tau}, lr={args.lr}, epochs={args.epochs}")
    print(f"# device={device}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # Load features
    feat_dir = Path(args.feat_root) / "train" / args.backbone / args.layer
    print(f"\nLoading features from: {feat_dir}")
    features = preload_features(feat_dir, args.max_images, args.seed)
    print(f"  Loaded {len(features)} images")

    D = features[0].shape[-1]
    G = D // args.embedding_dim
    print(f"  D={D}, G={G}, d={args.embedding_dim}")

    # Flatten and normalize
    all_vectors = []
    for start in range(0, len(features), 200):
        end = min(start + 200, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode="per_image")
        all_vectors.append(Y.reshape(-1, D).cpu())
    Z_flat = torch.cat(all_vectors, dim=0)
    print(f"  Total training vectors: {Z_flat.shape[0]}")

    # Build codec
    pq = SoftPQ(G, args.K, args.embedding_dim, lmbda=args.lmbda)
    rot = OrthogonalTransform(D)

    # Warm start
    if args.init_from_orfc and os.path.exists(args.init_from_orfc):
        print(f"\n  Warm-starting from ORFC: {args.init_from_orfc}")
        data = np.load(args.init_from_orfc, allow_pickle=True)
        rot.init_from_opq(data["R"])
        pq.init_codebooks([data["codebooks"][g] for g in range(G)])
    elif args.warm_start:
        print(f"\n  Warm-starting codebooks from k-means...")
        pq.init_from_kmeans(Z_flat, device=str(device))
        print(f"  Warm-starting rotation from PCA...")
        from cofai.entropy_models.orfc_model import learn_pca_rotation
        R_np = learn_pca_rotation(Z_flat.numpy(), args.embedding_dim)
        rot.init_from_opq(R_np)

    codec = FeatureCodec(pq, rot).to(device)

    # Train
    print(f"\n  Starting training...")
    t0 = time.time()
    codec = train_soft_pq(
        codec, Z_flat, device=str(device),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.tau,
    )
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    # Save
    bb_dir = Path(args.weights_dir) / args.backbone
    bb_dir.mkdir(parents=True, exist_ok=True)

    init_tag = "ws" if args.warm_start else "km"
    bt_dim = D
    tag = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
           f"_bt{bt_dim}_{init_tag}"
           f"_lmbda{args.lmbda}_tau{args.tau}"
           f"_lr{args.lr}_ep{args.epochs}"
           f"_n{args.max_images}_s{args.seed}")
    out_path = bb_dir / f"{tag}.pt"

    save_codec(codec, str(out_path))
    print(f"\n  Saved: {out_path}")
    print(f"  Size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

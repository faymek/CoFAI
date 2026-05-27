#!/usr/bin/env python
"""
ORFC Training — learn rotation matrix + PQ codebooks, save to .npz.

Only requires pre-extracted features (no backbone model needed).
Trained weights are shared between classification and segmentation tasks.

Usage:
    python train_orfc.py --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32
    python train_orfc.py --backbone clip_vitl14 --layer blk05 --K 256 --embedding_dim 16
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
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "cofai", "entropy_models"))
sys.path.insert(0, PROJECT_ROOT)

from orfc_model import batch_normalize_gpu, batched_assign, learn_orfc_rotation
from utils import set_seed, preload_features

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def train_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print(f"\n{'#' * 70}")
    print(f"# ORFC Training")
    print(f"# backbone={args.backbone}, layer={args.layer}")
    print(f"# K={args.K}, embedding_dim={args.embedding_dim}")
    print(f"# device={device}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load training features ----
    train_dir = Path(args.feat_root) / "train" / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    print(f"\nTrain dir: {train_dir}")
    print(f"  Found {len(train_files)} feature files")

    if len(train_files) == 0:
        raise RuntimeError(f"No .npy files in {train_dir}")

    features_train, _ = preload_features(train_files, num_workers=8)

    D = features_train[0].shape[1]
    num_groups = D // args.embedding_dim
    print(f"  D={D}, num_groups={num_groups}, embedding_dim={args.embedding_dim}")

    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train = [features_train[i] for i in idx]
        print(f"  Subsampled to {len(features_train)} images")

    # ---- Flatten + normalize ----
    print(f"\nPreparing training vectors...")
    all_vectors = []
    for start in range(0, len(features_train), 200):
        end = min(start + 200, len(features_train))
        X = torch.from_numpy(
            np.stack(features_train[start:end])
        ).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode='per_image')
        all_vectors.append(Y.reshape(-1, D).cpu().numpy())
        del X, Y
    full_vectors = np.concatenate(all_vectors, axis=0)
    del all_vectors, features_train

    max_flat = args.kmeans_max_samples // num_groups
    if full_vectors.shape[0] > max_flat:
        rng2 = np.random.RandomState(args.seed)
        idx2 = rng2.choice(full_vectors.shape[0], max_flat, replace=False)
        full_vectors = full_vectors[idx2]

    print(f"  Training vectors: {full_vectors.shape[0]:,} x {D}")

    # ---- ORFC Training ----
    print(f"\n{'=' * 60}")
    print(f"  ORFC alternating optimization")
    print(f"{'=' * 60}")

    t0 = time.time()
    R, codebooks, history = learn_orfc_rotation(
        full_vectors, num_groups, args.embedding_dim, args.K,
        max_iter_orfc=20, max_iter_kmeans=100,
        device=device, verbose=True,
    )
    train_time = time.time() - t0
    del full_vectors
    torch.cuda.empty_cache()

    print(f"\n  ORFC training done: MSE={history[-1][0]:.8f} ({train_time:.1f}s)")

    # ---- Compute PMF for rANS encoding ----
    print(f"\n  Computing PMF from training features...")
    codebooks_np = np.stack(codebooks)  # [num_groups, K, embedding_dim]

    train_dir_pmf = Path(args.feat_root) / "train" / args.backbone / args.layer
    train_files_pmf = sorted(train_dir_pmf.glob("*.npy"))
    features_pmf, _ = preload_features(train_files_pmf, num_workers=8)

    if args.max_train_images > 0 and len(features_pmf) > args.max_train_images:
        rng_pmf = np.random.RandomState(args.seed)
        idx_pmf = rng_pmf.choice(len(features_pmf), args.max_train_images, replace=False)
        features_pmf = [features_pmf[i] for i in idx_pmf]

    label_counts = np.zeros((num_groups, args.K), dtype=np.int64)
    R_t = torch.from_numpy(R).float().to(device)
    cb_t = torch.from_numpy(codebooks_np).float().to(device)

    for start in range(0, len(features_pmf), 200):
        end = min(start + 200, len(features_pmf))
        X = torch.from_numpy(np.stack(features_pmf[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode='per_image')
            flat = Y.reshape(-1, D)
            Z = flat @ R_t
            z_3d = Z.reshape(-1, num_groups, args.embedding_dim).permute(1, 0, 2).contiguous()
            _, labels = batched_assign(z_3d, cb_t, device=device)
            labels_np = labels.cpu().numpy()
            for g in range(num_groups):
                for lbl in labels_np[g]:
                    label_counts[g, lbl] += 1
        del X, Y

    pmf = label_counts.astype(np.float32)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    pmf = np.maximum(pmf, 1e-7)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)

    print(f"    PMF shape: {pmf.shape}")
    print(f"    PMF entropy: {-(pmf * np.log2(pmf)).sum(axis=1).mean():.4f} bits/group")

    del features_pmf, R_t, cb_t
    torch.cuda.empty_cache()

    # ---- Save weights ----
    out_dir = os.path.join(args.weights_dir, args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.layer}_K{args.K}_e{args.embedding_dim}"
    out_path = os.path.join(out_dir, f"{tag}.npz")

    np.savez(out_path, R=R, codebooks=codebooks_np, pmf=pmf)

    print(f"\n  Weights saved: {out_path}")
    print(f"    R shape: {R.shape}")
    print(f"    codebooks shape: {codebooks_np.shape}")
    print(f"    pmf shape: {pmf.shape}")
    print(f"    file size: {os.path.getsize(out_path) / 1024:.1f} KB")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="ORFC training — save rotation + codebooks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14", "clip_vitl14"])
    parser.add_argument("--layer", type=str, required=True,
                        help="Layer name (e.g. blk05, blk10)")
    parser.add_argument("--K", type=int, required=True,
                        help="Codebook size")
    parser.add_argument("--embedding_dim", type=int, required=True,
                        help="Per-group embedding dimension")

    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "orfc"))
    parser.add_argument("--weights_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "orfc"))

    args = parser.parse_args()
    train_and_save(args)


if __name__ == '__main__':
    main()

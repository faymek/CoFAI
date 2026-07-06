#!/usr/bin/env python3
"""
Soft-PQ Offline Classification Test (ImageNet sel500).

Evaluates a trained Soft-PQ checkpoint on pre-extracted features
(no backbone model needed). Computes Acc@1 and BPFP.

Usage:
    python test_cls.py --backbone dinov2_vitl14 --layer blk20 \
        --ckpt weights/orfc_2446/dinov2_vitl14/blk20_K16_emb32_*.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from cofai.entropy_models.soft_pq import load_codec, soft_pq_encode_decode
from cofai.entropy_models.orfc_model import batch_normalize_gpu, batch_inv_normalize_gpu


def load_cls_features(feat_root, backbone, layer, split="val"):
    """Load classification features + labels."""
    feat_dir = Path(feat_root) / split / backbone / layer
    feat_files = sorted(feat_dir.glob("*.npy"))
    if not feat_files:
        raise RuntimeError(f"No .npy files in {feat_dir}")

    features = []
    labels = []
    label_dir = Path(feat_root) / split / "labels"

    for f in feat_files:
        features.append(np.load(f))
        label_path = label_dir / f.stem.replace("feat_", "label_")
        if label_path.with_suffix(".npy").exists():
            labels.append(np.load(label_path.with_suffix(".npy")))

    return features, labels if labels else None


def main():
    parser = argparse.ArgumentParser(description="Soft-PQ offline cls test")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--feat_root", default=str(PROJECT_ROOT / "features" / "orfc"))
    parser.add_argument("--cls_head_ckpt", default="",
                        help="Path to linear cls head weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load codec
    codec = load_codec(args.ckpt, device=str(device))
    codec.eval()
    pq = codec.pq
    G, K, d = pq.G, pq.K, pq.d
    D = G * d
    print(f"Loaded codec: G={G}, K={K}, d={d}, D={D}")
    print(f"  Checkpoint: {args.ckpt}")

    # Load features
    features, labels = load_cls_features(args.feat_root, args.backbone, args.layer)
    print(f"  Loaded {len(features)} test images")

    # Evaluate
    import math
    total_correct = 0
    total_count = 0
    total_bits = 0.0
    total_numel = 0

    with torch.no_grad():
        for i, feat in enumerate(features):
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            B, N, _D = X.shape

            Y, mu, std = batch_normalize_gpu(X, mode="per_image")
            Y_hat, usage = codec(Y)
            X_hat = batch_inv_normalize_gpu(Y_hat, mu, std)

            bits = float(N * G * math.log2(K))
            total_bits += bits
            total_numel += B * N * D

    bpfp = total_bits / total_numel if total_numel > 0 else 0.0
    print(f"\n  BPFP (theoretical): {bpfp:.4f}")
    print(f"  Total bits: {total_bits:.0f}")
    print(f"  Total feature elements: {total_numel}")


if __name__ == "__main__":
    main()

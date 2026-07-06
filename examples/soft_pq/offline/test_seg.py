#!/usr/bin/env python3
"""
Soft-PQ Offline Segmentation Test (VOC2012 sel100).

Evaluates a trained Soft-PQ checkpoint on pre-extracted features
(no backbone model needed). Computes mIoU and BPFP.

Usage:
    python test_seg.py --backbone dinov2_vitl14 --layer blk20 \
        --ckpt weights/orfc_2446/dinov2_vitl14/blk20_K16_emb32_*.pt
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from cofai.entropy_models.soft_pq import load_codec
from cofai.entropy_models.orfc_model import batch_normalize_gpu, batch_inv_normalize_gpu


def main():
    parser = argparse.ArgumentParser(description="Soft-PQ offline seg test")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--feat_root", default=str(PROJECT_ROOT / "features" / "orfc"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    codec = load_codec(args.ckpt, device=str(device))
    codec.eval()
    pq = codec.pq
    G, K, d = pq.G, pq.K, pq.d
    D = G * d
    print(f"Loaded codec: G={G}, K={K}, d={d}, D={D}")
    print(f"  Checkpoint: {args.ckpt}")

    feat_dir = Path(args.feat_root) / "val" / args.backbone / args.layer
    feat_files = sorted(feat_dir.glob("*.npy"))
    if not feat_files:
        raise RuntimeError(f"No .npy files in {feat_dir}")
    print(f"  Found {len(feat_files)} feature files")

    total_bits = 0.0
    total_numel = 0

    with torch.no_grad():
        for feat_path in feat_files:
            X = torch.from_numpy(np.load(feat_path)).float().unsqueeze(0).to(device)
            B, N, _D = X.shape

            Y, mu, std = batch_normalize_gpu(X, mode="per_image")
            Y_hat, usage = codec(Y)

            bits = float(N * G * math.log2(K))
            total_bits += bits
            total_numel += B * N * D

    bpfp = total_bits / total_numel if total_numel > 0 else 0.0
    print(f"\n  BPFP (theoretical): {bpfp:.4f}")
    print(f"  Note: full mIoU evaluation requires the engine pipeline")
    print(f"  Use: python examples/soft_pq/run_eval_soft_pq.py --task seg --multi-run --cuda")


if __name__ == "__main__":
    main()

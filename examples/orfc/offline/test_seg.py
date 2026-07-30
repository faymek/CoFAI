#!/usr/bin/env python
"""
ORFC Segmentation Test — load pre-trained weights, evaluate VOC2012 mIoU + rate.
DINOv2 only (CLIP segmentation not supported).

Usage:
    python test_seg.py --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32
    python test_seg.py --backbone dinov2_vitg14 --layer blk19 --K 256 --embedding_dim 32
"""

import os
import sys
import argparse
import json
import math
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
SOURCE_ROOT = Path(__file__).resolve().parents[3]

OFFLINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OFFLINE_DIR)
sys.path.insert(0, str(SOURCE_ROOT))

import compressai  # noqa: F401

from cofai.latent_codecs.orfc_normalization import normalize_orfc_features
from cofai.ops.orfc import batched_assign
from utils import set_seed, preload_features
from backbone.wrapper import SegmentationEvaluator

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

try:
    import logging
    from mmcv.utils import get_logger
    logger = get_logger('mmcv')
    logger.setLevel(logging.WARNING)
except ImportError:
    pass


# ================================================================
#                    rANS Rate Evaluation
# ================================================================

try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans as _ans
    _HAS_ANS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ANS = False


def histogram_pmf(labels, G, K, smoothing=1.0):
    pmfs = []
    for g in range(G):
        counts = np.zeros(K, dtype=np.float64)
        np.add.at(counts, labels[g], 1)
        counts += smoothing
        pmfs.append(counts / counts.sum())
    return pmfs


def _build_rans_cdfs(pmf_np, G, K, precision=16):
    """Build per-group CDF tables for rANS (K+2 entries each)."""
    cdfs = []
    for g in range(G):
        p = torch.from_numpy(pmf_np[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdf = _pmf_to_quantized_cdf(p.tolist(), precision)
        cdfs.append(cdf)
    return cdfs


def rans_encode_bpt(labels_np, pmf_list, G, K, precision=16):
    if not _HAS_ANS:
        return None
    encoder = _ans.RansEncoder()
    N = labels_np.shape[1]
    cdfs = []
    cdf_sizes = []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdf = _pmf_to_quantized_cdf(p.tolist(), precision)
        cdfs.append(cdf)
        cdf_sizes.append(K + 2)
    symbols = []
    cdf_indices = []
    for n in range(N):
        for g in range(G):
            symbols.append(int(labels_np[g, n]))
            cdf_indices.append(g)
    byte_string = encoder.encode_with_indexes(
        symbols, cdf_indices, cdfs, cdf_sizes, [0] * G,
    )
    total_bits = len(byte_string) * 8
    return total_bits / N


def rans_encode_per_image_seg(image_features_list, codebooks, embedding_dim,
                              device, R, pmf_np):
    """Encode per-image with rANS for segmentation (matches CoFAI engine).

    Args:
        image_features_list: list of (T_i, D) arrays, one per image (all slides
                            from the same image concatenated along token dim).
        codebooks: list of G codebooks, each (K, d).
        R: rotation matrix (D, D).
        pmf_np: (G, K) PMF array from training.

    Returns:
        total_bits, total_tokens
    """
    if not _HAS_ANS:
        return None, None
    G = len(codebooks)
    K = codebooks[0].shape[0]
    C = image_features_list[0].shape[1]
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device)
    cdfs = _build_rans_cdfs(pmf_np, G, K)

    total_bits = 0
    total_tokens = 0

    for feat in image_features_list:
        T = feat.shape[0]
        X = torch.from_numpy(feat).float().to(device).unsqueeze(0)  # (1, T, D)
        with torch.no_grad():
            Y, _, _ = normalize_orfc_features(X, mode='per_image')
            flat = Y.reshape(-1, C)
            Z = flat @ R_t
            z_3d = Z.reshape(-1, G, embedding_dim).permute(1, 0, 2).contiguous()
            dists = torch.cdist(z_3d, cb_t)
            labels = dists.argmin(dim=-1).cpu().numpy()  # (G, T)

        img_bits = 0
        for g in range(G):
            encoder = _ans.RansEncoder()
            bs = encoder.encode_with_indexes(
                labels[g].tolist(),
                [0] * T,
                [cdfs[g]],
                [K + 2],
                [0],
            )
            img_bits += len(bs) * 8
        total_bits += img_bits
        total_tokens += T
        del X, Y, flat, Z, z_3d, dists, labels

    torch.cuda.empty_cache()
    return total_bits, total_tokens


def evaluate_rate(labels_test, labels_train, G, K):
    train_pmf = histogram_pmf(labels_train, G, K)
    test_pmf = histogram_pmf(labels_test, G, K, smoothing=0)
    N = labels_test.shape[1]

    xent_train = 0.0
    for g in range(G):
        log2_t = np.log2(train_pmf[g] + 1e-30)
        xent_train += -log2_t[labels_test[g]].sum()
    xent_train_bpt = xent_train / N

    empirical_entropy = 0.0
    for g in range(G):
        pg = test_pmf[g]
        pg = pg[pg > 0]
        empirical_entropy += -np.sum(pg * np.log2(pg))

    max_rate = G * math.log2(K)
    rans_train_bpt = rans_encode_bpt(labels_test, train_pmf, G, K)

    result = {
        'xent_train_bpt': float(xent_train_bpt),
        'empirical_entropy_bpt': float(empirical_entropy),
        'max_rate_bpt': float(max_rate),
    }
    if rans_train_bpt is not None:
        result['rans_train_bpt'] = float(rans_train_bpt)
    return result


# ================================================================
#                    PQ Label Extraction (for rate)
# ================================================================

def pq_get_labels(features, codebooks, embedding_dim, device, R=None,
                  chunk_images=200):
    num_groups = len(codebooks)
    C = features[0].shape[1]
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device) if R is not None else None
    all_labels = []
    for start in range(0, len(features), chunk_images):
        end = min(start + chunk_images, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        with torch.no_grad():
            Y, _, _ = normalize_orfc_features(X, mode='per_image')
            flat = Y.reshape(-1, C)
            Z = flat @ R_t if R_t is not None else flat
            z_3d = Z.reshape(-1, num_groups, embedding_dim) \
                    .permute(1, 0, 2).contiguous()
            dists = torch.cdist(z_3d, cb_t)
            labels = dists.argmin(dim=-1)
        all_labels.append(labels.cpu())
        del X, Y, flat, Z, z_3d, dists, labels
        torch.cuda.empty_cache()
    return torch.cat(all_labels, dim=1).numpy()


# ================================================================
#                    ORFC Segmentation Evaluator
# ================================================================

class ORFCSegmentationEvaluator(SegmentationEvaluator):
    """SegmentationEvaluator with ORFC encode/decode."""

    def __init__(self, R, codebooks, embedding_dim, layer_idx,
                 voc_root, weights_root, device='cuda', feat_dim=1024,
                 model_name='dinov2_vitl14'):
        self.R_t = torch.from_numpy(R).float().to(device)
        self.cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
        self.num_groups = len(codebooks)
        self.embedding_dim = embedding_dim
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.feat_dim = feat_dim
        self.model_name = model_name

    @torch.no_grad()
    def quantize_tokens(self, tokens_np):
        y, mu, std = self.normalize(tokens_np, mode='per_image')
        Y = torch.from_numpy(y).float().to(self.device).unsqueeze(0)
        flat = Y.reshape(-1, self.feat_dim)
        Z = flat @ self.R_t
        z_3d = Z.reshape(-1, self.num_groups, self.embedding_dim) \
                .permute(1, 0, 2).contiguous()
        z_hat_3d, _ = batched_assign(z_3d, self.cb_t, device=self.device)
        flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, self.feat_dim)
        Y_hat = (flat_hat @ self.R_t.T).reshape(1, -1, self.feat_dim)
        Mu = torch.from_numpy(mu).float().to(self.device).unsqueeze(0)
        Std = torch.from_numpy(std).float().to(self.device).unsqueeze(0)
        X_hat = Y_hat * Std + Mu
        return X_hat.squeeze(0)


# ================================================================
#                    Weight Loading
# ================================================================

def load_orfc_weights(weights_path):
    """Load ORFC weights from .npz file (R, codebooks, pmf)."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    data = np.load(weights_path)
    R = data['R']
    codebooks_arr = data['codebooks']  # [num_groups, K, embedding_dim]
    codebooks = [codebooks_arr[g] for g in range(codebooks_arr.shape[0])]
    pmf = data['pmf'] if 'pmf' in data else None  # (G, K)
    print(f"  Loaded weights: {weights_path}")
    print(f"    R: {R.shape}, codebooks: {codebooks_arr.shape}")
    if pmf is not None:
        print(f"    pmf: {pmf.shape}")
    return R, codebooks, pmf


# ================================================================
#                    Main
# ================================================================

def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])

    if args.backbone.startswith("clip"):
        print("ERROR: Segmentation evaluation is only supported for DINOv2.")
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"# ORFC Segmentation Test")
    print(f"# backbone={args.backbone}, layer={args.layer} (idx={layer_idx})")
    print(f"# K={args.K}, embedding_dim={args.embedding_dim}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load weights ----
    weights_path = os.path.join(
        args.weights_dir, args.backbone,
        f"{args.layer}_K{args.K}_e{args.embedding_dim}.npz"
    )
    R, codebooks, pmf_np = load_orfc_weights(weights_path)
    num_groups = len(codebooks)
    embedding_dim = codebooks[0].shape[1]
    K = codebooks[0].shape[0]
    D = num_groups * embedding_dim

    bits_per_token = num_groups * math.log2(K)
    print(f"  D={D}, G={num_groups}, d={embedding_dim}, K={K}")
    print(f"  bits/token={bits_per_token:.0f}, BPFP={bits_per_token/D:.4f}")

    # ---- Segmentation evaluation ----
    seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer

    print(f"\n{'=' * 60}")
    print(f"  [Segmentation] VOC2012 mIoU")
    print(f"  seg features: {seg_feat_dir}")
    print(f"{'=' * 60}")

    orfc_seg = ORFCSegmentationEvaluator(
        R=R, codebooks=codebooks,
        embedding_dim=embedding_dim,
        layer_idx=layer_idx,
        voc_root=args.voc_root,
        weights_root=args.weights_root,
        device=device,
        feat_dim=D,
        model_name=args.backbone,
    )
    seg_result = orfc_seg.evaluate(
        seg_feat_dir=str(seg_feat_dir),
        image_list=args.seg_image_list, verbose=True,
    )
    miou = float(seg_result['miou'])
    seg_acc = float(seg_result['acc'])
    print(f"  * mIoU = {miou:.4f} ({miou*100:.2f}%)")
    print(f"  * pixel acc = {seg_acc:.4f}")
    del orfc_seg

    # ---- Rate evaluation on VOC features (per-image) ----
    print(f"\n{'=' * 60}")
    print(f"  [Rate] Per-image rANS encoding (real codec rate)")
    print(f"{'=' * 60}")

    if args.seg_image_list and os.path.exists(args.seg_image_list):
        with open(args.seg_image_list) as fimg:
            seg_names = [l.strip() for l in fimg if l.strip()]
    else:
        seg_names = []

    # Group all slides per image into one feature array
    image_features_list = []
    seg_features_flat = []
    for sname in seg_names:
        fp = os.path.join(str(seg_feat_dir), f"{sname}.npy")
        if os.path.exists(fp):
            darr = np.load(fp)  # (num_slides, T, D)
            num_slides = darr.shape[0]
            img_feat = darr.reshape(-1, darr.shape[-1])  # (num_slides*T, D)
            image_features_list.append(img_feat)
            for si in range(num_slides):
                seg_features_flat.append(darr[si])

    rans_per_image_bpt = None
    rans_per_image_bpfp = None
    if pmf_np is not None and image_features_list:
        print(f"  Encoding per-image with stored PMF ({len(image_features_list)} images)...")
        total_bits, total_tokens = rans_encode_per_image_seg(
            image_features_list, codebooks, embedding_dim, device, R, pmf_np,
        )
        if total_bits is not None:
            rans_per_image_bpt = total_bits / total_tokens
            rans_per_image_bpfp = total_bits / (total_tokens * D)
            print(f"  * per-image rANS: {rans_per_image_bpt:.4f} bits/token, "
                  f"BPFP={rans_per_image_bpfp:.6f}")
    elif pmf_np is None:
        print(f"  [warn] No PMF in weights, falling back to dataset-level rate")

    # Dataset-level rate (for reference)
    train_dir = Path(args.feat_root) / "train" / args.backbone / args.layer
    train_files = sorted(train_dir.glob("*.npy"))
    features_train, _ = preload_features(train_files, num_workers=8)
    if args.max_train_images > 0 and len(features_train) > args.max_train_images:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(features_train), args.max_train_images, replace=False)
        features_train = [features_train[i] for i in idx]

    print(f"  Computing train labels ({len(features_train)} images)...")
    labels_train = pq_get_labels(features_train, codebooks, embedding_dim, device, R=R)
    del features_train

    rate_info = None
    if seg_features_flat:
        print(f"  Computing VOC seg labels ({len(seg_features_flat)} slides, dataset-level)...")
        seg_labels = pq_get_labels(
            seg_features_flat, codebooks, embedding_dim, device, R=R)
        rate_info = evaluate_rate(seg_labels, labels_train, num_groups, K)
        rans_str = (f"rANS(dataset)={rate_info['rans_train_bpt']:.4f}"
                    if 'rans_train_bpt' in rate_info else "rANS=N/A")
        print(f"  * dataset-level: xent={rate_info['xent_train_bpt']:.4f}  "
              f"H={rate_info['empirical_entropy_bpt']:.4f}  "
              f"max={rate_info['max_rate_bpt']:.0f}  {rans_str}  bits/token")
    else:
        print("  [warn] No VOC features found for rate evaluation")

    # ---- Summary ----
    bpfp_fixed = bits_per_token / D
    bpfp_real = rans_per_image_bpfp if rans_per_image_bpfp is not None else \
                (rate_info.get('rans_train_bpt', bits_per_token) / D if rate_info else bpfp_fixed)

    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.backbone} / {args.layer} / K={K} / e={embedding_dim}")
    print(f"  mIoU = {miou*100:.2f}%")
    print(f"  BPFP(fixed)      = {bpfp_fixed:.6f}")
    print(f"  BPFP(real/image) = {bpfp_real:.6f}")
    print(f"{'=' * 60}")

    # ---- Save results ----
    results = {
        'backbone': args.backbone,
        'layer': args.layer,
        'layer_idx': layer_idx,
        'K': K,
        'embedding_dim': embedding_dim,
        'num_groups': num_groups,
        'bits_per_token': float(bits_per_token),
        'bpfp_fixed': float(bpfp_fixed),
        'bpfp': float(bpfp_real),
        'miou': miou,
        'seg_acc': seg_acc,
    }
    if rate_info:
        results['rate_info'] = rate_info
    if rans_per_image_bpt is not None:
        results['rans_per_image_bpt'] = float(rans_per_image_bpt)
        results['rans_per_image_bpfp'] = float(rans_per_image_bpfp)

    out_dir = os.path.join(PROJECT_ROOT, 'results', 'orfc', args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"seg_{args.layer}_K{K}_e{embedding_dim}"
    out_path = os.path.join(out_dir, f'{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="ORFC segmentation test (load weights, DINOv2 only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--embedding_dim", type=int, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_images", type=int, default=5000,
                        help="Training images for rate PMF estimation")

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "orfc"))
    parser.add_argument("--weights_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "orfc"))
    parser.add_argument("--weights_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "pretrained"),
                        help="Pretrained backbone weights root")
    parser.add_argument("--seg_feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "orfc", "voc2012_100"))
    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data",
                                             "VOC2012_sel100"))
    parser.add_argument("--seg_image_list", type=str,
                        default=os.path.join(OFFLINE_DIR, "cfg",
                                             "voc2012_val_100.txt"))

    args = parser.parse_args()
    run_test(args)


if __name__ == '__main__':
    main()

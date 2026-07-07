#!/usr/bin/env python
"""
Soft-PQ Segmentation Test — load trained codec, evaluate VOC2012 mIoU + rate.
DINOv2 only (CLIP segmentation not supported).

Usage:
    python test_seg.py --backbone dinov2_vitl14 --layer blk10 \
        --ckpt_path ../../weights/orfc_2446/dinov2_vitl14_ori/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt

    python test_seg.py --backbone dinov2_vitg14 --layer blk19 \
        --K 8 --embedding_dim 32
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
PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
)

OFFLINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OFFLINE_DIR)
sys.path.insert(0, PROJECT_ROOT)

import compressai  # noqa: F401

from cofai.entropy_models.soft_pq import (
    load_codec, soft_pq_encode_decode,
)
from cofai.entropy_models.soft_pq_export import (
    try_load_sidecar_pmf, pmf_as_list, npz_path_for_codec,
)
from cofai.entropy_models.orfc_model import batch_normalize_gpu, batch_inv_normalize_gpu
from utils import set_seed, preload_features
from weights_paths import codec_weights_dir
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


def _codec_labels(features, codec, norm_mode, device, batch_size=32):
    """Run codec on features and return hard labels [G, N_total]."""
    codec.eval()
    pq = codec.pq
    all_labels = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            X = torch.from_numpy(
                np.stack(features[start:end])
            ).float().to(device)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            _ = codec(Y)
            all_labels.append(pq._last_labels.cpu())
            del X, Y
    return torch.cat(all_labels, dim=1).numpy()


def _histogram_pmf(labels, G, K, smoothing=1.0):
    pmfs = []
    for g in range(G):
        counts = np.zeros(K, dtype=np.float64)
        np.add.at(counts, labels[g], 1)
        counts += smoothing
        pmfs.append(counts / counts.sum())
    return pmfs


def _rans_encode_bpt(labels_np, pmf_list, G, K, precision=16):
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


def _build_rans_cdfs(pmf_list, G, K, precision=16):
    cdfs = []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdf = _pmf_to_quantized_cdf(p.tolist(), precision)
        cdfs.append(cdf)
    return cdfs


def rans_encode_per_image_seg(image_features_list, codec, norm_mode, device, pmf_np):
    """Encode per-image with rANS for segmentation using stored train PMF."""
    if not _HAS_ANS or pmf_np is None:
        return None, None
    codec.eval()
    pq = codec.pq
    G, K = pq.G, pq.K
    pmf_list = pmf_as_list(pmf_np)
    cdfs = _build_rans_cdfs(pmf_list, G, K)

    total_bits = 0
    total_tokens = 0

    with torch.no_grad():
        for feat in image_features_list:
            T = feat.shape[0]
            X = torch.from_numpy(feat).float().to(device).unsqueeze(0)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            _ = codec(Y)
            labels = pq._last_labels.cpu().numpy()  # [G, T]

            img_bits = 0
            for g in range(G):
                enc = _ans.RansEncoder()
                bs = enc.encode_with_indexes(
                    labels[g].tolist(),
                    [0] * T,
                    [cdfs[g]],
                    [K + 2],
                    [0],
                )
                img_bits += len(bs) * 8
            total_bits += img_bits
            total_tokens += T
            del X, Y

    torch.cuda.empty_cache()
    return total_bits, total_tokens


def evaluate_rate_seg(labels_test, pmf_np, G, K):
    """Dataset-level rate on seg labels using stored train PMF."""
    train_pmf = pmf_as_list(pmf_np)
    test_pmf = _histogram_pmf(labels_test, G, K, smoothing=0)
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
    rans_train_bpt = _rans_encode_bpt(labels_test, train_pmf, G, K)
    result = {
        'xent_train_bpt': float(xent_train_bpt),
        'empirical_entropy_bpt': float(empirical_entropy),
        'max_rate_bpt': float(max_rate),
    }
    if rans_train_bpt is not None:
        result['rans_train_bpt'] = float(rans_train_bpt)
    return result


# ================================================================
#                    Codec Segmentation Evaluator
# ================================================================

class CodecSegmentationEvaluator(SegmentationEvaluator):
    """SegmentationEvaluator with FeatureCodec encode/decode."""

    def __init__(self, codec, norm_mode, layer_idx,
                 voc_root, weights_root, device='cuda', feat_dim=1024,
                 model_name='dinov2_vitl14'):
        self.codec = codec
        self.norm_mode = norm_mode
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.feat_dim = feat_dim
        self.model_name = model_name

    @torch.no_grad()
    def quantize_tokens(self, tokens_np):
        X = torch.from_numpy(tokens_np).float().unsqueeze(0).to(self.device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=self.norm_mode)
        Y_hat, _ = self.codec(Y)
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        return X_hat.squeeze(0)


# ================================================================
#                    Main
# ================================================================

def resolve_ckpt_path(args):
    """Resolve checkpoint path from explicit --ckpt_path or from naming convention."""
    if args.ckpt_path:
        return args.ckpt_path
    D_map = {"dinov2_vitl14": 1024, "dinov2_vitg14": 1536}
    D = D_map[args.backbone]
    bt_dim = args.bottleneck_dim if args.bottleneck_dim > 0 else D
    bt_tag = f"bt{bt_dim}"
    ws_tag = "ws" if args.warm_start_opq else "km"
    mse_tag = "_mse" if args.mse_loss else ""
    rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    ckpt_name = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
                 f"_{bt_tag}_{ws_tag}{mse_tag}{rate_tag}{tau_tag}"
                 f"_lr{args.lr}_ep{args.epochs}"
                 f"_n{args.max_train_images}_s{args.seed}.pt")
    return os.path.join(codec_weights_dir(args.weights_dir, args.backbone), ckpt_name)


def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])

    if args.backbone.startswith("clip"):
        print("ERROR: Segmentation evaluation is only supported for DINOv2.")
        sys.exit(1)

    print(f"\n{'#' * 70}")
    print(f"# Soft-PQ Segmentation Test")
    print(f"# backbone={args.backbone}, layer={args.layer} (idx={layer_idx})")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load codec ----
    ckpt_path = resolve_ckpt_path(args)
    print(f"\n  Loading codec: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    codec = load_codec(ckpt_path, device=device)

    pq = codec.pq
    G, K, d = pq.G, pq.K, pq.d
    D = G * d
    bits_per_token = G * math.log2(K)
    print(f"  G={G}, K={K}, d={d}, D={D}")
    print(f"  bits/token={bits_per_token:.0f}, BPFP={bits_per_token/D:.4f}")

    pmf_np = try_load_sidecar_pmf(ckpt_path)
    if pmf_np is not None:
        print(f"  PMF sidecar: {npz_path_for_codec(ckpt_path)}")
    else:
        print(f"  [warn] No PMF sidecar at {npz_path_for_codec(ckpt_path)}; "
              f"rate metrics will be unavailable")

    # ---- Segmentation evaluation ----
    seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer

    print(f"\n{'=' * 60}")
    print(f"  [Segmentation] VOC2012 mIoU")
    print(f"  seg features: {seg_feat_dir}")
    print(f"{'=' * 60}")

    codec_seg = CodecSegmentationEvaluator(
        codec=codec, norm_mode=args.norm_mode,
        layer_idx=layer_idx,
        voc_root=args.voc_root,
        weights_root=args.weights_root,
        device=device,
        feat_dim=D,
        model_name=args.backbone,
    )
    seg_result = codec_seg.evaluate(
        seg_feat_dir=str(seg_feat_dir),
        image_list=args.seg_image_list, verbose=True,
    )
    miou = float(seg_result['miou'])
    seg_acc = float(seg_result['acc'])
    print(f"  * mIoU = {miou:.4f} ({miou*100:.2f}%)")
    print(f"  * pixel acc = {seg_acc:.4f}")
    del codec_seg

    # ---- Rate evaluation ----
    print(f"\n{'=' * 60}")
    print(f"  [Rate] Per-image rANS (stored train PMF)")
    print(f"{'=' * 60}")

    if args.seg_image_list and os.path.exists(args.seg_image_list):
        with open(args.seg_image_list) as fimg:
            seg_names = [l.strip() for l in fimg if l.strip()]
    else:
        seg_names = []

    image_features_list = []
    for sname in seg_names:
        fp = os.path.join(str(seg_feat_dir), f"{sname}.npy")
        if os.path.exists(fp):
            darr = np.load(fp)
            img_feat = darr.reshape(-1, darr.shape[-1])
            image_features_list.append(img_feat)

    rans_per_image_bpfp = None
    rate_info = None
    if image_features_list and pmf_np is not None:
        total_bits, total_tokens = rans_encode_per_image_seg(
            image_features_list, codec, args.norm_mode, device, pmf_np,
        )
        if total_bits is not None:
            rans_per_image_bpt = total_bits / total_tokens
            rans_per_image_bpfp = total_bits / (total_tokens * D)
            print(f"  * per-image rANS: {rans_per_image_bpt:.4f} bits/token, "
                  f"BPFP={rans_per_image_bpfp:.6f}")

        seg_features_flat = []
        for sname in seg_names:
            fp = os.path.join(str(seg_feat_dir), f"{sname}.npy")
            if os.path.exists(fp):
                darr = np.load(fp)
                for si in range(darr.shape[0]):
                    seg_features_flat.append(darr[si])

        if seg_features_flat:
            print(f"  Computing dataset-level rate...")
            seg_labels = _codec_labels(
                seg_features_flat, codec, args.norm_mode, device, batch_size=1,
            )
            rate_info = evaluate_rate_seg(seg_labels, pmf_np, G, K)
            rans_str = (f"rANS(dataset)={rate_info['rans_train_bpt']:.4f}"
                        if rate_info.get('rans_train_bpt') else "rANS=N/A")
            print(f"  * dataset-level: xent={rate_info['xent_train_bpt']:.4f}  "
                  f"H={rate_info['empirical_entropy_bpt']:.4f}  "
                  f"max={rate_info['max_rate_bpt']:.0f}  {rans_str}  bits/token")

    torch.cuda.empty_cache()

    # ---- Summary ----
    bpfp_fixed = bits_per_token / D
    bpfp_real = rans_per_image_bpfp if rans_per_image_bpfp is not None else \
                (rate_info.get('rans_train_bpt', bits_per_token) / D
                 if rate_info else bpfp_fixed)

    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.backbone} / {args.layer} / K={K} / d={d}")
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
        'embedding_dim': d,
        'num_groups': G,
        'bits_per_token': float(bits_per_token),
        'bpfp_fixed': float(bpfp_fixed),
        'bpfp': float(bpfp_real),
        'miou': miou,
        'seg_acc': seg_acc,
        'ckpt_path': ckpt_path,
        'npz_path': str(npz_path_for_codec(ckpt_path)),
    }
    if rate_info:
        results['rate_info'] = rate_info
    if rans_per_image_bpfp is not None:
        results['rans_per_image_bpfp'] = float(rans_per_image_bpfp)

    out_dir = os.path.join(PROJECT_ROOT, 'results', 'orfc_2446', args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_stem = Path(ckpt_path).stem
    out_path = os.path.join(out_dir, f'seg_{ckpt_stem}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Soft-PQ segmentation test (load codec, DINOv2 only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Explicit checkpoint path (overrides auto-resolve)")
    parser.add_argument("--norm_mode", type=str, default="per_image")

    # For auto-resolving ckpt_path
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--bottleneck_dim", type=int, default=0)
    parser.add_argument("--lmbda", type=float, default=0.5)
    parser.add_argument("--tau_start", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--warm_start_opq", action="store_true", default=True)
    parser.add_argument("--no_warm_start", dest="warm_start_opq",
                        action="store_false")
    parser.add_argument("--mse_loss", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "orfc"))
    parser.add_argument("--weights_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "orfc_2446"))
    parser.add_argument("--weights_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "pretrained"),
                        help="Pretrained backbone weights root")
    parser.add_argument("--seg_feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "orfc",
                                             "voc2012_100"))
    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "VOC2012_sel100"))
    parser.add_argument("--seg_image_list", type=str,
                        default=os.path.join(OFFLINE_DIR, "cfg",
                                             "voc2012_val_100.txt"))

    args = parser.parse_args()
    run_test(args)


if __name__ == '__main__':
    main()

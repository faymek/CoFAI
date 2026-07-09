#!/usr/bin/env python
"""
PQFC Classification Test — load trained codec, evaluate accuracy + rate.

Usage:
    python test_cls.py --backbone dinov2_vitl14 --layer blk10 \
        --ckpt_path ../../weights/pqfc/dinov2_vitl14/blk10_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt

    python test_cls.py --backbone dinov2_vitl14 --layer blk10 \
        --K 64 --embedding_dim 32
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

# Unbuilt CompressAI source trees must not shadow the pip install (missing _CXX).
_BAD_COMPRESSAI_MARKERS = ("ORFC/coding/CompressAI", "coding/CompressAI")


def _scrub_compressai_source_path():
    def _bad(p: str) -> bool:
        norm = p.replace("\\", "/")
        return any(m in norm for m in _BAD_COMPRESSAI_MARKERS)

    sys.path[:] = [p for p in sys.path if p and not _bad(p)]
    pypath = os.environ.get("PYTHONPATH", "")
    if pypath:
        parts = [p for p in pypath.split(os.pathsep) if p and not _bad(p)]
        if parts:
            os.environ["PYTHONPATH"] = os.pathsep.join(parts)
        else:
            os.environ.pop("PYTHONPATH", None)


_scrub_compressai_source_path()

import compressai  # noqa: F401

from cofai.entropy_models.soft_pq import (
    load_codec, soft_pq_encode_decode,
)
from cofai.entropy_models.soft_pq_export import (
    try_load_sidecar_pmf, pmf_as_list, npz_path_for_codec,
)
from cofai.entropy_models.orfc_model import batch_normalize_gpu
from utils import set_seed, preload_features, load_gt, evaluate_accuracy
from weights_paths import codec_weights_dir
from backbone.wrapper import Dinov2Wrapper, ClipWrapper

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


def _build_rans_cdfs(pmf_np, G, K, precision=16):
    cdfs = []
    for g in range(G):
        p = torch.from_numpy(pmf_np[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdf = _pmf_to_quantized_cdf(p.tolist(), precision)
        cdfs.append(cdf)
    return cdfs


def rans_encode_per_image(features, codec, norm_mode, device, pmf_np,
                          chunk_images=200):
    """Encode per-image with rANS using stored train-set PMF from sidecar .npz."""
    if not _HAS_ANS:
        return None, None
    if pmf_np is None:
        return None, None
    codec.eval()
    pq = codec.pq
    G, K = pq.G, pq.K
    pmf_list = pmf_as_list(pmf_np)
    cdfs = _build_rans_cdfs(pmf_list, G, K)

    total_bits = 0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(features), chunk_images):
            end = min(start + chunk_images, len(features))
            X = torch.from_numpy(
                np.stack(features[start:end])
            ).float().to(device)
            B, T_img, _ = X.shape
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            _ = codec(Y)
            labels_all = pq._last_labels.cpu().numpy()  # [G, B*T]

            for img_idx in range(B):
                img_labels = labels_all[:, img_idx * T_img:(img_idx + 1) * T_img]
                N = T_img
                img_bits = 0
                for g in range(G):
                    enc = _ans.RansEncoder()
                    bs = enc.encode_with_indexes(
                        img_labels[g].tolist(),
                        [0] * N,
                        [cdfs[g]],
                        [K + 2],
                        [0],
                    )
                    img_bits += len(bs) * 8
                total_bits += img_bits
                total_tokens += N

            del X, Y
            torch.cuda.empty_cache()

    return total_bits, total_tokens


def evaluate_rate(labels_test, pmf_np, G, K):
    """Compute rate metrics on test labels using stored train PMF."""
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
#                    Main
# ================================================================

def resolve_ckpt_path(args):
    """Resolve checkpoint path from explicit --ckpt_path or from naming convention."""
    if args.ckpt_path:
        return args.ckpt_path
    # Auto-resolve from args
    D_map = {"dinov2_vitl14": 1024, "dinov2_vitg14": 1536, "clip_vitl14": 1024}
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

    print(f"\n{'#' * 70}")
    print(f"# PQFC Classification Test")
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

    # ---- Load features ----
    test_dir = Path(args.feat_root) / "test" / args.backbone / args.layer
    test_files = sorted(test_dir.glob("*.npy"))

    print(f"\n  Loading features: test={len(test_files)}")
    features_test, basenames_test = preload_features(test_files, num_workers=8)
    gt_test = load_gt(args.gt_path)

    # ---- Load backbone ----
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

    # ---- Classification ----
    print(f"\n{'=' * 60}")
    print(f"  [Classification] ImageNet top-1 accuracy")
    print(f"{'=' * 60}")

    xhat = soft_pq_encode_decode(
        features_test, codec, args.norm_mode, device,
    )
    acc = evaluate_accuracy(xhat, basenames_test, gt_test, wrapper, layer_idx, device)
    print(f"  * Accuracy = {acc:.4f} ({acc*100:.2f}%)")
    del xhat

    # ---- Rate ----
    print(f"\n{'=' * 60}")
    print(f"  [Rate] Per-image rANS (stored train PMF)")
    print(f"{'=' * 60}")

    wrapper.backbone.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    rans_per_image_bpfp = None
    rate_info = None
    if pmf_np is not None:
        total_bits, total_tokens = rans_encode_per_image(
            features_test, codec, args.norm_mode, device, pmf_np,
        )
        if total_bits is not None:
            rans_per_image_bpt = total_bits / total_tokens
            rans_per_image_bpfp = total_bits / (total_tokens * D)
            print(f"  * per-image rANS: {rans_per_image_bpt:.4f} bits/token, "
                  f"BPFP={rans_per_image_bpfp:.6f}")

        print(f"  Computing test labels for dataset-level rate...")
        labels_test_np = _codec_labels(
            features_test, codec, args.norm_mode, device,
        )
        rate_info = evaluate_rate(labels_test_np, pmf_np, G, K)
        rans_str = (f"rANS(dataset)={rate_info['rans_train_bpt']:.4f}"
                    if 'rans_train_bpt' in rate_info else "rANS=N/A")
        print(f"  * dataset-level: xent={rate_info['xent_train_bpt']:.4f}  "
              f"H={rate_info['empirical_entropy_bpt']:.4f}  "
              f"max={rate_info['max_rate_bpt']:.0f}  {rans_str}  bits/token")

    # ---- Summary ----
    bpfp_fixed = bits_per_token / D
    bpfp_real = rans_per_image_bpfp if rans_per_image_bpfp is not None else (
        rate_info.get('rans_train_bpt', bits_per_token) / D if rate_info else bpfp_fixed
    )

    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.backbone} / {args.layer} / K={K} / d={d}")
    print(f"  Acc@1 = {acc*100:.2f}%")
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
        'accuracy': float(acc),
        'rate_info': rate_info or {},
        'npz_path': str(npz_path_for_codec(ckpt_path)),
        'ckpt_path': ckpt_path,
    }
    if rans_per_image_bpfp is not None:
        results['rans_per_image_bpfp'] = float(rans_per_image_bpfp)

    out_dir = os.path.join(PROJECT_ROOT, 'results', 'pqfc', args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_stem = Path(ckpt_path).stem
    out_path = os.path.join(out_dir, f'cls_{ckpt_stem}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Soft-PQ classification test (load codec checkpoint)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14", "clip_vitl14"])
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="Explicit checkpoint path (overrides auto-resolve)")
    parser.add_argument("--norm_mode", type=str, default="per_image")

    # For auto-resolving ckpt_path if not explicit
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
                        default=os.path.join(PROJECT_ROOT, "weights", "pqfc"))
    parser.add_argument("--weights_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "weights", "pretrained"),
                        help="Pretrained backbone weights root")
    parser.add_argument("--classnames", type=str,
                        default=os.path.join(OFFLINE_DIR, "cfg", "classnames.txt"))
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(OFFLINE_DIR, "cfg",
                                             "imagenet_selected_label500.txt"))

    args = parser.parse_args()
    run_test(args)


if __name__ == '__main__':
    main()

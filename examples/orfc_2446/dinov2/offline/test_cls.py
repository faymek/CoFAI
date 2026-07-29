#!/usr/bin/env python
"""
Soft-PQ Classification Test — load trained codec, evaluate accuracy + rate.

Usage:
    PYTHONPATH="$SOURCE_ROOT" poetry -C "$PROJECT_ROOT" run python \
        "$SOURCE_ROOT/examples/orfc_2446/dinov2/offline/test_cls.py" \
        --backbone dinov2_vitl14 --layer blk10 --ckpt_path <artifact>.npz
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

OFFLINE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(SOURCE_ROOT / ".env")
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError(f"PROJECT_ROOT is required; set it or add it to {SOURCE_ROOT / '.env'}")
ORFC_OFFLINE_DIR = SOURCE_ROOT / "examples" / "orfc" / "offline"
sys.path.insert(0, OFFLINE_DIR)
sys.path.insert(0, str(SOURCE_ROOT))

from cofai.latent_codecs.orfc_normalization import ORFC_NORM_MODES
from examples.orfc.offline.utils import (
    evaluate_accuracy,
    load_gt,
    preload_features,
    set_seed,
)
from eval_utils import load_eval_codec, real_rate_bits, reconstruct_features, resolve_artifact_path
from examples.orfc.offline.backbone.wrapper import Dinov2Wrapper

import warnings

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

try:
    import logging
    from mmcv.utils import get_logger

    logger = get_logger("mmcv")
    logger.setLevel(logging.WARNING)
except ImportError:
    pass


# ================================================================
#                    Main
# ================================================================


def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    print(f"# Soft-PQ Classification Test")
    print(f"# backbone={args.backbone}, layer={args.layer} (idx={layer_idx})")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load codec (.npz only) ----
    ckpt_path = resolve_artifact_path(args)
    print(f"\n  Loading codec: {ckpt_path}")
    latent = load_eval_codec(ckpt_path, device, norm_mode=args.norm_mode)

    G, K, d = latent.G, latent.K, latent.embedding_dim
    D = latent.feat_dim
    bits_per_token = G * math.log2(K)
    print(f"  G={G}, K={K}, d={d}, D={D}")
    print(f"  bits/token={bits_per_token:.0f}, BPFP={bits_per_token/D:.4f}")
    print(f"  norm_mode={latent.norm_mode} n_prefix={latent.n_prefix}")
    print("  PMF loaded from npz (real rANS enabled)")

    # ---- Load features ----
    test_dir = Path(args.feat_root) / "test" / args.backbone / args.layer
    test_files = sorted(test_dir.glob("*.npy"))

    print(f"\n  Loading features: test={len(test_files)}")
    features_test, basenames_test = preload_features(test_files, num_workers=8)
    gt_test = load_gt(args.gt_path)

    # ---- Load backbone ----
    print(f"\n  Loading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(
        head_layers=1,
        model_name=args.backbone,
        weights_root=args.weights_root,
        device=device,
    )

    # ---- Classification ----
    print(f"\n{'=' * 60}")
    print(f"  [Classification] ImageNet top-1 accuracy")
    print(f"{'=' * 60}")

    xhat = reconstruct_features(features_test, latent, device)
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
    total_bits, total_tokens = real_rate_bits(features_test, latent, device)
    if total_tokens > 0:
        rans_per_image_bpt = total_bits / total_tokens
        rans_per_image_bpfp = total_bits / (total_tokens * D)
        print(f"  * per-image rate: {rans_per_image_bpt:.4f} bits/token, " f"BPFP={rans_per_image_bpfp:.6f}")
        rate_info = {
            "bits_per_token": rans_per_image_bpt,
            "bpfp": rans_per_image_bpfp,
            "has_pmf": latent.pmf is not None,
        }

    # ---- Summary ----
    bpfp_fixed = bits_per_token / D
    bpfp_real = (
        rans_per_image_bpfp
        if rans_per_image_bpfp is not None
        else (rate_info.get("rans_train_bpt", bits_per_token) / D if rate_info else bpfp_fixed)
    )

    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.backbone} / {args.layer} / K={K} / d={d}")
    print(f"  Acc@1 = {acc*100:.2f}%")
    print(f"  BPFP(fixed)      = {bpfp_fixed:.6f}")
    print(f"  BPFP(real/image) = {bpfp_real:.6f}")
    print(f"{'=' * 60}")

    # ---- Save results ----
    results = {
        "backbone": args.backbone,
        "layer": args.layer,
        "layer_idx": layer_idx,
        "K": K,
        "embedding_dim": d,
        "num_groups": G,
        "bits_per_token": float(bits_per_token),
        "bpfp_fixed": float(bpfp_fixed),
        "bpfp": float(bpfp_real),
        "accuracy": float(acc),
        "rate_info": rate_info or {},
        "npz_path": str(ckpt_path),
        "ckpt_path": ckpt_path,
    }
    if rans_per_image_bpfp is not None:
        results["rans_per_image_bpfp"] = float(rans_per_image_bpfp)

    out_dir = os.path.join(PROJECT_ROOT, "results", "orfc_2446", args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_stem = Path(ckpt_path).stem
    out_path = os.path.join(out_dir, f"cls_{ckpt_stem}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Soft-PQ classification test (load codec checkpoint)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, required=True, choices=["dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="", help="Explicit checkpoint path (overrides auto-resolve)")
    parser.add_argument("--norm_mode", choices=ORFC_NORM_MODES, default=None)

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
    parser.add_argument("--no_warm_start", dest="warm_start_opq", action="store_false")
    parser.add_argument("--mse_loss", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feat_root", type=str, default=os.path.join(PROJECT_ROOT, "features", "orfc"))
    parser.add_argument("--weights_dir", type=str, default=os.path.join(PROJECT_ROOT, "weights", "orfc_2446"))
    parser.add_argument(
        "--weights_root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "weights", "pretrained"),
        help="Pretrained backbone weights root",
    )
    parser.add_argument(
        "--gt_path", type=str, default=str(ORFC_OFFLINE_DIR / "cfg" / "imagenet_selected_label500.txt")
    )

    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()

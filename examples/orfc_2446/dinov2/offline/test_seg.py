#!/usr/bin/env python
"""
Soft-PQ Segmentation Test — load trained codec, evaluate VOC2012 mIoU + rate.
DINOv2 only (CLIP segmentation not supported).

Usage:
    PYTHONPATH="$SOURCE_ROOT" poetry -C "$PROJECT_ROOT" run python \
        "$SOURCE_ROOT/examples/orfc_2446/dinov2/offline/test_seg.py" \
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
SHARED_OFFLINE_DIR = SOURCE_ROOT / "examples" / "orfc" / "offline"
sys.path.insert(0, OFFLINE_DIR)
sys.path.insert(0, str(SOURCE_ROOT))

from cofai.latent_codecs.orfc_normalization import ORFC_NORM_MODES
from examples.orfc_2446.offline.artifacts import npz_path_for_codec
from examples.orfc.offline.utils import set_seed
from eval_utils import load_eval_codec, real_rate_bits, resolve_artifact_path
from examples.orfc.offline.backbone.wrapper import SegmentationEvaluator

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
#                    Codec Segmentation Evaluator
# ================================================================


class CodecSegmentationEvaluator(SegmentationEvaluator):
    """SegmentationEvaluator with the shared ORFC latent codec."""

    def __init__(
        self,
        codec,
        norm_mode,
        layer_idx,
        voc_root,
        weights_root,
        device="cuda",
        feat_dim=1024,
        model_name="dinov2_vitl14",
    ):
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
        # The shared ORFC codec applies norm + PQ + denorm internally.
        h_hat = self.codec(X)["h_hat"]
        return h_hat.squeeze(0)


# ================================================================
#                    Main
# ================================================================


def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    print(f"# Soft-PQ Segmentation Test")
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

    # ---- Segmentation evaluation ----
    seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer

    print(f"\n{'=' * 60}")
    print(f"  [Segmentation] VOC2012 mIoU")
    print(f"  seg features: {seg_feat_dir}")
    print(f"{'=' * 60}")

    codec_seg = CodecSegmentationEvaluator(
        codec=latent,
        norm_mode=args.norm_mode,
        layer_idx=layer_idx,
        voc_root=args.voc_root,
        weights_root=args.weights_root,
        device=device,
        feat_dim=D,
        model_name=args.backbone,
    )
    seg_result = codec_seg.evaluate(
        seg_feat_dir=str(seg_feat_dir),
        image_list=args.seg_image_list,
        verbose=True,
    )
    miou = float(seg_result["miou"])
    seg_acc = float(seg_result["acc"])
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
    if image_features_list:
        total_bits, total_tokens = real_rate_bits(image_features_list, latent, device)
        if total_tokens > 0:
            rans_per_image_bpt = total_bits / total_tokens
            rans_per_image_bpfp = total_bits / (total_tokens * D)
            print(f"  * per-image rate: {rans_per_image_bpt:.4f} bits/token, " f"BPFP={rans_per_image_bpfp:.6f}")
            rate_info = {
                "bits_per_token": rans_per_image_bpt,
                "bpfp": rans_per_image_bpfp,
                "has_pmf": latent.pmf is not None,
            }
    torch.cuda.empty_cache()

    # ---- Summary ----
    bpfp_fixed = bits_per_token / D
    bpfp_real = (
        rans_per_image_bpfp
        if rans_per_image_bpfp is not None
        else (rate_info.get("rans_train_bpt", bits_per_token) / D if rate_info else bpfp_fixed)
    )

    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.backbone} / {args.layer} / K={K} / d={d}")
    print(f"  mIoU = {miou*100:.2f}%")
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
        "miou": miou,
        "seg_acc": seg_acc,
        "ckpt_path": ckpt_path,
        "npz_path": str(npz_path_for_codec(ckpt_path)),
    }
    if rate_info:
        results["rate_info"] = rate_info
    if rans_per_image_bpfp is not None:
        results["rans_per_image_bpfp"] = float(rans_per_image_bpfp)

    out_dir = os.path.join(PROJECT_ROOT, "results", "orfc_2446", args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_stem = Path(ckpt_path).stem
    out_path = os.path.join(out_dir, f"seg_{ckpt_stem}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Soft-PQ segmentation test (load codec, DINOv2 only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", type=str, required=True, choices=["dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, default="", help="Explicit checkpoint path (overrides auto-resolve)")
    parser.add_argument("--norm_mode", choices=ORFC_NORM_MODES, default=None)

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
        "--seg_feat_root", type=str, default=os.path.join(PROJECT_ROOT, "features", "orfc", "voc2012_100")
    )
    parser.add_argument("--voc_root", type=str, default=os.path.join(PROJECT_ROOT, "data", "VOC2012_sel100"))
    parser.add_argument("--seg_image_list", type=str, default=str(SHARED_OFFLINE_DIR / "cfg" / "voc2012_val_100.txt"))

    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()

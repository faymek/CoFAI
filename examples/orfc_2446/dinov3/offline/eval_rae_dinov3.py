#!/usr/bin/env python3
"""DINOv3-L + RAE reconstruction eval (mirrors orfc_2446/eval_rae.py).

Flow (k=1 / slot=24):
  Image -> encode(blocks[:24]) -> SoftPQ compress -> non-affine LN(patches)
        -> GeneralDecoder -> PSNR / MS-SSIM / LPIPS

Usage:
  poetry run python examples/orfc_2446/dinov3/offline/eval_rae_dinov3.py \\
    --orfc_ckpt weights/orfc_2446_dinov3/dinov3_large_256px/slot24_K256_..._wReg.pt \\
    --compress_reg --max_images 100 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_COFAI_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.backbone.timm import Dinov3TimmBackbone
from cofai.entropy_models.orfc_model import batch_inv_normalize_gpu, batch_normalize_gpu
from cofai.entropy_models.soft_pq import codec_forward, load_codec, load_codec_meta
from cofai.heads.rae import GeneralDecoder

from lib.orfc_codec import NORM_MODE_CHOICES, resolve_norm_settings

# RAEv2 DINOv3 decoder is trained to emit near-[0,1] pixels directly
# (see third_party/RAEv2 stage1: decode() = unpatchify only, no ImageNet denorm).
# Using ImageNet mean/std here washes recon to ~13 dB; identity keeps ~19 dB bypass.
PIXEL_MEAN = (0.0, 0.0, 0.0)
PIXEL_STD = (1.0, 1.0, 1.0)

DECODER_CONFIGS = {
    "ViTXL": {
        "cfg": {
            "decoder_hidden_size": 1152,
            "decoder_intermediate_size": 4096,
            "decoder_num_attention_heads": 16,
            "decoder_num_hidden_layers": 28,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "num_channels": 3,
            "qkv_bias": True,
        },
        "default_path": "weights/RAE/decoders/dinov3/large/decoder.pt",
        "encoder_hidden_size": 1024,
    },
}


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def load_imagenet_sel500(data_root):
    img_dir = Path(data_root) / "img"
    list_file = Path(data_root) / "labels.txt"
    with open(list_file) as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return [p for n in names if (p := img_dir / n).exists()]


def load_pathname_list(pathname_list, imagenet_root):
    paths = []
    with open(pathname_list) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            class_id, img_name = parts[0], parts[1]
            p = Path(imagenet_root) / class_id / f"{img_name}.JPEG"
            if p.exists():
                paths.append(p)
    return paths


def compress_decompress(
    features, codec, device, norm_mode="per_image", n_prefix=0, prefix_bypass=False,
):
    X_norm, mu, std = batch_normalize_gpu(
        features, mode=norm_mode, n_prefix=n_prefix,
    )
    codec.eval()
    with torch.no_grad():
        Y_hat, _ = codec_forward(
            X_norm, codec, n_prefix=n_prefix, prefix_bypass=prefix_bypass,
        )
    return batch_inv_normalize_gpu(Y_hat, mu, std)


def extract_rae_patches(h, num_prefix, norm_eps):
    patches = h[:, num_prefix:, :]
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True, unbiased=False)
    return (patches - mean) / torch.sqrt(var + norm_eps)


def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2, reduction="none").mean(dim=[1, 2, 3])
    return -10 * torch.log10(mse + 1e-10)


def compute_ms_ssim(img1, img2):
    try:
        from pytorch_msssim import ms_ssim
        return ms_ssim(img1, img2, data_range=1.0, size_average=False)
    except ImportError:
        from torchmetrics.functional.image import (
            multiscale_structural_similarity_index_measure as ms_ssim_fn,
        )
        return ms_ssim_fn(img1, img2, data_range=1.0)


def compute_lpips(img1, img2, lpips_fn):
    return lpips_fn(img1 * 2 - 1, img2 * 2 - 1).squeeze()


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    img_size = args.img_size
    patch_size = args.patch_size
    token_per_dim = img_size // patch_size
    num_patches = token_per_dim ** 2
    token_res = (token_per_dim, token_per_dim)

    print(f"\n{'=' * 70}")
    print(f"  DINOv3-L + RAE eval ({img_size}px)  slot={args.slot}")
    print(f"  compress_reg={args.compress_reg}  device={device}")
    print(f"{'=' * 70}")

    print("\n[1/4] Loading DINOv3-L backbone...")
    ckpt = None if args.backbone_ckpt in ("none", "", None) else args.backbone_ckpt
    backbone = Dinov3TimmBackbone(
        model_size="large",
        img_size=img_size,
        patch_size=patch_size,
        dynamic_size=False,
        slot=args.slot,
        n_last_blocks=1,
        pretrained=False,
        ckpt_path=ckpt,
        device=str(device),
    )
    backbone.model.to(device).eval()
    dino = backbone.model
    num_prefix = int(dino.num_prefix_tokens)
    num_reg = num_prefix - 1
    norm_eps = float(dino.norm.eps)
    print(f"  prefix={num_prefix} (reg={num_reg})  D={dino.embed_dim}  patches={num_patches}")

    dec_info = DECODER_CONFIGS[args.decoder_type]
    decoder_path = (
        args.decoder_path if args.decoder_path != "auto"
        else str(_COFAI_ROOT / dec_info["default_path"])
    )
    print(f"[2/4] Loading RAE decoder from {decoder_path}")
    decoder_patch_size = 16
    decoder_image_size = decoder_patch_size * token_per_dim
    decoder = GeneralDecoder(
        decoder_config=dec_info["cfg"],
        patch_size=decoder_patch_size,
        image_size=decoder_image_size,
        pretrained_path=decoder_path,
        encoder_img_size=img_size,
        encoder_patch_size=patch_size,
        encoder_hidden_size=dec_info["encoder_hidden_size"],
        encoder_mean=list(PIXEL_MEAN),
        encoder_std=list(PIXEL_STD),
        device=device,
    )
    decoder.eval()
    print("  pixel postprocess: identity (RAEv2 DINOv3; no ImageNet denorm)")

    print(f"[3/4] Loading codec {args.orfc_ckpt}")
    ckpt_meta = load_codec_meta(args.orfc_ckpt)
    norm_mode, n_prefix, norm_src = resolve_norm_settings(
        norm_mode=args.norm_mode,
        n_prefix=args.n_prefix,
        ckpt_path=args.orfc_ckpt,
        ckpt_meta=ckpt_meta,
        default_n_prefix=num_prefix,
        fallback_norm="per_image",
    )
    codec = load_codec(args.orfc_ckpt, device=str(device))
    codec.eval()
    K = codec.pq.K
    G = codec.pq.G
    bpfp = float(np.log2(K) * G)
    train_tokens = ckpt_meta.get("train_tokens", "all")
    prefix_bypass = train_tokens == "patch"
    if prefix_bypass:
        # SoftPQ only on patches; CLS+reg identity
        num_codec_tokens = num_patches
        # Full sequence needed for bypass path
        use_full_seq = True
    elif args.compress_reg or ckpt_meta.get("compress_reg", False):
        num_codec_tokens = 1 + num_reg + num_patches
        use_full_seq = True
    else:
        num_codec_tokens = 1 + num_patches
        use_full_seq = False
    bits_per_image = bpfp * num_codec_tokens
    print(f"  K={K} G={G}  {bpfp:.1f} bits/token x {num_codec_tokens} = {bits_per_image:.0f} bits/img")
    print(f"  train_tokens={train_tokens}  prefix_bypass={prefix_bypass}")
    print(f"  norm_mode={norm_mode}  n_prefix={n_prefix}  (source={norm_src})")

    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    except ImportError:
        print("  [warn] lpips unavailable")

    print("[4/4] Dataset...")
    if args.imagenet_sel500_root and args.imagenet_sel500_root != "none":
        img_files = load_imagenet_sel500(args.imagenet_sel500_root)
        print(f"  ImageNet_val_sel500: {len(img_files)}")
    else:
        img_files = load_pathname_list(args.pathname_list, args.imagenet_root)
        print(f"  pathname_list: {len(img_files)}")
    if args.max_images and args.max_images < len(img_files):
        img_files = img_files[: args.max_images]
    print(f"  Evaluating {len(img_files)} images")

    tfm = build_transform(img_size)
    gt_tfm = transforms.Compose([
        transforms.Resize(decoder_image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(decoder_image_size),
        transforms.ToTensor(),
    ])

    metrics = {k: [] for k in (
        "psnr_bypass", "psnr_codec", "msssim_bypass", "msssim_codec",
        "lpips_bypass", "lpips_codec",
    )}

    t0 = time.time()
    for img_path in tqdm(img_files, desc="eval_rae"):
        img = Image.open(img_path).convert("RGB")
        x_raw = tfm(img).unsqueeze(0).to(device)
        gt = gt_tfm(img).unsqueeze(0).to(device)

        h = backbone.encode(x_raw)

        # Bypass (no codec)
        feat_bp = extract_rae_patches(h, num_prefix, norm_eps)
        rec_bp = decoder.predict(feat_bp, token_res=token_res, token_format="patch")

        # Codec path
        if use_full_seq:
            h_hat = compress_decompress(
                h, codec, device,
                norm_mode=norm_mode, n_prefix=n_prefix,
                prefix_bypass=prefix_bypass,
            )
        else:
            cls = h[:, 0:1, :]
            reg = h[:, 1:1 + num_reg, :]
            patches = h[:, 1 + num_reg:, :]
            cp = torch.cat([cls, patches], dim=1)
            # Without compress_reg, training drops reg from the SoftPQ stream;
            # split_reg_cls_patch is not meaningful — fall back to per_image /
            # split_cls_patch over cls+patch only.
            cp_n_prefix = 1 if norm_mode.startswith("split_") else 0
            cp_mode = "split_cls_patch" if norm_mode.startswith("split_") else norm_mode
            cp_hat = compress_decompress(
                cp, codec, device, norm_mode=cp_mode, n_prefix=cp_n_prefix,
            )
            h_hat = torch.cat([cp_hat[:, 0:1, :], reg, cp_hat[:, 1:, :]], dim=1)

        feat_cd = extract_rae_patches(h_hat, num_prefix, norm_eps)
        rec_cd = decoder.predict(feat_cd, token_res=token_res, token_format="patch")

        metrics["psnr_bypass"].append(compute_psnr(rec_bp, gt).item())
        metrics["psnr_codec"].append(compute_psnr(rec_cd, gt).item())
        metrics["msssim_bypass"].append(compute_ms_ssim(rec_bp, gt).item())
        metrics["msssim_codec"].append(compute_ms_ssim(rec_cd, gt).item())
        if lpips_fn is not None:
            metrics["lpips_bypass"].append(compute_lpips(rec_bp, gt, lpips_fn).item())
            metrics["lpips_codec"].append(compute_lpips(rec_cd, gt, lpips_fn).item())

    elapsed = time.time() - t0
    avg = {k: float(np.mean(v)) if v else None for k, v in metrics.items()}

    print(f"\n{'=' * 70}")
    print(f"  Results: slot={args.slot} K={K}  n={len(img_files)}  {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<12} {'Bypass':<14} {'Codec':<14} {'Delta':<12}")
    print(f"  {'-' * 52}")
    for name, key_bp, key_cd in (
        ("PSNR (dB)", "psnr_bypass", "psnr_codec"),
        ("MS-SSIM", "msssim_bypass", "msssim_codec"),
        ("LPIPS", "lpips_bypass", "lpips_codec"),
    ):
        if avg[key_bp] is None:
            continue
        d = avg[key_cd] - avg[key_bp]
        print(f"  {name:<12} {avg[key_bp]:<14.4f} {avg[key_cd]:<14.4f} {d:+.4f}")
    print(f"  Bitrate: {bpfp:.1f} b/tok x {num_codec_tokens} = {bits_per_image:.0f} bits/img")
    print(f"{'=' * 70}\n")

    out = {
        "ckpt": args.orfc_ckpt,
        "slot": args.slot,
        "K": K,
        "G": G,
        "compress_reg": args.compress_reg or ckpt_meta.get("compress_reg", False),
        "train_tokens": train_tokens,
        "prefix_bypass": prefix_bypass,
        "norm_mode": norm_mode,
        "n_prefix": n_prefix,
        "norm_source": norm_src,
        "n_images": len(img_files),
        "metrics": avg,
        "bpfp_theoretical": bpfp,
        "bits_per_image": bits_per_image,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.orfc_ckpt).stem
    out_path = out_dir / f"rae_recon_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orfc_ckpt", type=str, required=True)
    p.add_argument("--backbone_ckpt", type=str,
                   default="weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.pth")
    p.add_argument("--decoder_type", type=str, default="ViTXL", choices=list(DECODER_CONFIGS))
    p.add_argument("--decoder_path", type=str, default="auto")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--slot", type=int, default=24)
    p.add_argument("--compress_reg", action="store_true")
    p.add_argument(
        "--norm_mode", choices=NORM_MODE_CHOICES, default=None,
        help="Feature norm before SoftPQ (default: auto from ckpt meta/filename)",
    )
    p.add_argument("--n_prefix", type=int, default=0)
    p.add_argument("--max_images", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--imagenet_sel500_root", type=str,
                   default=str(_COFAI_ROOT / "data" / "ImageNet_val_sel500"))
    p.add_argument("--pathname_list", type=str, default="none")
    p.add_argument("--imagenet_root", type=str,
                   default="/data4/workspace/zlt/featcodec/data/imagenet/images/val")
    p.add_argument("--out_dir", type=str,
                   default="examples/orfc_2446/dinov3/results")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

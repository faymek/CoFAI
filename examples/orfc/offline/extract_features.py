#!/usr/bin/env python3
"""
ORFC Feature Extraction — extract DINOv2 intermediate block features for offline training/eval.

Supports two modes:
  - cls: Single-image classification features (Resize+CenterCrop 224) → [1+N, D] per image
  - seg: Slide-window segmentation features (crop 512, stride 341) → [num_slides, 1+N, D] per image

Usage:
    # Classification features (ImageNet, wnid subdirectory layout)
    python extract_features.py cls \
        --backbone dinov2_vitl14 \
        --blocks 5,10,15,20 \
        --image_root /path/to/imagenet/val \
        --image_list images_5000.txt \
        --out_root $PROJECT_ROOT/features/orfc/train/dinov2_vitl14

    # Segmentation features (VOC2012, flat JPEGImages)
    python extract_features.py seg \
        --backbone dinov2_vitl14 \
        --blocks 5,10,15,20 \
        --voc_root $PROJECT_ROOT/data/VOC2012_sel100 \
        --image_list $PROJECT_ROOT/data/VOC2012_sel100/voc2012_val_100.txt \
        --out_root $PROJECT_ROOT/features/orfc/voc2012_100/dinov2_vitl14
"""

import os
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import sys
sys.path.insert(0, os.path.join(PROJECT_ROOT, "cofai", "backbone"))

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", category=UserWarning)

from dinov2.models import vision_transformer as vits

BACKBONE_REGISTRY = {
    "dinov2_vitl14": {
        "vit_fn": "vit_large",
        "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0, block_chunks=0),
        "embed_dim": 1024,
        "pretrain": "dinov2_vitl14_pretrain.pth",
    },
    "dinov2_vitg14": {
        "vit_fn": "vit_giant2",
        "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0,
                           block_chunks=0, ffn_layer="swiglufused"),
        "embed_dim": 1536,
        "pretrain": "dinov2_vitg14_pretrain.pth",
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_SIZE = (512, 512)
STRIDE = (341, 341)
PATCH_SIZE = 14


def load_backbone(backbone_name, weights_root, device):
    reg = BACKBONE_REGISTRY[backbone_name]
    builder = getattr(vits, reg["vit_fn"])
    model = builder(**reg["vit_kwargs"])
    ckpt_path = os.path.join(weights_root, reg["pretrain"])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    print(f"  Backbone loaded: {ckpt_path} ({reg['vit_fn']}, dim={reg['embed_dim']})")
    return model, reg


def build_cls_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_seg_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CenterPadding(nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def forward(self, x):
        import itertools
        pads = list(itertools.chain.from_iterable(
            self._get_pad(m) for m in x.shape[:1:-1]
        ))
        return F.pad(x, pads)

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        return pad_size // 2, pad_size - pad_size // 2


def get_slide_crops(h_img, w_img, crop_size, stride):
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    crops = []
    for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
        for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crops.append((y1, x1, y2, x2))
    return crops


# ========================= Classification Extraction =========================

@torch.no_grad()
def extract_cls(args):
    device = args.device
    backbone, reg = load_backbone(args.backbone, args.weights_root, device)
    tfm = build_cls_transform()
    blocks = [int(x) for x in args.blocks.split(",")]
    out_root = args.out_root

    for b in blocks:
        os.makedirs(os.path.join(out_root, f"blk{b:02d}"), exist_ok=True)

    with open(args.image_list, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    if args.list_format == "wnid_base":
        pairs = [l.split() for l in lines]
    else:
        pairs = [(None, l) for l in lines]

    print(f"  Extracting {len(pairs)} images, blocks={blocks}")
    if args.max_images and args.max_images < len(pairs):
        pairs = pairs[:args.max_images]
        print(f"  Limited to {args.max_images} images")

    t0 = time.time()
    n = 0
    for wnid, base in tqdm(pairs, desc="Extract cls"):
        if wnid:
            img_path = os.path.join(args.image_root, wnid, base + ".JPEG")
        else:
            fname = base if "." in base else base + ".JPEG"
            img_path = os.path.join(args.image_root, fname)

        if not os.path.isfile(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        feats = backbone.get_intermediate_layers(
            x, n=blocks, reshape=False, norm=False, return_class_token=True
        )

        for i, b in enumerate(blocks):
            patch_tokens, cls_token = feats[i]
            full_seq = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
            arr = full_seq.squeeze(0).cpu().numpy().astype(np.float32)
            save_path = os.path.join(out_root, f"blk{b:02d}", f"{base}.npy")
            np.save(save_path, arr)

        n += 1

    print(f"  Done. Extracted {n} images in {time.time()-t0:.1f}s")


# ========================= Segmentation Extraction =========================

@torch.no_grad()
def extract_seg(args):
    device = args.device
    backbone, reg = load_backbone(args.backbone, args.weights_root, device)
    blocks = [int(x) for x in args.blocks.split(",")]
    out_root = args.out_root
    voc_root = args.voc_root
    center_pad = CenterPadding(PATCH_SIZE)

    for b in blocks:
        os.makedirs(os.path.join(out_root, f"blk{b:02d}"), exist_ok=True)

    with open(args.image_list, 'r') as f:
        image_ids = [l.strip() for l in f if l.strip()]

    if args.max_images and args.max_images < len(image_ids):
        image_ids = image_ids[:args.max_images]

    seg_tfm = build_seg_transform()
    resize_short = 512

    print(f"  Extracting {len(image_ids)} VOC images, blocks={blocks}")
    t0 = time.time()
    n = 0

    for name in tqdm(image_ids, desc="Extract seg"):
        img_path = os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
        if not os.path.isfile(img_path):
            continue

        img = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img.size
        scale = resize_short / min(h_orig, w_orig)
        new_h, new_w = int(h_orig * scale + 0.5), int(w_orig * scale + 0.5)
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)

        img_tensor = seg_tfm(img_resized).unsqueeze(0).to(device)
        h_img, w_img = img_tensor.shape[2], img_tensor.shape[3]

        crops = get_slide_crops(h_img, w_img, CROP_SIZE, STRIDE)

        crop_imgs = []
        for y1, x1, y2, x2 in crops:
            crop_img = img_tensor[:, :, y1:y2, x1:x2]
            crop_padded = center_pad(crop_img)
            crop_imgs.append(crop_padded)

        for b in blocks:
            feature_list = []
            for ci in range(0, len(crop_imgs), args.batch_size):
                batch = torch.cat(crop_imgs[ci:ci+args.batch_size], dim=0)
                feats = backbone.get_intermediate_layers(
                    batch, n=[b], reshape=False, norm=False, return_class_token=True
                )
                patch_tokens, cls_token = feats[0]
                full_seq = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
                feature_list.append(full_seq.cpu())

            all_feats = torch.cat(feature_list, dim=0).numpy().astype(np.float32)
            save_path = os.path.join(out_root, f"blk{b:02d}", f"{name}.npy")
            np.save(save_path, all_feats)

        n += 1

    print(f"  Done. Extracted {n} images in {time.time()-t0:.1f}s")


# ========================= CLI =========================

def main():
    parser = argparse.ArgumentParser(
        description="ORFC Feature Extraction (DINOv2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # cls sub-command
    p_cls = sub.add_parser("cls", help="Extract classification features (ImageNet)")
    p_cls.add_argument("--backbone", required=True, choices=list(BACKBONE_REGISTRY.keys()))
    p_cls.add_argument("--blocks", default="5,10,15,20",
                       help="Comma-separated block indices")
    p_cls.add_argument("--image_root", required=True,
                       help="ImageNet image root (e.g. data/imagenet/images/val)")
    p_cls.add_argument("--image_list", required=True,
                       help="Image list file")
    p_cls.add_argument("--list_format", default="wnid_base",
                       choices=["wnid_base", "flat"],
                       help="List format: 'wnid_base' for '<wnid> <base>' or 'flat' for filenames")
    p_cls.add_argument("--out_root", required=True,
                       help="Output root (e.g. features/orfc/train/dinov2_vitl14)")
    p_cls.add_argument("--weights_root",
                       default=os.path.join(PROJECT_ROOT, "weights", "pretrained"))
    p_cls.add_argument("--max_images", type=int, default=None)
    p_cls.add_argument("--device", default="cuda")

    # seg sub-command
    p_seg = sub.add_parser("seg", help="Extract segmentation features (VOC2012 slide-window)")
    p_seg.add_argument("--backbone", required=True, choices=list(BACKBONE_REGISTRY.keys()))
    p_seg.add_argument("--blocks", default="5,10,15,20",
                       help="Comma-separated block indices")
    p_seg.add_argument("--voc_root",
                       default=os.path.join(PROJECT_ROOT, "data", "VOC2012_sel100"))
    p_seg.add_argument("--image_list",
                       default=os.path.join(PROJECT_ROOT, "data", "VOC2012_sel100",
                                            "voc2012_val_100.txt"))
    p_seg.add_argument("--out_root", required=True,
                       help="Output root (e.g. features/orfc/voc2012_100/dinov2_vitl14)")
    p_seg.add_argument("--weights_root",
                       default=os.path.join(PROJECT_ROOT, "weights", "pretrained"))
    p_seg.add_argument("--batch_size", type=int, default=4)
    p_seg.add_argument("--max_images", type=int, default=None)
    p_seg.add_argument("--device", default="cuda")

    args = parser.parse_args()
    if args.mode == "cls":
        extract_cls(args)
    elif args.mode == "seg":
        extract_seg(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract DINOv3-L/16 ImageNet features for PQFC training (slot24 / blk23).

Saves per-image ``.npy`` with full encode output ``[T, D]`` =
``[CLS + 4 reg + patches, 1024]``. Default ``img_size=224`` → ``T=201``.

Usage (from CoFAI root)::

  poetry run python examples/pqfc/offline/extract_train_features.py \\
      --img_size 224 --slot 24 --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_COFAI_ROOT = Path(__file__).resolve().parents[3]
_FEATCODEC_ROOT = _COFAI_ROOT.parent
if str(_COFAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_COFAI_ROOT))

from cofai.backbone.timm import Dinov3TimmBackbone


def _default_pathname_list() -> Path:
    candidates = [
        _FEATCODEC_ROOT / "utils" / "imagenet_selected_pathname5000.txt",
        _COFAI_ROOT / "examples" / "pqfc" / "cfg" / "imagenet_selected_pathname5000.txt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _default_imagenet_root() -> Path:
    candidates = [
        _FEATCODEC_ROOT / "data" / "imagenet" / "images" / "val",
        _COFAI_ROOT / "data" / "ImageNet_val",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def _resolve_backbone(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = _COFAI_ROOT / p
    if p.is_file():
        return str(p)
    # allow .pth <-> .safetensors fallback
    alt = p.with_suffix(".safetensors" if p.suffix == ".pth" else ".pth")
    if alt.is_file():
        return str(alt)
    return str(p)


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def load_image_paths(pathname_list: str, imagenet_root: str):
    paths, names = [], []
    with open(pathname_list, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_id, img_name = parts[0], parts[1]
                img_path = Path(imagenet_root) / class_id / f"{img_name}.JPEG"
                if img_path.exists():
                    paths.append(img_path)
                    names.append(img_name)
    return paths, names


def main():
    p = argparse.ArgumentParser(description="Extract DINOv3 ImageNet features for PQFC train")
    p.add_argument("--pathname_list", type=str, default=str(_default_pathname_list()))
    p.add_argument("--imagenet_root", type=str, default=str(_default_imagenet_root()))
    p.add_argument(
        "--backbone_ckpt",
        default="weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.pth",
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--slot", type=int, default=24, help="Encode blocks[:slot]; 24 = Layer23")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_images", type=int, default=0, help="0 = all")
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Default: features/train/dinov3_vitl16/blk23 under PROJECT_ROOT",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = p.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(_COFAI_ROOT))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        layer = f"blk{args.slot - 1:02d}" if args.slot > 0 else f"slot{args.slot:02d}"
        args.output_dir = str(
            _COFAI_ROOT / "features" / "train" / "dinov3_vitl16" / layer
        )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _resolve_backbone(args.backbone_ckpt)
    print(f"\n{'=' * 70}")
    print(f"  DINOv3-L/16 train feature extract  slot={args.slot}  {args.img_size}px")
    print(f"  backbone: {ckpt}")
    print(f"  imagenet: {args.imagenet_root}")
    print(f"  list:     {args.pathname_list}")
    print(f"  output:   {out_dir}")
    print(f"{'=' * 70}")

    if not Path(args.pathname_list).is_file():
        raise SystemExit(f"pathname_list not found: {args.pathname_list}")
    if not Path(args.imagenet_root).is_dir():
        raise SystemExit(
            f"imagenet_root not found: {args.imagenet_root}\n"
            "Prepare ImageNet val (class folders) and pass --imagenet_root."
        )
    if not Path(ckpt).is_file():
        raise SystemExit(
            f"backbone not found: {ckpt}\n"
            "Run: bash examples/pqfc/scripts/prepare_dinov3.sh"
        )

    print("\n[1/3] Loading backbone...")
    backbone = Dinov3TimmBackbone(
        model_size="large",
        img_size=args.img_size,
        patch_size=args.patch_size,
        dynamic_size=False,
        slot=args.slot,
        n_last_blocks=1,
        pretrained=False,
        ckpt_path=ckpt,
        device=str(device),
    )
    backbone.model.to(device).eval()
    print(f"  prefix={backbone.model.num_prefix_tokens}  embed={backbone.model.embed_dim}")

    print("[2/3] Loading image list...")
    img_files, img_names = load_image_paths(args.pathname_list, args.imagenet_root)
    if args.max_images > 0:
        img_files = img_files[: args.max_images]
        img_names = img_names[: args.max_images]
    print(f"  {len(img_files)} images found on disk")
    if not img_files:
        raise SystemExit("No images matched pathname_list under imagenet_root")

    tfm = build_transform(args.img_size)
    saved = skipped = 0
    print("[3/3] Extracting...")
    with torch.inference_mode():
        for i in tqdm(range(0, len(img_files), args.batch_size), desc="Extract"):
            batch_files = img_files[i:i + args.batch_size]
            batch_names = img_names[i:i + args.batch_size]
            todo_idx = []
            for j, name in enumerate(batch_names):
                out_path = out_dir / f"{name}.npy"
                if args.skip_existing and out_path.exists():
                    skipped += 1
                else:
                    todo_idx.append(j)
            if not todo_idx:
                continue
            imgs = [tfm(Image.open(batch_files[j]).convert("RGB")) for j in todo_idx]
            x = torch.stack(imgs).to(device)
            h = backbone.encode(x)
            for k, j in enumerate(todo_idx):
                np.save(out_dir / f"{batch_names[j]}.npy", h[k].cpu().numpy().astype(np.float32))
                saved += 1
            del x, h

    sample = next(out_dir.glob("*.npy"), None)
    if sample is None:
        raise SystemExit("No features written")
    arr = np.load(sample)
    expect_t = 1 + 4 + (args.img_size // args.patch_size) ** 2
    print(f"\n  Saved={saved}  skipped={skipped}  total_npy={len(list(out_dir.glob('*.npy')))}")
    print(f"  Sample {sample.name}: shape={arr.shape} (expect T={expect_t}, D=1024)")
    if arr.shape[0] != expect_t or arr.shape[1] != 1024:
        raise SystemExit(f"Unexpected shape {arr.shape}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract DINOv3-L/16 ImageNet features for PQFC training (slot24 / blk23)."""

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

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_SOURCE_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.backbone.timm import Dinov3TimmBackbone
from lib.config_utils import resolve_project_root as _resolve_pr


def _default_pathname_list(project_root: Path) -> Path:
    featcodec_root = project_root.parent
    candidates = [
        featcodec_root / "utils" / "imagenet_selected_pathname5000.txt",
        project_root / "data" / "imagenet_selected_pathname5000.txt",
        _SOURCE_ROOT / "data" / "imagenet_selected_pathname5000.txt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _default_imagenet_root(project_root: Path) -> Path:
    candidates = [
        project_root.parent / "data" / "imagenet" / "images" / "val",
        project_root / "data" / "imagenet" / "images" / "val",
        project_root / "data" / "ImageNet_val",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _resolve_backbone(path: str, project_root: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = project_root / p
    if p.is_file():
        return str(p)
    alt = p.with_suffix(".safetensors" if p.suffix == ".pth" else ".pth")
    if alt.is_file():
        return str(alt)
    return str(p)


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )


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
    project_root = _resolve_pr()
    p = argparse.ArgumentParser(description="Extract DINOv3 ImageNet features for PQFC train")
    p.add_argument("--pathname_list", type=str, default=str(_default_pathname_list(project_root)))
    p.add_argument("--imagenet_root", type=str, default=str(_default_imagenet_root(project_root)))
    p.add_argument(
        "--backbone_ckpt",
        default="weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors",
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--slot", type=int, default=24, help="Encode blocks[:slot]; 24 = Layer23")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_images", type=int, default=0, help="0 = all")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = p.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(project_root))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        layer = f"blk{args.slot - 1:02d}" if args.slot > 0 else f"slot{args.slot:02d}"
        args.output_dir = str(project_root / "features" / "train" / "dinov3_vitl16" / layer)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _resolve_backbone(args.backbone_ckpt, project_root)
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
            batch_files = img_files[i : i + args.batch_size]
            batch_names = img_names[i : i + args.batch_size]
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
                np.save(
                    out_dir / f"{batch_names[j]}.npy",
                    h[k].cpu().numpy().astype(np.float32),
                )
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

#!/usr/bin/env python3
"""Extract DINOv3-L/16 features at a given slot (default slot24 / blk23).

Saves per-image .npy with full encode output [T, D] = [cls+reg+patches, 1024].

Modes:
  - ImageNet (default): square Resize+CenterCrop to ``img_size``.
    For img_size=256, patch16 → T=261 (5 prefix + 256 patches).
  - ADE ``--dataset ade``: keep native HxW, pad to multiple of ``pad_multiple``.
    For 683×512 + pad16 → 688×512 → T=5+43×32=1381.

Usage:
  # ImageNet 5k @ 256px
  PYTHONPATH="$SOURCE_ROOT" poetry -C "$PROJECT_ROOT" run python \\
    "$SOURCE_ROOT/examples/orfc_2446/dinov3/offline/extract_features_dinov3.py" \\
    --img_size 256 --slot 24 --device cuda:0 \\
    --pathname_list data/imagenet_selected_pathname5000.txt

  # ADE 683×512 5k (pad to 16)
  PYTHONPATH="$SOURCE_ROOT" poetry -C "$PROJECT_ROOT" run python \\
    "$SOURCE_ROOT/examples/orfc_2446/dinov3/offline/extract_features_dinov3.py" \\
    --dataset ade --slot 24 --device cuda:0 \\
    --max_images 5000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_COFAI_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_COFAI_ROOT / ".env")
if str(_COFAI_ROOT) not in sys.path:
    sys.path.insert(0, str(_COFAI_ROOT))

from cofai.backbone.timm import Dinov3TimmBackbone
from cofai.transforms import PadToMultiple, ToTensor


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )


def build_ade_transform(pad_multiple: int):
    """Use the same image transforms as the formal DINOv3 evaluation plans."""
    return transforms.Compose(
        [
            PadToMultiple(pad_multiple, keys=["img"]),
            ToTensor(keys=["img"]),
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


def load_ade_paths(pathname_list: str | None, ade_root: str):
    """One filename (or stem) per line → (path, stem) for .npy naming."""
    root = Path(ade_root)
    if pathname_list is None:
        paths = sorted(path for path in root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return paths, [path.stem for path in paths]

    paths, names = [], []
    with open(pathname_list, "r") as f:
        for line in f:
            tok = line.strip().split()[0] if line.strip() else ""
            if not tok:
                continue
            name = Path(tok).name
            stem = Path(name).stem
            img_path = root / name
            if not img_path.exists():
                # allow stem-only lines
                for ext in (".jpg", ".JPG", ".png", ".JPEG"):
                    cand = root / f"{stem}{ext}"
                    if cand.exists():
                        img_path = cand
                        break
            if img_path.exists():
                paths.append(img_path)
                names.append(stem)
            else:
                print(f"[warn] missing: {root / name}")
    return paths, names


def main():
    p = argparse.ArgumentParser(description="Extract DINOv3-L features (ImageNet / ADE)")
    p.add_argument(
        "--dataset",
        choices=("imagenet", "ade"),
        default="imagenet",
        help="Image source / list format",
    )
    p.add_argument(
        "--pathname_list",
        default=None,
        help="Optional image list; required for ImageNet, optional for ADE directory scan",
    )
    p.add_argument(
        "--imagenet_root",
        default=None,
    )
    p.add_argument(
        "--ade_root",
        default=None,
    )
    p.add_argument(
        "--backbone_ckpt",
        default="weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors",
    )
    p.add_argument("--img_size", type=int, default=256, help="Square crop size (imagenet mode)")
    p.add_argument("--pad_multiple", type=int, default=16, help="Pad H/W to this multiple (ade mode)")
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--slot", type=int, default=24, help="Encode blocks[:slot]; 24 = last layer")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_images", type=int, default=0, help="0 = all")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = p.parse_args()

    project_root_value = os.environ.get("PROJECT_ROOT")
    if not project_root_value:
        raise RuntimeError(f"PROJECT_ROOT is required; set it or add it to {_COFAI_ROOT / '.env'}")
    project_root = Path(project_root_value).expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"PROJECT_ROOT is not a directory: {project_root}")

    if args.pathname_list:
        pathname_list = Path(args.pathname_list).expanduser()
        if not pathname_list.is_absolute():
            pathname_list = project_root / pathname_list
        args.pathname_list = str(pathname_list)
    elif args.dataset == "imagenet":
        raise ValueError("--pathname_list is required for dataset=imagenet")
    args.imagenet_root = args.imagenet_root or str(project_root / "data/imagenet/images/val")
    args.ade_root = args.ade_root or str(project_root / "data/ADEChallengeData2016/images/training")
    for attr in ("imagenet_root", "ade_root"):
        path = Path(getattr(args, attr)).expanduser()
        setattr(args, attr, str(path if path.is_absolute() else project_root / path))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.output_dir is None:
        if args.dataset == "ade":
            args.output_dir = str(project_root / "features/orfc_2446/dinov3/ade")
        else:
            args.output_dir = str(
                project_root / f"features/orfc_2446/dinov3/imagenet_{args.img_size}px/slot{args.slot:02d}"
            )
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    use_ade = args.dataset == "ade"
    print(f"\n{'=' * 70}")
    print(f"  DINOv3-L/16 feature extract  slot={args.slot}  dataset={args.dataset}")
    if use_ade:
        print(f"  Native size + pad_multiple={args.pad_multiple}")
    else:
        print(f"  Square {args.img_size}px")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 70}")

    print("\n[1/3] Loading backbone...")
    backbone = Dinov3TimmBackbone(
        model_size="large",
        img_size=args.img_size if not use_ade else 512,
        patch_size=args.patch_size,
        dynamic_size=use_ade,
        slot=args.slot,
        n_last_blocks=1,
        pretrained=False,
        ckpt_path=str(
            Path(args.backbone_ckpt) if Path(args.backbone_ckpt).is_absolute() else project_root / args.backbone_ckpt
        ),
        device=str(device),
    )
    backbone.model.to(device).eval()
    print(f"  prefix={backbone.model.num_prefix_tokens}  embed={backbone.model.embed_dim}")

    print("[2/3] Loading image list...")
    if use_ade:
        img_files, img_names = load_ade_paths(args.pathname_list, args.ade_root)
    else:
        img_files, img_names = load_image_paths(args.pathname_list, args.imagenet_root)
    if args.max_images > 0:
        img_files = img_files[: args.max_images]
        img_names = img_names[: args.max_images]
    print(f"  {len(img_files)} images")

    tfm = build_ade_transform(args.pad_multiple) if use_ade else build_transform(args.img_size)
    saved = skipped = 0
    sample_hw = None
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

            if use_ade:
                imgs = []
                for j in todo_idx:
                    image = Image.open(batch_files[j]).convert("RGB")
                    sample = {"img": np.asarray(image, dtype=np.float32) / 255.0}
                    imgs.append(tfm(sample)["img"])
                # ADE training images share a spatial size after center padding.
                x = torch.stack(imgs).to(device)
                if sample_hw is None:
                    sample_hw = (int(x.shape[2]), int(x.shape[3]))
            else:
                imgs = [tfm(Image.open(batch_files[j]).convert("RGB")) for j in todo_idx]
                x = torch.stack(imgs).to(device)

            h = backbone.encode(x)  # [B, T, D]
            for k, j in enumerate(todo_idx):
                out_path = out_dir / f"{batch_names[j]}.npy"
                np.save(out_path, h[k].cpu().numpy().astype(np.float32))
                saved += 1
            del x, h

    # Sanity check
    sample = next(out_dir.glob("ADE_train_*.npy"), None) or next(out_dir.glob("*.npy"))
    arr = np.load(sample)
    if use_ade:
        if sample_hw is None:
            # all skipped — infer from first image
            im0 = Image.open(img_files[0])
            w0, h0 = im0.size
            ph = (args.pad_multiple - h0 % args.pad_multiple) % args.pad_multiple
            pw = (args.pad_multiple - w0 % args.pad_multiple) % args.pad_multiple
            sample_hw = (h0 + ph, w0 + pw)
        th, tw = sample_hw[0] // args.patch_size, sample_hw[1] // args.patch_size
        expect_t = 1 + 4 + th * tw
        print(f"\n  Saved={saved}  skipped={skipped}  total_npy={len(list(out_dir.glob('*.npy')))}")
        print(f"  Padded input HxW={sample_hw} → token_hw=({th},{tw})")
        print(f"  Sample {sample.name}: shape={arr.shape} (expect T={expect_t}, D=1024)")
    else:
        expect_t = 1 + 4 + (args.img_size // args.patch_size) ** 2
        print(f"\n  Saved={saved}  skipped={skipped}  total_npy={len(list(out_dir.glob('*.npy')))}")
        print(f"  Sample {sample.name}: shape={arr.shape} (expect T={expect_t}, D=1024)")
    if arr.shape[0] != expect_t or arr.shape[1] != 1024:
        raise SystemExit(f"Unexpected shape {arr.shape}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

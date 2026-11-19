#!/usr/bin/env python3
"""Extract DINOv3 ViT-L/16 slot24 features for the PQFCv2 20k training set.

Images keep their native resolution and are center-padded to a multiple of 16,
matching the ADE20K and NYUv2 evaluation transforms. One ``.npy`` of shape
``[tokens, 1024]`` is written per image. Images that share a padded size are
batched together; different lengths are never stacked.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from cofai.backbone.timm import Dinov3TimmBackbone
from cofai.transforms import PadToMultiple, ToTensor
from examples.pqfcv2.lists import iter_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ade-list", type=Path, required=True)
    parser.add_argument("--nyu-list", type=Path, required=True)
    parser.add_argument("--ade-root", type=Path, required=True)
    parser.add_argument("--nyu-root", type=Path, required=True)
    parser.add_argument("--ade-out", type=Path, required=True)
    parser.add_argument("--nyu-out", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        type=Path,
        default=Path("weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors"),
    )
    parser.add_argument("--slot", type=int, default=24)
    parser.add_argument("--pad-multiple", type=int, default=16)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve(path: Path, root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def padded_hw(path: Path, multiple: int) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    return height + pad_h, width + pad_w


def load_batch(paths: list[Path], pad: PadToMultiple, to_tensor: ToTensor) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        sample = to_tensor(pad({"img": array}))
        tensors.append(sample["img"])
    return torch.stack(tensors, dim=0)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    root = args.project_root.expanduser().resolve()
    os.environ.setdefault("PROJECT_ROOT", str(root))
    samples = iter_sources(
        ade_list=resolve(args.ade_list, root),
        ade_root=resolve(args.ade_root, root),
        nyu_list=resolve(args.nyu_list, root),
        nyu_root=resolve(args.nyu_root, root),
        ade_out=resolve(args.ade_out, root),
        nyu_out=resolve(args.nyu_out, root),
    )
    if args.offset or args.limit:
        end = None if args.limit <= 0 else args.offset + args.limit
        samples = samples[args.offset : end]
    samples = [item for index, item in enumerate(samples) if index % args.num_shards == args.shard_index]
    for _, image_path, _ in samples:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    for out_dir in {path.parent for _, _, path in samples}:
        out_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    skipped = 0
    for dataset, image_path, feature_path in samples:
        if args.skip_existing and feature_path.is_file():
            skipped += 1
            continue
        height, width = padded_hw(image_path, args.pad_multiple)
        pending.append((dataset, image_path, feature_path, height, width))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    backbone_path = resolve(args.backbone, root)
    backbone = Dinov3TimmBackbone(
        model_size="large",
        img_size=512,
        patch_size=16,
        dynamic_size=True,
        slot=args.slot,
        n_last_blocks=1,
        cast_dtype="float32",
        pretrained=False,
        ckpt_path=str(backbone_path),
        device=str(device),
    )
    backbone.to(device).eval()
    prefix_tokens = int(backbone.model.num_prefix_tokens)
    if prefix_tokens != 5 or int(backbone.model.embed_dim) != 1024:
        raise RuntimeError(
            f"Expected 5 prefix tokens and 1024 channels, got {prefix_tokens}, {backbone.model.embed_dim}"
        )
    pad = PadToMultiple(args.pad_multiple, keys=["img"])
    to_tensor = ToTensor(keys=["img"])
    groups: dict[tuple[int, int], list[tuple[Path, Path]]] = defaultdict(list)
    for _, image_path, feature_path, height, width in pending:
        groups[(height, width)].append((image_path, feature_path))

    print(
        f"shard {args.shard_index}/{args.num_shards}  pending={len(pending)}  skipped={skipped}  device={device}",
        flush=True,
    )
    saved = 0
    with torch.inference_mode():
        for (height, width), items in tqdm(groups.items(), desc=f"gpu{args.shard_index}"):
            tokens = prefix_tokens + (height // 16) * (width // 16)
            batch_limit = max(1, args.max_batch_tokens // max(tokens, 1))
            for start in range(0, len(items), batch_limit):
                batch = items[start : start + batch_limit]
                images = load_batch([path for path, _ in batch], pad, to_tensor).to(device)
                if tuple(images.shape[-2:]) != (height, width):
                    raise RuntimeError(f"Padded size {tuple(images.shape[-2:])} != {(height, width)}")
                features = backbone.encode(images).float().cpu().numpy()
                for index, (_, feature_path) in enumerate(batch):
                    array = np.asarray(features[index], dtype=np.float32)
                    if array.shape != (tokens, 1024):
                        raise RuntimeError(f"{feature_path} has shape {array.shape}, expected {(tokens, 1024)}")
                    np.save(feature_path, array)
                    saved += 1
                del images, features
    print(f"shard {args.shard_index} saved={saved} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()

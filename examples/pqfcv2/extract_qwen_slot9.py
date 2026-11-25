#!/usr/bin/env python3
"""Extract Qwen3-VL vision-encoder slot9 features for the PQFCv2 20k set.

CTC encodes ``visual.blocks[:9]``, the output of layer 8 (blk08), which is the
first deepstack index. These tokens are patch tokens only. Images are resized
by the Qwen processor (``min_pixels=768*28*28``, ``max_pixels=1536*28*28``).
One ``.npy`` of shape ``[tokens, 1152]`` is written per image.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from cofai.backbone.qwen3vl import Qwen3VLBackbone
from examples.pqfcv2.lists import iter_sources

MIN_PIXELS = 768 * 28 * 28
MAX_PIXELS = 1536 * 28 * 28
CHANNELS = 1152
SLOT = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ade-list", type=Path, required=True)
    parser.add_argument("--nyu-list", type=Path, required=True)
    parser.add_argument("--ade-root", type=Path, required=True)
    parser.add_argument("--nyu-root", type=Path, required=True)
    parser.add_argument("--ade-out", type=Path, required=True)
    parser.add_argument("--nyu-out", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--grid-out", type=Path, default=None)
    parser.add_argument("--dataset", choices=("all", "ade", "nyu"), default="all")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model ID or local path to Qwen3-VL-8B-Instruct.",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
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
    if args.dataset != "all":
        samples = [item for item in samples if item[0] == args.dataset]
    if args.offset or args.limit:
        end = None if args.limit <= 0 else args.offset + args.limit
        samples = samples[args.offset : end]
    samples = [item for index, item in enumerate(samples) if index % args.num_shards == args.shard_index]
    for _, image_path, _ in samples:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
    for out_dir in {path.parent for _, _, path in samples}:
        out_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        (image_path, feature_path)
        for _, image_path, feature_path in samples
        if not (args.skip_existing and feature_path.is_file())
    ]
    skipped = len(samples) - len(pending)
    device = args.device if torch.cuda.is_available() else "cpu"
    model_arg = os.path.expandvars(args.model)
    model_candidate = Path(model_arg).expanduser()
    if model_candidate.is_absolute():
        model_path = str(model_candidate)
    elif (root / model_candidate).exists():
        model_path = str((root / model_candidate).resolve())
    else:
        model_path = model_arg
    print(f"loading Qwen3-VL vision encoder on {device} from {model_path}", flush=True)
    backbone = Qwen3VLBackbone(
        model_path=model_path,
        device=device,
        local_files_only=bool(args.local_files_only),
        dtype=torch.bfloat16,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        slot=SLOT,
    )
    # Work around PyTorch 2.9.0+cu128 Conv3d bf16 NCDHW fallback to slow_conv_dilated3d;
    # channels_last_3d keeps native Conv3d while dispatching to cuDNN.
    backbone.model.model.visual.patch_embed.proj.to(memory_format=torch.channels_last_3d)
    if backbone.slot != SLOT or int(backbone.first_ds_idx) != SLOT - 1:
        raise RuntimeError(
            f"Expected slot {SLOT} at deepstack layer {SLOT - 1}, "
            f"got slot {backbone.slot}, deepstack {backbone.first_ds_idx}"
        )
    print(
        f"dataset={args.dataset} shard={args.shard_index}/{args.num_shards}  pending={len(pending)}  "
        f"skipped={skipped}  slot={backbone.slot}  blk{backbone.first_ds_idx:02d}",
        flush=True,
    )
    feature_root = resolve(args.feature_root, root) if args.feature_root is not None else None
    grids: dict[str, list[int]] = {}
    saved = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for image_path, feature_path in tqdm(pending, desc=f"{args.dataset}-s{args.shard_index}"):
            tokens, token_res = backbone.encode_image(str(image_path))
            array = tokens.detach().float().cpu().numpy()
            if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != CHANNELS:
                raise RuntimeError(f"{image_path} has shape {array.shape}, expected (1, L, {CHANNELS})")
            length = int(array.shape[1])
            grid_h, grid_w = int(token_res[0]), int(token_res[1])
            if length != grid_h * grid_w:
                raise RuntimeError(
                    f"{image_path} token count {length} != grid {grid_h}x{grid_w}"
                )
            feature = np.ascontiguousarray(array[0], dtype=np.float32)
            np.save(feature_path, feature)
            if args.grid_out is not None:
                key = (
                    feature_path.resolve().relative_to(feature_root).as_posix()
                    if feature_root is not None
                    else feature_path.name
                )
                grids[key] = [grid_h, grid_w]
            saved += 1
    if args.grid_out is not None:
        grid_out = resolve(args.grid_out, root)
        grid_out.parent.mkdir(parents=True, exist_ok=True)
        grid_out.write_text(
            json.dumps(
                {
                    "slot": SLOT,
                    "channels": CHANNELS,
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                    "features": grids,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"grids={grid_out} ({len(grids)} entries)", flush=True)
    elapsed = time.perf_counter() - started
    rate = saved / elapsed if elapsed > 0 else 0.0
    print(
        f"saved={saved} skipped={skipped} elapsed={elapsed:.1f}s rate={rate:.2f} img/s",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assign each PQFCv2 feature to the padded patch grid it was extracted on.

Extraction pads an image to a multiple of 16. Training batches group features
by the resulting token count so same-length sequences stack into one forward.
Height and width stay on each sample because RoPE depends on the grid, even
when two grids contain the same number of tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from examples.pqfcv2.lists import iter_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--ade-list", type=Path, required=True)
    parser.add_argument("--nyu-list", type=Path, required=True)
    parser.add_argument("--ade-root", type=Path, required=True)
    parser.add_argument("--nyu-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--ade-out", type=Path, required=True)
    parser.add_argument("--nyu-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--pad-multiple", type=int, default=16)
    parser.add_argument("--n-prefix", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def resolve(path: Path, root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def padded_hw(path: Path, multiple: int) -> tuple[int, int]:
    """Match ``extract_features.padded_hw``: pad each side up to ``multiple``."""
    with Image.open(path) as image:
        width, height = image.size
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    return height + pad_h, width + pad_w


def npy_tokens(path: Path) -> int:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or int(array.shape[1]) != 1024:
        raise ValueError(f"Expected [tokens, 1024] in {path}, got {array.shape}")
    return int(array.shape[0])


def feature_key(feature_path: Path, feature_root: Path) -> str:
    return feature_path.resolve().relative_to(feature_root.resolve()).as_posix()


def load_grids(path: Path) -> dict[str, tuple[int, int]]:
    """Return ``feature key -> (patch_h, patch_w)``."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    grids = {}
    for key, grid in payload["features"].items():
        patch_h, patch_w = int(grid[0]), int(grid[1])
        if patch_h < 1 or patch_w < 1:
            raise ValueError(f"Invalid grid for {key}: {grid}")
        grids[key] = (patch_h, patch_w)
    return grids


def token_count(grid: tuple[int, int], n_prefix: int) -> int:
    patch_h, patch_w = grid
    return n_prefix + patch_h * patch_w


def make_token_batches(
    feature_paths: list[Path],
    feature_root: Path,
    grids: dict[str, tuple[int, int]],
    n_prefix: int,
    batch_size: int,
    seed: int,
    max_batch_tokens: int | None = None,
) -> list[list[Path]]:
    """Shuffle within each padded token count, cut batches, then shuffle batches.

    Every returned batch has one token count, so the features stack into
    ``[batch, tokens, channels]`` with no padding. The last batch of a count
    may be shorter than ``batch_size``. ``max_batch_tokens`` lowers the width
    when ``batch_size`` sequences of that length do not fit in GPU memory.
    Samples in one batch may still have different ``(patch_h, patch_w)``; the
    caller applies RoPE per sample.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    buckets: dict[int, list[Path]] = defaultdict(list)
    for path in feature_paths:
        key = feature_key(path, feature_root)
        if key not in grids:
            raise KeyError(f"{key} is missing from the grid index")
        buckets[token_count(grids[key], n_prefix)].append(path)
    rng = np.random.RandomState(seed)
    batches: list[list[Path]] = []
    for count in sorted(buckets):
        width = batch_size
        if max_batch_tokens is not None:
            width = max(1, min(batch_size, int(max_batch_tokens) // count))
        paths = buckets[count]
        order = rng.permutation(len(paths))
        ordered = [paths[int(index)] for index in order]
        for start in range(0, len(ordered), width):
            batches.append(ordered[start : start + width])
    batch_order = rng.permutation(len(batches))
    return [batches[int(index)] for index in batch_order]


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    os.environ.setdefault("PROJECT_ROOT", str(root))
    feature_root = resolve(args.feature_root, root)
    samples = iter_sources(
        ade_list=resolve(args.ade_list, root),
        ade_root=resolve(args.ade_root, root),
        nyu_list=resolve(args.nyu_list, root),
        nyu_root=resolve(args.nyu_root, root),
        ade_out=resolve(args.ade_out, root),
        nyu_out=resolve(args.nyu_out, root),
    )
    missing = [str(feature) for _, image, feature in samples if not image.is_file() or not feature.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} images or features are missing, first entries:\n{preview}")

    def measure(item: tuple[str, Path, Path]) -> tuple[str, int, int, int]:
        _dataset, image_path, feature_path = item
        height, width = padded_hw(image_path, args.pad_multiple)
        if height % args.pad_multiple or width % args.pad_multiple:
            raise RuntimeError(f"{image_path} padded to {height}x{width}")
        return feature_key(feature_path, feature_root), height // args.pad_multiple, width // args.pad_multiple, npy_tokens(feature_path)

    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        measured = list(pool.map(measure, samples))

    mismatches = []
    features: dict[str, list[int]] = {}
    for key, patch_h, patch_w, tokens in measured:
        expected = args.n_prefix + patch_h * patch_w
        if tokens != expected:
            mismatches.append(f"{key}: npy tokens {tokens}, grid {patch_h}x{patch_w} expects {expected}")
        features[key] = [patch_h, patch_w]
    if mismatches:
        preview = "\n".join(mismatches[:10])
        raise RuntimeError(f"{len(mismatches)} features do not match their padded grid:\n{preview}")
    if len(features) != len(samples):
        raise RuntimeError(f"Grid index has {len(features)} entries for {len(samples)} samples")

    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pad_multiple": args.pad_multiple,
        "patch_size": args.pad_multiple,
        "n_prefix": args.n_prefix,
        "features": features,
    }
    output.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    split = json.loads(resolve(args.split_path, root).read_text(encoding="utf-8"))
    train_counts: dict[int, int] = defaultdict(int)
    train_grids: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for key in split["train"]:
        grid = tuple(features[key])
        tokens = token_count(grid, args.n_prefix)
        train_counts[tokens] += 1
        train_grids[tokens].add(grid)
    ranked = sorted(train_counts.items(), key=lambda item: -item[1])
    print(f"wrote {output}")
    print(
        f"features={len(features)} grids={len({tuple(grid) for grid in features.values()})} "
        f"train_token_buckets={len(train_counts)}"
    )
    print("train top token counts:")
    for tokens, count in ranked[:12]:
        print(f"  tokens={tokens}  n={count}  grids={len(train_grids[tokens])}")
    singletons = sum(1 for count in train_counts.values() if count == 1)
    print(f"train singleton token counts={singletons}")


if __name__ == "__main__":
    main()

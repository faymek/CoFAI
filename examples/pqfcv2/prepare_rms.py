#!/usr/bin/env python3
"""Build the VQ-UFC RMS artifact for the combined ADE+NYU PQFCv2 split.

Five prefix positions each keep a channel RMS. All patch positions share one
channel RMS. Statistics use only the training files. ``n_val`` files are held
out with the M2488 seed and are not included in the RMS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from cofai.utils.utils_vqufc import compute_rms, save_rms, split_hash
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
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-prefix-tokens", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def resolve(path: Path, root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    feature_root = resolve(args.feature_root, root)
    samples = iter_sources(
        ade_list=resolve(args.ade_list, root),
        ade_root=resolve(args.ade_root, root),
        nyu_list=resolve(args.nyu_list, root),
        nyu_root=resolve(args.nyu_root, root),
        ade_out=resolve(args.ade_out, root),
        nyu_out=resolve(args.nyu_out, root),
    )
    features = []
    missing = []
    for _, _, feature_path in samples:
        if feature_path.is_file():
            features.append(feature_path.resolve())
        else:
            missing.append(str(feature_path))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} features are missing, first entries:\n{preview}")
    if args.n_val <= 0 or args.n_val >= len(features):
        raise ValueError(f"n_val={args.n_val} is outside 1..{len(features) - 1}")

    order = np.random.RandomState(args.seed).permutation(len(features))
    val_files = [features[int(index)] for index in order[: args.n_val]]
    train_files = [features[int(index)] for index in order[args.n_val :]]
    prefix_rms, patch_rms = compute_rms(train_files, args.num_prefix_tokens, args.eps)
    metadata = {
        "format_version": 1,
        "normalization_mode": "prefix_position_patch_channel_rms",
        "feature_root": str(feature_root),
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "split_sha256": split_hash(train_files, feature_root),
        "seed": args.seed,
        "n_val": args.n_val,
        "num_prefix_tokens": args.num_prefix_tokens,
        "prefix_rms_shape": list(prefix_rms.shape),
        "patch_rms_shape": list(patch_rms.shape),
        "eps": args.eps,
        "datasets": ["ade_1w", "nyu_1w"],
    }
    output = resolve(args.output, root)
    save_rms(output, prefix_rms, patch_rms, metadata)
    split_path = resolve(args.split_path, root)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split = {
        "seed": args.seed,
        "n_val": args.n_val,
        "feature_root": str(feature_root),
        "train": [str(path.relative_to(feature_root)) for path in train_files],
        "validation": [str(path.relative_to(feature_root)) for path in val_files],
    }
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"train={len(train_files)} validation={len(val_files)}")
    print(f"prefix_rms={tuple(prefix_rms.shape)} patch_rms={tuple(patch_rms.shape)}")
    print(f"rms={output}")
    print(f"split={split_path}")


if __name__ == "__main__":
    main()

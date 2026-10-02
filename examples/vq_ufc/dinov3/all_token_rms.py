#!/usr/bin/env python3
"""Build and load RMS statistics for all-token VQ-UFC training."""

from __future__ import annotations

import argparse
from pathlib import Path

from cofai.utils.utils_vqufc import compute_rms, save_rms, split_features, split_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-prefix-tokens", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    root, train_files, val_files = split_features(args.train_features, args.seed, args.val_ratio, args.max_files)
    prefix_rms, patch_rms = compute_rms(train_files, args.num_prefix_tokens, args.eps)
    metadata = {
        "format_version": 1,
        "normalization_mode": "prefix_position_patch_channel_rms",
        "feature_root": str(root),
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "split_sha256": split_hash(train_files, root),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "num_prefix_tokens": args.num_prefix_tokens,
        "prefix_rms_shape": list(prefix_rms.shape),
        "patch_rms_shape": list(patch_rms.shape),
        "eps": args.eps,
    }
    save_rms(args.output, prefix_rms, patch_rms, metadata)
    print(f"saved={Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()

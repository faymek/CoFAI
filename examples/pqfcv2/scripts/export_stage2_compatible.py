#!/usr/bin/env python3
"""Package a K2048e8 stage-1 snapshot and its RMS for the stage-2 loader."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Selected stage-1 NPZ snapshot")
    parser.add_argument("--rms", type=Path, required=True, help="Matching prefix/patch RMS NPZ")
    parser.add_argument("--output", type=Path, required=True, help="New output NPZ; must not exist")
    parser.add_argument("--assignment-lambda", type=float, default=10000.0)
    args = parser.parse_args()
    if not np.isfinite(args.assignment_lambda) or args.assignment_lambda <= 0:
        parser.error("--assignment-lambda must be finite and positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    with np.load(args.source, allow_pickle=False) as snapshot:
        arrays = {key: np.array(snapshot[key], copy=True) for key in snapshot.files}
    with np.load(args.rms, allow_pickle=False) as rms:
        for key in ("prefix_position_channel_rms", "patch_channel_rms"):
            value = np.array(rms[key], copy=True)
            if key in arrays and not np.array_equal(arrays[key], value):
                raise ValueError(f"Snapshot and supplied RMS disagree for {key}")
            arrays[key] = value

    shapes = {
        "R": (1024, 1024), "codebooks": (128, 2048, 8), "pmf": (128, 2048),
        "prefix_position_channel_rms": (5, 1024), "patch_channel_rms": (1024,),
    }
    for key, expected in shapes.items():
        if key not in arrays or arrays[key].shape != expected or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} must be a finite array with shape {expected}")
    for key in ("prefix_position_channel_rms", "patch_channel_rms"):
        if np.any(arrays[key] <= 0):
            raise ValueError(f"{key} must be positive")
    if np.any(arrays["pmf"] < 0) or np.any(arrays["pmf"].sum(axis=1) <= 0):
        raise ValueError("pmf must be nonnegative with positive mass for every group")
    arrays["lmbda"] = np.asarray(args.assignment_lambda, dtype=np.float64)
    arrays["export_source_sha256"] = np.asarray(sha256(args.source))
    arrays["export_rms_sha256"] = np.asarray(sha256(args.rms))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=".pqfcv2-export-", delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez(handle, **arrays)
        # Hard-link publication fails if the destination appeared concurrently.
        os.link(temp_path, args.output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    print(f"saved {args.output}")
    print(f"sha256={sha256(args.output)}")
    print(f"source_sha256={arrays['export_source_sha256'].item()}")
    print(f"rms_sha256={arrays['export_rms_sha256'].item()}")


if __name__ == "__main__":
    main()

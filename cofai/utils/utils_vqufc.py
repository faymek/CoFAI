"""Feature split and RMS helpers shared by VQ-UFC training tools."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np


def split_features(
    feature_root: str | Path,
    seed: int = 42,
    val_ratio: float = 0.05,
    max_files: int = 0,
) -> tuple[Path, list[Path], list[Path]]:
    root = Path(feature_root).expanduser().resolve()
    files = sorted(root.rglob("*.npy"))
    if max_files > 0:
        files = files[:max_files]
    if len(files) < 2:
        raise ValueError(f"At least two .npy files are required under {root}")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one")

    random.Random(seed).shuffle(files)
    val_count = min(max(1, round(len(files) * val_ratio)), len(files) - 1)
    return root, files[val_count:], files[:val_count]


def split_hash(files: list[Path], root: Path) -> str:
    paths = "\n".join(str(path.relative_to(root)) for path in files)
    return hashlib.sha256(paths.encode()).hexdigest()


def compute_rms(
    files: list[Path],
    num_prefix_tokens: int = 5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    prefix_sumsq = None
    patch_sumsq = None
    patch_count = 0

    for path in files:
        feature = np.load(path, mmap_mode="r", allow_pickle=False)
        if feature.ndim != 2 or feature.shape[0] <= num_prefix_tokens:
            raise ValueError(f"Expected [prefix+patch, channels] in {path}, got {feature.shape}")
        prefix = np.asarray(feature[:num_prefix_tokens], dtype=np.float32)
        patch = np.asarray(feature[num_prefix_tokens:], dtype=np.float32)
        if prefix_sumsq is None:
            prefix_sumsq = np.zeros(prefix.shape, dtype=np.float64)
            patch_sumsq = np.zeros(patch.shape[1], dtype=np.float64)
        if prefix.shape != prefix_sumsq.shape or patch.shape[1] != patch_sumsq.size:
            raise ValueError(f"Inconsistent feature shape in {path}: {feature.shape}")
        prefix64 = prefix.astype(np.float64)
        patch64 = patch.astype(np.float64)
        prefix_sumsq += prefix64 * prefix64
        patch_sumsq += np.sum(patch64 * patch64, axis=0)
        patch_count += patch.shape[0]

    prefix_rms = np.sqrt(prefix_sumsq / len(files) + eps).astype(np.float32)
    patch_rms = np.sqrt(patch_sumsq / patch_count + eps).astype(np.float32)
    return prefix_rms, patch_rms


def save_rms(
    output: str | Path,
    prefix_rms: np.ndarray,
    patch_rms: np.ndarray,
    metadata: dict,
) -> None:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prefix_position_channel_rms=prefix_rms,
        patch_channel_rms=patch_rms,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_rms(
    path: str | Path,
    train_files: list[Path],
    root: Path,
    num_prefix_tokens: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as data:
        prefix_rms = np.asarray(data["prefix_position_channel_rms"], dtype=np.float32)
        patch_rms = np.asarray(data["patch_channel_rms"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"].item()))

    expected = (num_prefix_tokens, patch_rms.size)
    if metadata.get("split_sha256") != split_hash(train_files, root):
        raise ValueError("RMS artifact was computed from a different training split")
    if prefix_rms.shape != expected or patch_rms.ndim != 1:
        raise ValueError(f"Invalid RMS shapes: prefix={prefix_rms.shape}, patch={patch_rms.shape}")
    if not np.isfinite(prefix_rms).all() or not np.isfinite(patch_rms).all():
        raise ValueError("RMS artifact contains non-finite values")
    if np.any(prefix_rms <= 0) or np.any(patch_rms <= 0):
        raise ValueError("RMS artifact contains non-positive values")
    return prefix_rms, patch_rms, metadata

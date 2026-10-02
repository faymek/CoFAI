"""Encode/decode helpers for PQFC training (FeatureCodec) and replay (ORFC npz)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from cofai.latent_codecs.orfc_normalization import (
    ORFC_NORM_MODES,
    denormalize_orfc_features,
    normalize_orfc_features,
)
from examples.orfc_2446.offline.soft_pq import codec_forward

NORM_MODE_CHOICES = ORFC_NORM_MODES
SPLIT_NORM_MODES = frozenset({"split_cls_patch", "split_reg_cls_patch"})


def resolve_n_prefix(norm_mode: str, n_prefix: int, default: int = 5) -> int:
    if norm_mode in SPLIT_NORM_MODES and n_prefix == 0:
        return default
    return n_prefix


def resolve_norm_settings(
    *,
    norm_mode: str | None,
    n_prefix: int = 0,
    artifact_meta: dict | None = None,
    default_n_prefix: int = 5,
    fallback_norm: str = "per_image",
) -> tuple[str, int, str]:
    metadata = artifact_meta or {}
    if norm_mode:
        resolved_mode = str(norm_mode)
        source = "cli"
    elif metadata.get("norm_mode"):
        resolved_mode = str(metadata["norm_mode"])
        source = "artifact"
    else:
        resolved_mode = str(fallback_norm)
        source = "config"

    if resolved_mode not in NORM_MODE_CHOICES:
        raise ValueError(f"Unsupported normalization mode: {resolved_mode!r}")

    resolved_prefix = int(n_prefix)
    if resolved_prefix <= 0 and metadata.get("n_prefix") is not None:
        resolved_prefix = int(metadata["n_prefix"])
    if resolved_prefix <= 0:
        resolved_prefix = resolve_n_prefix(resolved_mode, 0, default_n_prefix)
    if resolved_prefix < 0:
        raise ValueError(f"n_prefix must be non-negative, got {resolved_prefix}")
    return resolved_mode, resolved_prefix, source


def norm_sideinfo_bits(norm_mode: str, n_prefix: int, n_tokens: int) -> float:
    del n_tokens
    if norm_mode in SPLIT_NORM_MODES and n_prefix > 0:
        return 4 * 16
    if norm_mode == "per_image":
        return 2 * 16
    return 0.0


def encode_decode_single(
    feat_td: np.ndarray,
    codec,
    norm_mode: str,
    device,
    n_prefix: int = 0,
) -> np.ndarray:
    """Encode/decode one [T, D] array (runtime ORFC npz or training FeatureCodec)."""
    codec.eval()
    X = torch.from_numpy(feat_td[np.newaxis]).float().to(device)
    with torch.no_grad():
        if hasattr(codec, "compress") and hasattr(codec, "orfc_weights_path"):
            if hasattr(codec, "norm_mode"):
                codec.norm_mode = norm_mode
            if hasattr(codec, "n_prefix"):
                codec.n_prefix = int(n_prefix)
            return codec(X)["h_hat"].squeeze(0).cpu().numpy()

        Y, mu, std = normalize_orfc_features(X, mode=norm_mode, n_prefix=n_prefix)
        y_hat, _ = codec_forward(Y, codec)
        x_hat = denormalize_orfc_features(
            y_hat, mu, std, mode=norm_mode, n_prefix=n_prefix
        )
    return x_hat.squeeze(0).cpu().numpy()


def encode_decode_batch(
    features: Sequence[np.ndarray],
    codec,
    norm_mode: str,
    device,
    n_prefix: int = 0,
    chunk_images: int | None = None,
) -> list[np.ndarray]:
    from examples.orfc_2446.offline.soft_pq import soft_pq_encode_decode

    del chunk_images
    return soft_pq_encode_decode(
        list(features),
        codec,
        norm_mode,
        device,
        n_prefix=n_prefix,
    )


def theoretical_bpfp(codec, embed_dim: int) -> float:
    pq = codec.pq
    bpt = pq.G * math.log2(pq.K)
    return bpt / embed_dim


def infer_norm_mode_from_name(path_or_name: str | None) -> str | None:
    if not path_or_name:
        return None
    name = Path(path_or_name).name
    for suf in (".pt", ".npz", ".pth", ".json"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    for mode in ("split_reg_cls_patch", "split_cls_patch", "per_token_ln"):
        if mode in name:
            return mode
    return None

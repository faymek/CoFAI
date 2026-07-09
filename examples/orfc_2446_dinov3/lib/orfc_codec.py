"""ORFC / SoftPQ encode-decode helpers for offline replay."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from cofai.entropy_models.orfc_model import batch_inv_normalize_gpu, batch_normalize_gpu
from cofai.entropy_models.soft_pq import codec_forward, soft_pq_encode_decode

# norm modes with per-image group stats (2 groups × μ/σ fp16 side info)
SPLIT_NORM_MODES = frozenset({"split_cls_patch", "split_reg_cls_patch"})

NORM_MODE_CHOICES = (
    "per_image",
    "split_cls_patch",
    "split_reg_cls_patch",
    "per_token_ln",
)


def resolve_n_prefix(norm_mode: str, n_prefix: int, default: int = 5) -> int:
    if norm_mode in SPLIT_NORM_MODES and n_prefix == 0:
        return default
    return n_prefix


def norm_sideinfo_bits(norm_mode: str, n_prefix: int, n_tokens: int) -> float:
    """Side-information bits per image for normalization stats (fp16)."""
    if norm_mode in SPLIT_NORM_MODES and n_prefix > 0:
        return 4 * 16
    if norm_mode == "per_image":
        return 2 * 16
    if norm_mode == "per_token_ln":
        return 0.0
    return 0.0


def encode_decode_single(
    feat_td: np.ndarray,
    codec,
    norm_mode: str,
    device,
    n_prefix: int = 0,
    prefix_bypass: bool = False,
) -> np.ndarray:
    """Encode/decode one [T, D] feature array."""
    codec.eval()
    X = torch.from_numpy(feat_td[np.newaxis]).float().to(device)
    with torch.no_grad():
        Y, mu, std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        y_hat, _ = codec_forward(
            Y, codec, n_prefix, prefix_bypass=prefix_bypass,
        )
        x_hat = batch_inv_normalize_gpu(y_hat, mu, std)
    return x_hat.squeeze(0).cpu().numpy()


def encode_decode_batch(
    features: Sequence[np.ndarray],
    codec,
    norm_mode: str,
    device,
    n_prefix: int = 0,
    chunk_images: int | None = None,
    prefix_bypass: bool = False,
) -> list[np.ndarray]:
    """Batch encode/decode when all features share the same shape."""
    return soft_pq_encode_decode(
        list(features),
        codec,
        norm_mode,
        device,
        chunk_images=chunk_images,
        n_prefix=n_prefix,
        prefix_bypass=prefix_bypass,
    )


def encode_decode_variable(
    features: Sequence[np.ndarray],
    codec,
    norm_mode: str,
    device,
    n_prefix: int = 0,
    prefix_bypass: bool = False,
) -> list[np.ndarray]:
    """Per-image encode/decode for variable token lengths."""
    return [
        encode_decode_single(
            f, codec, norm_mode, device,
            n_prefix=n_prefix, prefix_bypass=prefix_bypass,
        )
        for f in features
    ]


def theoretical_bpfp(codec, embed_dim: int) -> float:
    pq = codec.pq
    bpt = pq.G * math.log2(pq.K)
    return bpt / embed_dim

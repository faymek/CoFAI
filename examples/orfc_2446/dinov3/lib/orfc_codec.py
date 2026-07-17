"""ORFC / SoftPQ encode-decode helpers for offline replay."""

from __future__ import annotations

import math
from pathlib import Path
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


def infer_norm_mode_from_name(path_or_name: str | None) -> str | None:
    """Parse norm mode from ckpt filename tags (longest match first).

    NOTE: do not use ``Path(...).stem`` on names that already lack ``.pt`` —
    tags like ``cls1.0`` / ``lmbda0.0`` would be truncated at the last dot.
    """
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


def resolve_norm_settings(
    *,
    norm_mode: str | None,
    n_prefix: int = 0,
    ckpt_path: str | None = None,
    ckpt_meta: dict | None = None,
    default_n_prefix: int = 5,
    fallback_norm: str = "split_cls_patch",
) -> tuple[str, int, str]:
    """Resolve (norm_mode, n_prefix, source) for train/eval alignment.

    Priority: explicit CLI ``norm_mode`` > ckpt meta > filename tag > fallback.
    ``n_prefix`` from CLI (if >0) else ckpt meta else auto for split_* modes.
    """
    source = "fallback"
    resolved = norm_mode
    if resolved:
        source = "cli"
    else:
        meta = ckpt_meta or {}
        if meta.get("norm_mode"):
            resolved = str(meta["norm_mode"])
            source = "ckpt_meta"
        else:
            inferred = infer_norm_mode_from_name(ckpt_path)
            if inferred:
                resolved = inferred
                source = "ckpt_name"
            else:
                # Filenames omit the tag for the historical default per_image.
                if ckpt_path and "raetail" in Path(ckpt_path).stem:
                    resolved = "per_image"
                    source = "raetail_legacy"
                else:
                    resolved = fallback_norm
                    source = "fallback"

    meta = ckpt_meta or {}
    resolved_n = n_prefix
    if resolved_n <= 0:
        meta_n = meta.get("n_prefix")
        if meta_n is not None and int(meta_n) > 0:
            resolved_n = int(meta_n)
        else:
            resolved_n = resolve_n_prefix(resolved, 0, default_n_prefix)
    else:
        resolved_n = resolve_n_prefix(resolved, resolved_n, default_n_prefix)
    return resolved, resolved_n, source


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
) -> np.ndarray:
    """Encode/decode one [T, D] feature array.

    Supports SoftPQFeatureCodec (.npz) and legacy FeatureCodec (.pt).
    """
    codec.eval()
    X = torch.from_numpy(feat_td[np.newaxis]).float().to(device)
    with torch.no_grad():
        # SoftPQFeatureCodec: R/codebooks/pmf from .npz
        if hasattr(codec, "R") and hasattr(codec, "_encode_decode"):
            if hasattr(codec, "norm_mode"):
                codec.norm_mode = norm_mode
            if hasattr(codec, "n_prefix"):
                codec.n_prefix = int(n_prefix)
            x_hat = codec(X)["h_hat"]
            return x_hat.squeeze(0).cpu().numpy()

        Y, mu, std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        y_hat, _ = codec_forward(Y, codec)
        x_hat = batch_inv_normalize_gpu(y_hat, mu, std)
    return x_hat.squeeze(0).cpu().numpy()


def encode_decode_batch(
    features: Sequence[np.ndarray],
    codec,
    norm_mode: str,
    device,
    n_prefix: int = 0,
    chunk_images: int | None = None,
) -> list[np.ndarray]:
    """Batch encode/decode when all features share the same shape."""
    return soft_pq_encode_decode(
        list(features),
        codec,
        norm_mode,
        device,
        chunk_images=chunk_images,
        n_prefix=n_prefix,
    )


def encode_decode_variable(
    features: Sequence[np.ndarray],
    codec,
    norm_mode: str,
    device,
    n_prefix: int = 0,
) -> list[np.ndarray]:
    """Per-image encode/decode for variable token lengths."""
    return [
        encode_decode_single(
            f, codec, norm_mode, device,
            n_prefix=n_prefix,
        )
        for f in features
    ]


def theoretical_bpfp(codec, embed_dim: int) -> float:
    pq = codec.pq
    bpt = pq.G * math.log2(pq.K)
    return bpt / embed_dim

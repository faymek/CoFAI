"""Small offline helpers around the shared ORFC latent codec."""

from __future__ import annotations

import numpy as np
import torch

from cofai.latent_codecs.orfc_normalization import ORFC_NORM_MODES

NORM_MODE_CHOICES = ORFC_NORM_MODES
_SPLIT_NORM_MODES = {"split_cls_patch", "split_reg_cls_patch"}


def resolve_norm_settings(
    *,
    norm_mode: str | None,
    n_prefix: int = 0,
    artifact_meta: dict | None = None,
    default_n_prefix: int = 5,
    fallback_norm: str = "split_cls_patch",
) -> tuple[str, int, str]:
    """Resolve normalization from an explicit option or artifact metadata.

    Checkpoint filename inference and legacy checkpoint fallbacks are
    intentionally unsupported: released ``.npz`` artifacts carry their own
    normalization metadata.
    """
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
    if resolved_prefix <= 0 and resolved_mode in _SPLIT_NORM_MODES:
        resolved_prefix = int(default_n_prefix)
    if resolved_prefix < 0:
        raise ValueError(f"n_prefix must be non-negative, got {resolved_prefix}")
    return resolved_mode, resolved_prefix, source


def encode_decode_single(
    feature: np.ndarray,
    codec,
    device,
) -> np.ndarray:
    """Reconstruct one ``[tokens, channels]`` array with the runtime codec."""
    tokens = torch.from_numpy(feature).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        reconstruction = codec(tokens)["h_hat"]
    return reconstruction.squeeze(0).cpu().numpy()

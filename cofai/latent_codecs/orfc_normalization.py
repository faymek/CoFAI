"""Normalization shared by ORFC artifact generation and runtime codecs."""

from __future__ import annotations

import torch


__all__ = [
    "ORFC_NORM_MODES",
    "normalize_orfc_features",
    "denormalize_orfc_features",
]


ORFC_NORM_MODES = (
    "per_image",
    "per_token_ln",
    "split_cls_patch",
    "split_reg_cls_patch",
)


def _group_stats(
    tokens: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = tokens.mean(dim=(1, 2))
    variance = (tokens - mean[:, None, None]).square().mean(dim=(1, 2))
    return mean, (variance + eps).sqrt()


def _compute_normalization_stats(
    tokens: torch.Tensor,
    *,
    mode: str,
    n_prefix: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 3:
        raise ValueError(f"ORFC normalization expects [B, T, D], got {tuple(tokens.shape)}")
    if mode not in ORFC_NORM_MODES:
        raise ValueError(f"Unsupported ORFC normalization mode: {mode!r}")

    _, n_tokens, _ = tokens.shape
    if mode == "per_image":
        mean, std = _group_stats(tokens, eps)
        return mean[:, None], std[:, None]
    if mode == "per_token_ln":
        mean = tokens.mean(dim=2)
        variance = (tokens - mean[:, :, None]).square().mean(dim=2)
        return mean, (variance + eps).sqrt()
    if not 0 < n_prefix < n_tokens:
        raise ValueError(
            f"{mode} requires 0 < n_prefix < tokens; "
            f"got n_prefix={n_prefix}, tokens={n_tokens}"
        )
    if mode == "split_cls_patch":
        prefix_mean, prefix_std = _group_stats(tokens[:, :n_prefix], eps)
        patch_mean, patch_std = _group_stats(tokens[:, n_prefix:], eps)
        return (
            torch.stack((prefix_mean, patch_mean), dim=1),
            torch.stack((prefix_std, patch_std), dim=1),
        )
    if n_prefix < 2:
        raise ValueError("split_reg_cls_patch requires at least one register token")
    register_mean, register_std = _group_stats(tokens[:, 1:n_prefix], eps)
    cls_patch = torch.cat((tokens[:, :1], tokens[:, n_prefix:]), dim=1)
    cls_patch_mean, cls_patch_std = _group_stats(cls_patch, eps)
    return (
        torch.stack((register_mean, cls_patch_mean), dim=1),
        torch.stack((register_std, cls_patch_std), dim=1),
    )


def _expand_normalization_stats(
    values: torch.Tensor,
    *,
    mode: str,
    n_prefix: int,
    n_tokens: int,
) -> torch.Tensor:
    batch_size = values.shape[0]
    if mode == "per_image":
        return values[:, :1, None]
    if mode == "per_token_ln":
        return values.reshape(batch_size, n_tokens, 1)

    expanded = values.new_empty(batch_size, n_tokens, 1)
    if mode == "split_cls_patch":
        expanded[:, :n_prefix, 0] = values[:, :1]
        expanded[:, n_prefix:, 0] = values[:, 1:2]
        return expanded
    if mode != "split_reg_cls_patch":
        raise ValueError(f"Unsupported ORFC normalization mode: {mode!r}")

    expanded[:, 1:n_prefix, 0] = values[:, :1]
    expanded[:, :1, 0] = values[:, 1:2]
    expanded[:, n_prefix:, 0] = values[:, 1:2]
    return expanded


def normalize_orfc_features(
    tokens: torch.Tensor,
    *,
    mode: str = "per_image",
    n_prefix: int = 0,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize ORFC features and return compact mean/std side information."""
    mean, std = _compute_normalization_stats(
        tokens,
        mode=mode,
        n_prefix=n_prefix,
        eps=eps,
    )
    n_tokens = tokens.shape[1]
    mean_full = _expand_normalization_stats(
        mean,
        mode=mode,
        n_prefix=n_prefix,
        n_tokens=n_tokens,
    )
    std_full = _expand_normalization_stats(
        std,
        mode=mode,
        n_prefix=n_prefix,
        n_tokens=n_tokens,
    )
    return (tokens - mean_full) / std_full, mean, std


def denormalize_orfc_features(
    normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    mode: str = "per_image",
    n_prefix: int = 0,
) -> torch.Tensor:
    """Invert :func:`normalize_orfc_features`."""
    n_tokens = normalized.shape[1]
    mean_full = _expand_normalization_stats(
        mean,
        mode=mode,
        n_prefix=n_prefix,
        n_tokens=n_tokens,
    )
    std_full = _expand_normalization_stats(
        std,
        mode=mode,
        n_prefix=n_prefix,
        n_tokens=n_tokens,
    )
    return normalized * std_full + mean_full

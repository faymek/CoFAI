"""Deterministic sparse and fixed-length token restoration."""

from __future__ import annotations

import torch


def restore_sparse(tokens: torch.Tensor, indices: torch.Tensor, token_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact tokens and their original positions with shape checks."""
    if tokens.ndim != 3 or indices.ndim != 2:
        raise ValueError("tokens must be [B, K, C] and indices must be [B, K]")
    if tokens.shape[:2] != indices.shape:
        raise ValueError("tokens and indices must agree in their first two dimensions")
    if torch.any(indices < 0) or torch.any(indices >= token_count):
        raise ValueError("indices outside token_count")
    return tokens, indices


def restore_fixed_length(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    token_count: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Scatter compact patch tokens into a deterministic fixed-length tensor."""
    tokens, indices = restore_sparse(tokens, indices, token_count)
    restored = torch.full(
        (tokens.shape[0], token_count, tokens.shape[-1]),
        fill_value,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    restored.scatter_(1, indices.unsqueeze(-1).expand_as(tokens), tokens)
    return restored

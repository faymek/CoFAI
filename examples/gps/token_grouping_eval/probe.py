"""Small, deterministic probes used by the Token Grouping example."""

from __future__ import annotations

import torch


def operating_point_mask(token_count: int, rho: float) -> torch.Tensor:
    """Build the canonical raster operating-point mask used by the proxy sweep."""
    if token_count <= 0 or not 0.0 <= rho < 1.0:
        raise ValueError("token_count must be positive and rho must be in [0, 1)")
    kept = max(1, int(round(token_count * (1.0 - rho))))
    mask = torch.zeros(token_count, dtype=torch.bool)
    mask[:kept] = True
    return mask


def mask_summary(mask: torch.Tensor) -> dict[str, int | float]:
    """Return mask cardinality without attaching task-specific interpretation."""
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("mask must be a one-dimensional boolean tensor")
    token_count = int(mask.numel())
    kept_tokens = int(mask.sum().item())
    return {
        "token_count": token_count,
        "kept_tokens": kept_tokens,
        "keep_ratio": kept_tokens / token_count,
    }

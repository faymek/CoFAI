"""Shared token utilities for the GPS ReID backbone."""

from __future__ import annotations

import torch


def shuffle_unit(features, shift, group, begin=1):
    """Apply the unchanged JPM shift-and-shuffle operation."""
    batch_size = features.size(0)
    dim = features.size(-1)
    shifted = torch.cat(
        [
            features[:, begin - 1 + shift :],
            features[:, begin : begin - 1 + shift],
        ],
        dim=1,
    )
    if shifted.size(1) == 0:
        raise RuntimeError("shuffle_unit received no patch tokens")
    while shifted.size(1) % group != 0:
        shifted = torch.cat([shifted, shifted[:, -1:, :]], dim=1)
    shuffled = shifted.view(batch_size, group, -1, dim)
    shuffled = torch.transpose(shuffled, 1, 2).contiguous()
    return shuffled.view(batch_size, -1, dim)

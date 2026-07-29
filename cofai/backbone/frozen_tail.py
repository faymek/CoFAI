"""Frozen backbone tails for feature-distortion training.

A frozen tail maps an intermediate feature tensor through the remaining
backbone blocks and final normalization layer. Its parameters stay frozen and
in evaluation mode, while the regular ``forward`` path remains differentiable
with respect to the input features.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "FrozenTail",
    "Dinov3FrozenTail",
]


class FrozenTail(nn.Module):
    """Remaining token-sequence blocks followed by a normalization layer."""

    def __init__(self, blocks: Iterable[nn.Module], norm_layer: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenTail:
        """Keep the frozen teacher in evaluation mode."""
        return super().train(False)

    def _run_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        del block_index
        return block(x)

    def forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Run the frozen blocks without the final normalization."""
        for block_index, block in enumerate(self.blocks):
            x = self._run_block(block, x, block_index)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.forward_blocks(x))

    @torch.no_grad()
    def forward_nograd(self, x: torch.Tensor) -> torch.Tensor:
        """Run the teacher path without constructing an autograd graph."""
        return self(x)


class Dinov3FrozenTail(FrozenTail):
    """RoPE-aware frozen tail for a split DINOv3 timm backbone."""

    def __init__(
        self,
        model: nn.Module,
        split_layer_idx: int,
        rope: Any,
        attn_mask: torch.Tensor | None = None,
    ):
        n_blocks = len(model.blocks)
        if not -1 <= split_layer_idx < n_blocks:
            raise ValueError(f"split_layer_idx must be in [-1, {n_blocks - 1}], got {split_layer_idx}")

        self.tail_start = split_layer_idx + 1
        super().__init__(model.blocks[self.tail_start :], model.norm)
        self.rope = rope
        self.attn_mask = attn_mask
        self.rope_mixed = bool(getattr(model, "rope_mixed", False))

    def _run_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        global_index = self.tail_start + block_index
        rope = self.rope[global_index] if self.rope_mixed else self.rope
        if rope is None and self.attn_mask is None:
            return block(x)
        return block(x, rope=rope, attn_mask=self.attn_mask)

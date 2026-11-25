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
import torch.nn.functional as F

__all__ = [
    "FrozenTail",
    "Dinov3FrozenTail",
    "Qwen3VLVisionFrozenTail",
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
        self.rope_module = getattr(model, "rope", None)

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


class Qwen3VLVisionFrozenTail(nn.Module):
    """Frozen Qwen3-VL visual tail: ``visual.blocks[slot:]`` + ``visual.merger.norm``.

    Matches candidate A (post-final-LayerNorm, before 2x2 spatial merge MLP).
    Loads only the visual tail onto ``device`` and discards the language model.
    """

    def __init__(
        self,
        model_path: str,
        slot: int = 9,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="cpu",
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
        )
        visual = model.model.visual
        n_blocks = len(visual.blocks)
        if not 0 <= slot <= n_blocks:
            raise ValueError(f"slot must be in [0, {n_blocks}], got {slot}")
        if getattr(visual.merger, "use_postshuffle_norm", False):
            raise RuntimeError("Expected visual.merger to use pre-shuffle LayerNorm(hidden_size)")

        self.slot = int(slot)
        self.spatial_merge_size = int(visual.spatial_merge_size)
        self.tail_dtype = dtype
        self.blocks = nn.ModuleList(list(visual.blocks[self.slot :]))
        self.norm = visual.merger.norm
        self.rotary_pos_emb = visual.rotary_pos_emb
        del model, visual

        self.requires_grad_(False)
        self.blocks.to(device=device, dtype=dtype)
        self.norm.to(device=device, dtype=dtype)
        self.rotary_pos_emb.to(device=device, dtype=torch.float32)
        self.eval()

        self._cu_seqlens: torch.Tensor | None = None
        self._position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
        self._expected_tokens: int = 0

    def train(self, mode: bool = True) -> Qwen3VLVisionFrozenTail:
        return super().train(False)

    @torch.no_grad()
    def set_grids(self, grids: list[tuple[int, int]]) -> None:
        """Precompute ``cu_seqlens`` and 2D RoPE for a packed batch of ``(H, W)`` grids."""
        from transformers.vision_utils import get_vision_position_ids

        device = next(self.norm.parameters()).device
        grid_thw = torch.tensor(
            [[1, int(height), int(width)] for height, width in grids],
            dtype=torch.long,
            device=device,
        )
        lengths = grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        rotary = self.rotary_pos_emb(position_ids)
        seq_len = int(lengths.sum().item())
        rotary = rotary.reshape(seq_len, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        self._cu_seqlens = cu_seqlens
        self._position_embeddings = (emb.cos(), emb.sin())
        self._expected_tokens = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cu_seqlens is None or self._position_embeddings is None:
            raise RuntimeError("Call set_grids(grids) before forwarding Qwen3VLVisionFrozenTail")
        orig_shape = x.shape
        if x.ndim == 3:
            batch, length, channels = x.shape
            hidden = x.reshape(batch * length, channels)
        elif x.ndim == 2:
            hidden = x
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {tuple(x.shape)}")
        if hidden.shape[0] != self._expected_tokens:
            raise ValueError(
                f"Expected {self._expected_tokens} packed tokens from set_grids, got {hidden.shape[0]}"
            )
        hidden = hidden.to(dtype=self.tail_dtype)
        for block in self.blocks:
            hidden = block(
                hidden,
                cu_seqlens=self._cu_seqlens,
                position_embeddings=self._position_embeddings,
            )
        hidden = self.norm(hidden)
        if len(orig_shape) == 3:
            return hidden.reshape(orig_shape)
        return hidden

    @torch.no_grad()
    def forward_nograd(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

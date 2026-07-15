"""DINOv3 (timm) frozen-tail wrappers for SoftPQ delta-L_ref training."""

from __future__ import annotations

import torch

from cofai.entropy_models.soft_pq import FrozenTail


class Dinov3FrozenTail:
    """RoPE-aware frozen tail for intermediate DINOv3 split layers."""

    def __init__(
        self,
        model,
        split_layer_idx: int,
        rope,
        attn_mask=None,
        device="cuda",
    ):
        self.tail_start = split_layer_idx + 1
        self.blocks = list(model.blocks[self.tail_start :])
        self.norm = model.norm
        self.rope = rope
        self.attn_mask = attn_mask
        self.rope_mixed = bool(getattr(model, "rope_mixed", False))
        self.device = device

        for blk in self.blocks:
            blk.to(device).eval()
            for p in blk.parameters():
                p.requires_grad_(False)
        self.norm.to(device).eval()
        for p in self.norm.parameters():
            p.requires_grad_(False)

    def _rope_for_block(self, global_i: int):
        if self.rope is None:
            return None
        if self.rope_mixed and isinstance(self.rope, (list, tuple)):
            return self.rope[global_i]
        return self.rope

    def forward_blocks(self, x):
        """Run remaining ViT blocks only (no final LayerNorm)."""
        for global_i in range(self.tail_start, self.tail_start + len(self.blocks)):
            local_i = global_i - self.tail_start
            blk = self.blocks[local_i]
            rope_i = None
            if self.rope is not None:
                rope_i = self.rope[global_i] if self.rope_mixed else self.rope
            if rope_i is not None or self.attn_mask is not None:
                x = blk(x, rope=rope_i, attn_mask=self.attn_mask)
            else:
                x = blk(x)
        return x

    def _forward(self, x):
        return self.norm(self.forward_blocks(x))

    def __call__(self, x):
        return self._forward(x)

    @torch.no_grad()
    def forward_nograd(self, x):
        return self._forward(x)

    def to(self, device):
        for blk in self.blocks:
            blk.to(device)
        self.norm.to(device)
        self.device = device
        return self


def _prime_rope_cache(backbone, token_hw: tuple[int, int], device: torch.device):
    """Run a dummy encode so Dinov3TimmBackbone caches RoPE for token_hw."""
    h_p, w_p = token_hw
    img = torch.zeros(1, 3, h_p * 16, w_p * 16, device=device)
    with torch.no_grad():
        backbone.encode(img)
    return backbone._rope, backbone._attn_mask


def build_dinov3_tail(backbone, layer_idx: int, token_hw: tuple[int, int], device):
    """Build frozen tail for split at layer_idx (0-based block index).

    For blk23 (last block), tail is norm-only. Earlier layers use RoPE-aware tail.
    """
    model = backbone.model
    n_blocks = len(model.blocks)
    dev = str(device)

    if layer_idx >= n_blocks - 1:
        return FrozenTail([], model.norm, device=dev)

    rope, attn_mask = _prime_rope_cache(backbone, token_hw, device)
    return Dinov3FrozenTail(model, layer_idx, rope, attn_mask, device=dev)

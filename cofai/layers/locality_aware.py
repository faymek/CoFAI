# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple

import torch
from einops import rearrange
from torch import nn

from .dcvc_layers import WSiLU


class _WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, dim=2)
        return x1 + x2


class LocalityAwareBlock(nn.Module):
    """Locality-Aware Block for a prefix-plus-patch token sequence.

    The design is adapted from the DC blocks used throughout the DCVC family.
    Prefix tokens are transformed independently with an MLP, while patch
    tokens are reshaped to their spatial grid and processed by a pointwise-
    depthwise-pointwise convolution. A shared gated feed-forward network then
    mixes channels for the complete token sequence.
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        shortcut=False,
        force_adaptor=False,
        zero_init_residual=False,
    ):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Linear(in_ch, out_ch)
        self.shortcut = shortcut
        self.prefix_linear = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            WSiLU(),
            nn.Linear(out_ch, out_ch),
        )
        self.patch_dconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Linear(out_ch, out_ch * 4),
            _WSiLUChunkAdd(),
            nn.Linear(out_ch * 2, out_ch),
        )
        if zero_init_residual:
            nn.init.zeros_(self.prefix_linear[-1].weight)
            nn.init.zeros_(self.prefix_linear[-1].bias)
            nn.init.zeros_(self.patch_dconv[-1].weight)
            nn.init.zeros_(self.patch_dconv[-1].bias)
            nn.init.zeros_(self.ffn[-1].weight)
            nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor, resolution: Tuple[int, int]):
        if self.adaptor is not None:
            x = self.adaptor(x)

        height, width = resolution
        num_prefix_tokens = x.shape[1] - height * width
        x_prefix = self.prefix_linear(x[:, :num_prefix_tokens])

        x_patch = rearrange(
            x[:, num_prefix_tokens:],
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )
        x_patch = self.patch_dconv(x_patch)
        x_patch = rearrange(x_patch, "b c h w -> b (h w) c")

        x_trans = torch.cat((x_prefix, x_patch), dim=1).contiguous()
        out = x_trans + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        return out

"""View-decoupled Transformer, adapted from VDT/FastReID and Ross Wightman's ViT.

ViT implementation copyright 2020 Ross Wightman.
"""

import torch
from torch import nn


class VDTAttention(nn.Module):
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attention @ v).transpose(1, 2).reshape(batch, length, dim))


class VDTMlp(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VDTBlock(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = VDTAttention(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = VDTMlp(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VDTPatchEmbed(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=16, stride=16)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VDTBackbone(nn.Module):
    def __init__(self, image_size=(256, 128), dim=768, slot=11):
        super().__init__()
        self.slot = slot
        self.patch_embed = VDTPatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.view_token = nn.Parameter(torch.zeros(1, 1, dim))
        count = (image_size[0] // 16) * (image_size[1] // 16)
        self.pos_embed = nn.Parameter(torch.zeros(1, count + 2, dim))
        self.blocks = nn.ModuleList([VDTBlock(dim) for _ in range(12)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def run_blocks(self, tokens, start, stop):
        for block in self.blocks[start:stop]:
            tokens = block(tokens)
            tokens[:, 0] = tokens[:, 0] - tokens[:, 1]
        return tokens

    def encode(self, image, **kwargs):
        patches = self.patch_embed((image - 0.5) / 0.5)
        batch = image.shape[0]
        tokens = torch.cat(
            (self.cls_token.expand(batch, -1, -1), self.view_token.expand(batch, -1, -1), patches), dim=1
        )
        return self.run_blocks(tokens + self.pos_embed, 0, self.slot)

    def final_tokens(self, tokens):
        return self.norm(self.run_blocks(tokens, self.slot, len(self.blocks)))

    def decode(self, h, *, tasks, **kwargs):
        return {"reid": self.final_tokens(h)} if "reid" in tasks else {}

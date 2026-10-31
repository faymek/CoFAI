"""Batch-normalized VDT retrieval embedding."""

from torch import nn


class VDTReIDHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.bottleneck = nn.BatchNorm1d(dim)

    def forward(self, tokens):
        return self.bottleneck(tokens[:, 0] - tokens[:, 1])

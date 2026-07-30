"""Embedding head for the TransReID JPM evaluation path."""

from __future__ import annotations

import torch
import torch.nn as nn

from cofai.backbone.gps_transreid import TransReIDJPMFeatures


class TransReIDJPMHead(nn.Module):
    """Apply the released TransReID bottlenecks and embedding concatenation."""

    def __init__(
        self,
        bottleneck: nn.Module,
        local_bottlenecks: list[nn.Module],
        *,
        neck_feature: str,
    ):
        super().__init__()
        self.neck_feature = str(neck_feature)
        self.bottleneck = bottleneck
        self.local_bottlenecks = nn.ModuleList(local_bottlenecks)
        if len(self.local_bottlenecks) != 4:
            raise ValueError("the TransReID JPM head requires four local bottlenecks")

    def forward(self, features: TransReIDJPMFeatures) -> torch.Tensor:
        if not isinstance(features, TransReIDJPMFeatures):
            raise TypeError(
                "TransReIDJPMHead expects TransReIDJPMFeatures from "
                "GPSTransReIDBackbone.decode"
            )
        if self.training:
            raise RuntimeError("TransReIDJPMHead is an evaluation-only embedding head")
        if len(features.local_token_features) != len(self.local_bottlenecks):
            raise ValueError(
                "TransReID JPM features and head must expose four local branches"
            )

        global_bn = self.bottleneck(features.bottleneck_global_feature)
        local_bn = [
            bottleneck(tokens[:, 0])
            for bottleneck, tokens in zip(
                self.local_bottlenecks,
                features.local_token_features,
                strict=True,
            )
        ]

        if self.neck_feature == "after":
            parts = [global_bn, *(feature / 4 for feature in local_bn)]
        else:
            local_features = [
                tokens[:, 0] / 4 for tokens in features.local_token_features
            ]
            parts = [features.global_feature, *local_features]
        return torch.cat(parts, dim=1)

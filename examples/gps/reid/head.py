"""Task head for checkpoint-compatible GPS vehicle retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GPSReIDFeatures:
    """Backbone features consumed by the GPS ReID embedding head."""

    global_feature: torch.Tensor
    bottleneck_global_feature: torch.Tensor
    local_token_features: tuple[torch.Tensor, ...]


class GPSReIDHead(nn.Module):
    """Apply the released GPS bottlenecks and embedding concatenation."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.neck_feature = str(model.neck_feat)
        self.cls_token_num = int(model.cls_token_num)
        self.bottleneck = model.bottleneck
        self.local_bottlenecks = nn.ModuleList(
            [
                model.bottleneck_1,
                model.bottleneck_2,
                model.bottleneck_3,
                model.bottleneck_4,
            ]
        )

    def forward(self, features: GPSReIDFeatures) -> torch.Tensor:
        if not isinstance(features, GPSReIDFeatures):
            raise TypeError(
                "GPSReIDHead expects GPSReIDFeatures from GPSReIDBackbone.decode"
            )
        if self.training:
            raise RuntimeError("GPSReIDHead is an evaluation-only embedding head")
        if len(features.local_token_features) != len(self.local_bottlenecks):
            raise ValueError("GPS ReID backbone and head must expose four local branches")

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
            batch_size = features.global_feature.shape[0]
            local_features = [
                tokens[:, : self.cls_token_num].reshape(batch_size, -1) / 4
                for tokens in features.local_token_features
            ]
            parts = [features.global_feature, *local_features]
        return torch.cat(parts, dim=1)

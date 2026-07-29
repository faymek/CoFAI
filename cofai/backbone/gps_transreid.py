"""Checkpoint-compatible GPS TransReID backbone integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cofai.index_codecs import BoundedIndexSetBatch


@dataclass(frozen=True)
class GPSTransReIDFeatures:
    """Task features emitted by the GPS TransReID decoder."""

    global_feature: torch.Tensor
    bottleneck_global_feature: torch.Tensor
    local_token_features: tuple[torch.Tensor, ...]


def shuffle_unit(features, shift, group, begin=1):
    """Apply the TransReID JPM shift-and-shuffle operation."""
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


class GPSTransReIDBackbone(nn.Module):
    """Adapt a released GPS TransReID model to the CoFAI backbone boundary.

    GPS grouping remains inside the specialized multi-view TransReID encoder
    because it is interleaved with several Transformer blocks. The decoder
    preserves the checkpoint's global branch and four JPM local branches.
    """

    def __init__(
        self,
        encoder: nn.Module,
        global_decoder: nn.Module,
        local_decoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.global_decoder = global_decoder
        self.local_decoder = local_decoder

    @property
    def patches_per_view(self) -> int:
        """Return the spatial token count produced for one input view."""
        return int(self.encoder.patch_embed.num_patches)

    @property
    def token_grouper(self):
        """Return the grouping callable used inside the GPS Transformer."""
        return self.encoder.token_grouper

    @token_grouper.setter
    def token_grouper(self, grouper) -> None:
        self.encoder.token_grouper = grouper

    def encode(
        self,
        x,
        label=None,
        cam_label=None,
        view_label=None,
        dataset_name="query",
        **kwargs,
    ):
        h, _order, _flops, selection = self.encoder(
            x,
            cam_label=cam_label,
            view_label=view_label,
            label=label,
            dataset_name=dataset_name,
        )
        indices = selection["indices"]
        token_count = int(selection["token_count"])
        if h.ndim != 3 or h.shape[0] != indices.shape[0]:
            raise ValueError("GPS compact features and selection indices disagree")
        if h.shape[1] != indices.shape[1] + 1:
            raise ValueError(
                "GPS compact feature must contain one CLS token plus retained patches"
            )

        index_sets = {}
        is_pruned = indices.shape[1] < token_count
        if is_pruned:
            index_sets["selection_map"] = BoundedIndexSetBatch(
                indices=indices,
                universe_size=token_count,
            )

        return {
            "h": h,
            "index_sets": index_sets,
        }

    def decode(self, h, *, tasks, **kwargs):
        if "reid" not in tasks:
            return {}

        global_tokens = self.global_decoder(h)
        global_feature = global_tokens[:, 0]
        patch_length = (h.size(1) - 1) // 4
        cls_tokens = h[:, :1]
        patch_tokens = shuffle_unit(h, shift=8, group=2)

        local_token_features = []
        for index in range(4):
            start = index * patch_length
            local_tokens = patch_tokens[:, start : start + patch_length]
            local_token_features.append(
                self.local_decoder(torch.cat((cls_tokens, local_tokens), dim=1))
            )

        return {
            "reid": GPSTransReIDFeatures(
                global_feature=global_feature,
                bottleneck_global_feature=global_feature,
                local_token_features=tuple(local_token_features),
            ),
        }

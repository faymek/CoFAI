"""Checkpoint-compatible GPS TransReID backbone integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cofai.index_codecs import AdaptiveBitmapIndexCodec

_SELECTION_MAP_CODEC = AdaptiveBitmapIndexCodec()


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

    def __init__(self, model: nn.Module):
        super().__init__()
        self.base = model.base
        self.global_decoder = model.b1
        self.local_decoder = model.b2
        self.cls_token_num = int(model.cls_token_num)
        self.shuffle_groups = int(model.shuffle_groups)
        self.shift_num = int(model.shift_num)
        self.divide_length = int(model.divide_length)
        self.rearrange = bool(model.rearrange)

    @property
    def patches_per_view(self) -> int:
        """Return the spatial token count produced for one input view."""
        return int(self.base.patch_embed.num_patches)

    @property
    def token_grouper(self):
        """Return the grouping callable used inside the GPS Transformer."""
        return self.base.token_grouper

    @token_grouper.setter
    def token_grouper(self, grouper) -> None:
        self.base.token_grouper = grouper

    def encode(
        self,
        x,
        label=None,
        cam_label=None,
        view_label=None,
        multi_view=False,
        flip_view=False,
        extra_token=False,
        dataset_name="train",
        **kwargs,
    ):
        if multi_view:
            h, _order, _flops, selection = self.base(
                x,
                cam_label=cam_label,
                view_label=view_label,
                label=label,
                multi_view=True,
                flip_view=flip_view,
                extra_token=extra_token,
                dataset_name=dataset_name,
            )
        else:
            h = self.base(
                x,
                cam_label=cam_label,
                view_label=view_label,
                label=label,
                multi_view=False,
                extra_token=extra_token,
                dataset_name=dataset_name,
            )
            selection = None

        if selection is None:
            raise RuntimeError(
                "GPS encoder did not expose a final token-selection map; "
                "multi-view GPS inference is required"
            )
        indices = selection["indices"]
        token_count = int(selection["token_count"])
        if h.ndim != 3 or h.shape[0] != indices.shape[0]:
            raise ValueError("GPS compact features and selection indices disagree")
        if h.shape[1] != indices.shape[1] + 1:
            raise ValueError(
                "GPS compact feature must contain one CLS token plus retained patches"
            )

        strings: dict[str, list[list[bytes]]] = {}
        is_pruned = indices.shape[1] < token_count
        if is_pruned:
            rows = []
            for values in indices.detach().cpu().tolist():
                payload = _SELECTION_MAP_CODEC.encode(
                    values,
                    token_count,
                ).to_bytes()
                rows.append([payload])
            strings["selection_map"] = rows

        return {
            "h": h,
            "strings": strings,
            "pstate": {
                "extra_token": bool(extra_token),
                "original_patch_tokens": token_count,
                "retained_patch_tokens": int(indices.shape[1]),
                "feature_dimension": int(h.shape[-1]),
            },
        }

    def decode(self, h, *, pstate, tasks, **kwargs):
        if "reid" not in tasks:
            return {}

        batch_size = h.shape[0]
        global_tokens = self.global_decoder(h)
        extra_token = bool(pstate.get("extra_token", False))
        if extra_token:
            global_feature = global_tokens[:, : self.cls_token_num].reshape(
                batch_size,
                -1,
            )
            cls_token_num = self.cls_token_num
            bottleneck_global_feature = global_tokens[:, 0]
        else:
            global_feature = global_tokens[:, 0]
            cls_token_num = 1
            bottleneck_global_feature = global_feature

        patch_length = (h.size(1) - cls_token_num) // self.divide_length
        cls_tokens = h[:, :cls_token_num]
        if self.rearrange:
            patch_tokens = shuffle_unit(
                h,
                self.shift_num,
                self.shuffle_groups,
                cls_token_num,
            )
        else:
            patch_tokens = h[:, cls_token_num:]

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
                bottleneck_global_feature=bottleneck_global_feature,
                local_token_features=tuple(local_token_features),
            ),
        }

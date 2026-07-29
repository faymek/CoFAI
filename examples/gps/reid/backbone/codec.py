"""Feature-codec adapter for the checkpoint-compatible GPS ReID model."""

from __future__ import annotations

import torch.nn as nn

from cofai.token_codecs import encode_selection_indices


class GPSReIDBackbone(nn.Module):
    """Expose the legacy GPS model through a DINO-like encode/decode boundary.

    GPS grouping remains inside ``model.encode`` because it is interleaved with
    several Transformer blocks. The adapter only serializes the final selection
    map as side information; it does not move or rename checkpointed modules.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def patches_per_view(self) -> int:
        """Return the spatial token count produced for one input view."""
        return int(self.model.base.patch_embed.num_patches)

    @property
    def token_grouper(self):
        """Return the grouping callable used inside the GPS Transformer."""
        return self.model.base.token_grouper

    @token_grouper.setter
    def token_grouper(self, grouper) -> None:
        self.model.base.token_grouper = grouper

    def encode(self, x, **kwargs):
        encoded = self.model.encode(x, **kwargs)
        h = encoded["h"]
        context = dict(encoded["context"])

        selection = encoded.get("selection")
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
        bits: dict[str, float] = {}
        is_pruned = indices.shape[1] < token_count
        if is_pruned:
            rows = []
            total_bits = 0
            for values in indices.detach().cpu().tolist():
                payload = encode_selection_indices(values, token_count).to_bytes()
                rows.append([payload])
                total_bits += len(payload) * 8
            strings["selection_map"] = rows
            bits["selection_map"] = float(total_bits)

        context.update(
            {
                "original_patch_tokens": token_count,
                "retained_patch_tokens": int(indices.shape[1]),
                "feature_dimension": int(h.shape[-1]),
            }
        )
        return {
            "h": h,
            "context": context,
            "strings": strings,
            "bits": bits,
        }

    def decode(self, h, *, context, tasks, **kwargs):
        output = self.model.decode(h, context=context, tasks=tasks)
        if self.model.training:
            return {"reid": output}
        feature, order = output
        return {
            "reid": feature,
            "order": order,
        }

"""Runtime collection of real GPS token-selection maps."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from cofai.backbone import GPSTransReIDBackbone
from cofai.models import CommonFeatureCodecModel


@dataclass
class ProbeRecord:
    n_tokens: int
    kept_indices: list[int]
    preprocess_ms: float


def _timed_call(device: torch.device, function, *args, **kwargs):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = function(*args, **kwargs)
        end.record()
        torch.cuda.synchronize(device)
        return value, float(start.elapsed_time(end))
    start = time.perf_counter()
    value = function(*args, **kwargs)
    return value, (time.perf_counter() - start) * 1000.0


class TokenGroupingProbe:
    """Wrap a grouper and record the final fused three-view selection map."""

    def __init__(self, grouper, patches_per_view: int):
        self.grouper = grouper
        self.patches_per_view = int(patches_per_view)
        self.records: list[ProbeRecord] = []
        self._batch_elapsed_ms = 0.0

    def __call__(self, *args, **kwargs):
        tokens = kwargs.get("tokens", args[0] if args else None)
        original_indices = kwargs.get("original_indices")
        if tokens is None or original_indices is None:
            raise ValueError("probe requires tokens and original_indices")

        grouped, elapsed_ms = _timed_call(tokens.device, self.grouper, *args, **kwargs)
        self._batch_elapsed_ms += elapsed_ms

        view_indices = torch.div(
            original_indices,
            self.patches_per_view,
            rounding_mode="floor",
        )
        is_fused_query = bool(
            original_indices.numel()
            and torch.all(
                view_indices.max(dim=1).values > view_indices.min(dim=1).values
            )
        )
        if is_fused_query:
            elapsed_per_group = self._batch_elapsed_ms / max(
                grouped.kept_indices.shape[0], 1
            )
            for indices in grouped.kept_indices.detach().cpu().tolist():
                kept = sorted(int(index) for index in indices)
                if len(kept) != len(set(kept)):
                    raise RuntimeError(
                        "selection map contains duplicate original indices"
                    )
                self.records.append(
                    ProbeRecord(
                        n_tokens=3 * self.patches_per_view,
                        kept_indices=kept,
                        preprocess_ms=elapsed_per_group,
                    )
                )
            self._batch_elapsed_ms = 0.0
        return grouped


def install_token_grouping_probe(model) -> TokenGroupingProbe:
    """Install and return a collector on a GPS ReID model."""
    backbone = model.backbone if isinstance(model, CommonFeatureCodecModel) else model
    if not isinstance(backbone, GPSTransReIDBackbone):
        raise TypeError("token grouping probes require GPSTransReIDBackbone")
    probe = TokenGroupingProbe(
        backbone.token_grouper,
        backbone.patches_per_view,
    )
    backbone.token_grouper = probe
    return probe

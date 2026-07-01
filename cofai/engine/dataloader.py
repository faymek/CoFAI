"""Build DataLoader that yields `EvalBatch`."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from cofai.engine.schema import EvalBatch

__all__ = ["build_dataloader", "collate_fn"]


def collate_fn(batch: List[Dict[str, Any]]) -> EvalBatch:
    """Stack the shared image input and preserve per-task sample payloads."""

    if not batch:
        raise ValueError("Empty batch")

    xs: List[torch.Tensor] = []
    samples: List[Dict[str, Any]] = []
    for item in batch:
        if "img" not in item:
            raise KeyError("Eval batch items must contain the shared `img` field.")
        sample = dict(item)
        xs.append(_to_chw(sample.pop("img")))
        samples.append(sample)
    return EvalBatch(inputs={"img": torch.stack(xs, dim=0)}, samples=samples)


def build_dataloader(cfg: Any, dataset: Any) -> DataLoader:
    """Return DataLoader producing EvalBatch."""

    raw = OmegaConf.to_container(OmegaConf.select(cfg, "dataloader", default={}) or {}, resolve=True)
    kw = raw if isinstance(raw, dict) else {}
    kw.pop("dataset", None)
    kw.pop("collate_fn", None)
    return DataLoader(dataset, collate_fn=collate_fn, **kw)


def _to_chw(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
        if t.dim() == 3:
            t = t.permute(2, 0, 1)
    else:
        # PIL: use ToTensor lazily to avoid importing torchvision here.
        from torchvision.transforms import ToTensor

        t = ToTensor()(x)
    if t.dim() == 4 and t.size(0) == 1:
        t = t[0]
    if t.dim() != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape={tuple(t.shape)}")
    return t

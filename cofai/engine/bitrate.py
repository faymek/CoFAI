"""Bit accounting for coded units emitted by CoFAI models and codecs."""

from __future__ import annotations

import math
from typing import Any

import torch


__all__ = ["bits_from_coded_unit", "bits_from_coded_data"]


def bits_from_coded_unit(data: dict[str, Any]) -> dict[str, float]:
    """Return per-stream bits from one real or estimated coded unit."""
    if "strings" in data:
        return {
            str(name): float(sum(len(stream[0]) for stream in streams) * 8.0)
            for name, streams in data["strings"].items()
        }
    if "likelihoods" in data:
        return {
            str(name): float((torch.log(likelihoods).sum() / (-math.log(2))).item())
            for name, likelihoods in data["likelihoods"].items()
        }
    if "bits" in data:
        return {str(name): float(bits) for name, bits in data["bits"].items()}
    raise KeyError("Expected key `strings`, `likelihoods`, or `bits` in coded unit")


def bits_from_coded_data(out: dict[str, Any]) -> dict[str, float]:
    """Flatten per-stream bits from any coded-data container."""
    if "type" not in out:
        return bits_from_coded_unit(out)

    out_type = out.get("type")
    if out_type == "unit":
        return bits_from_coded_unit(out["data"])
    if out_type == "frame":
        flat: dict[str, float] = {}
        for layer_name, coded_unit in out["data"].items():
            bits_items = bits_from_coded_unit(coded_unit)
            for name, bits in bits_items.items():
                flat[f"{layer_name}.{name}"] = float(bits)
        return flat
    if out_type == "frame_wise_video":
        flat = {}
        for frame_name, coded_frame in out["data"].items():
            for layer_name, coded_unit in coded_frame["data"].items():
                bits_items = bits_from_coded_unit(coded_unit)
                for name, bits in bits_items.items():
                    flat[f"{frame_name}.{layer_name}.{name}"] = float(bits)
        return flat
    if out_type == "layer_wise_video":
        flat = {}
        for layer_name, coded_frame in out["data"].items():
            for frame_name, coded_unit in coded_frame["data"].items():
                bits_items = bits_from_coded_unit(coded_unit)
                for name, bits in bits_items.items():
                    flat[f"{layer_name}.{frame_name}.{name}"] = float(bits)
        return flat
    if out_type == "slide_crops":
        flat = {}
        for coded_unit in out["data"]:
            bits_items = bits_from_coded_unit(coded_unit)
            for name, bits in bits_items.items():
                flat[name] = flat.get(name, 0.0) + float(bits)
        return flat
    raise NotImplementedError(f"Unsupported type: {out_type!r}")

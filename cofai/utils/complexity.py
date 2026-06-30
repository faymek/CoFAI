"""Parameter and FLOPs measurement for latent codecs."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.flop_counter import FlopCounterMode


def count_params(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def measure_flops(module: Any, *args: Any, **kwargs: Any) -> Optional[int]:
    """FLOPs of one ``module(*args, **kwargs)`` forward.

    Returns ``None`` only when measurement raises; a successful run with zero
    recorded ATen FLOPs (e.g. bypass pass-through) returns ``0``.
    """
    try:
        flop_counter = FlopCounterMode(display=False)
        with torch.no_grad(), flop_counter:
            module(*args, **kwargs)
        return int(flop_counter.get_total_flops())
    except Exception:
        return None


def latent_codec_complexity(codec: torch.nn.Module, h, token_res, qp=0) -> dict[str, Any]:
    """Params and encode/decode FLOPs for a latent codec on given tokens."""
    out: dict[str, Any] = {"codec_params": count_params(codec)}
    enc_flops = measure_flops(codec.compress, h, token_res, qp)
    if enc_flops is not None:
        out["codec_enc_flops"] = enc_flops
    with torch.no_grad():
        coded = codec.compress(h, token_res, qp=qp)
    dec_flops = measure_flops(codec.decompress, **coded)
    if dec_flops is not None:
        out["codec_dec_flops"] = dec_flops
    return out

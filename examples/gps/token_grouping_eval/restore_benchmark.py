"""Deterministic restoration benchmark helpers."""

from __future__ import annotations

import time

import torch

from cofai.token_grouping.restore import restore_fixed_length


@torch.inference_mode()
def benchmark_fixed_restore(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    token_count: int,
    repeats: int = 20,
    warmup: int = 5,
    fill_value: float = 0.0,
) -> dict[str, float | tuple[int, ...]]:
    """Measure scatter restoration and report its output shape."""
    if repeats <= 0 or warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    for _ in range(warmup):
        restore_fixed_length(tokens, indices, token_count, fill_value)
    if tokens.is_cuda:
        torch.cuda.synchronize(tokens.device)
    start = time.perf_counter()
    restored = None
    for _ in range(repeats):
        restored = restore_fixed_length(tokens, indices, token_count, fill_value)
    if tokens.is_cuda:
        torch.cuda.synchronize(tokens.device)
    elapsed = (time.perf_counter() - start) / repeats
    return {"restore_seconds": elapsed, "output_shape": tuple(restored.shape)}

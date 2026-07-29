"""Bit-true codec for direct FP16 feature transmission."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


_DTYPE_TO_NAME = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


class FP16Codec(nn.Module):
    """Transmit a feature tensor as contiguous IEEE FP16 samples.

    The wire representation contains one byte string per batch item. Decoder-side
    tensors are cast back to the encoder input dtype after the FP16 round trip so
    downstream modules can retain their original parameter dtype.
    """

    stream_name = "fp16"

    def forward(self, h, token_res=None, qp=0, **kwargs):
        self._validate_input(h)
        h_hat = h.to(dtype=torch.float16).to(dtype=h.dtype)
        return {
            "h_hat": h_hat,
            "bits": {self.stream_name: float(h.numel() * 16)},
        }

    def compress(self, h, token_res=None, qp=0, **kwargs):
        self._validate_input(h)
        wire = h.detach().to(device="cpu", dtype=torch.float16).contiguous()
        rows = [
            [wire[index].numpy().tobytes(order="C")] for index in range(wire.shape[0])
        ]
        return {
            "strings": {self.stream_name: rows},
            "pstate": {
                "fp16_shape": tuple(int(value) for value in h.shape),
                "fp16_output_dtype": _DTYPE_TO_NAME[h.dtype],
                "fp16_output_device": str(h.device),
                "fp16_token_res": token_res,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        try:
            rows = strings[self.stream_name]
        except KeyError as exc:
            raise KeyError(f"missing {self.stream_name!r} feature stream") from exc

        shape = tuple(int(value) for value in pstate["fp16_shape"])
        if len(shape) < 2:
            raise ValueError("fp16_shape must include batch and feature dimensions")
        if len(rows) != shape[0]:
            raise ValueError(
                f"FP16 stream batch mismatch: expected {shape[0]}, got {len(rows)}"
            )

        sample_shape = shape[1:]
        sample_numel = 1
        for value in sample_shape:
            sample_numel *= value
        expected_bytes = sample_numel * 2

        samples = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 1:
                raise ValueError(
                    "each FP16 stream row must contain exactly one byte string"
                )
            payload = row[0]
            if not isinstance(payload, bytes):
                raise TypeError("FP16 stream payloads must be bytes")
            if len(payload) != expected_bytes:
                raise ValueError(
                    "FP16 stream size mismatch: "
                    f"expected {expected_bytes} bytes, got {len(payload)}"
                )
            sample = torch.frombuffer(
                bytearray(payload),
                dtype=torch.float16,
            ).reshape(sample_shape)
            samples.append(sample)

        h_hat = torch.stack(samples, dim=0)
        dtype_name = str(pstate["fp16_output_dtype"])
        try:
            output_dtype = _NAME_TO_DTYPE[dtype_name]
        except KeyError as exc:
            raise ValueError(f"unsupported FP16 output dtype: {dtype_name!r}") from exc
        output_device = kwargs.get("device", pstate["fp16_output_device"])
        h_hat = h_hat.to(device=output_device, dtype=output_dtype)
        return {"h_hat": h_hat}

    @staticmethod
    def _validate_input(h: Any) -> None:
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"FP16Codec expects a tensor, got {type(h)!r}")
        if h.ndim < 2:
            raise ValueError("FP16Codec expects a batched feature tensor")
        if h.dtype not in _DTYPE_TO_NAME:
            raise TypeError(f"unsupported feature dtype for FP16Codec: {h.dtype}")

"""Bit-true codec for raw transmission in a native PyTorch floating dtype."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


_DTYPE_NAMES = (
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)
_NAME_TO_DTYPE = {
    name: getattr(torch, name) for name in _DTYPE_NAMES if hasattr(torch, name)
}
_DTYPE_TO_NAME = {dtype: name for name, dtype in _NAME_TO_DTYPE.items()}


def _resolve_dtype(dtype: str | torch.dtype) -> tuple[str, torch.dtype]:
    if isinstance(dtype, torch.dtype):
        try:
            return _DTYPE_TO_NAME[dtype], dtype
        except KeyError as exc:
            raise ValueError(f"unsupported raw wire dtype: {dtype}") from exc
    name = str(dtype).removeprefix("torch.")
    try:
        return name, _NAME_TO_DTYPE[name]
    except KeyError as exc:
        choices = ", ".join(_NAME_TO_DTYPE)
        raise ValueError(
            f"unsupported raw wire dtype {dtype!r}; expected one of: {choices}"
        ) from exc


class RawDtypeCodec(nn.Module):
    """Cast features to a native floating dtype and transmit their raw bytes.

    This is a scalar quantization baseline rather than an entropy codec. The wire
    representation contains one byte string per batch item. Decoded values are cast
    back to the encoder input dtype so downstream modules retain their parameter
    dtype.
    """

    stream_name = "feature"

    def __init__(self, dtype: str | torch.dtype = "float16"):
        super().__init__()
        self.dtype_name, self.wire_dtype = _resolve_dtype(dtype)
        self.bits_per_value = int(self.wire_dtype.itemsize * 8)

    def forward(self, h, token_res=None, qp=0, **kwargs):
        self._validate_input(h)
        h_hat = h.to(dtype=self.wire_dtype).to(dtype=h.dtype)
        return {
            "h_hat": h_hat,
            "bits": {
                self.stream_name: float(h.numel() * self.bits_per_value),
            },
        }

    def compress(self, h, token_res=None, qp=0, **kwargs):
        self._validate_input(h)
        wire = h.detach().to(device="cpu", dtype=self.wire_dtype).contiguous()
        rows = [
            [wire[index].view(torch.uint8).numpy().tobytes(order="C")]
            for index in range(wire.shape[0])
        ]
        return {
            "strings": {self.stream_name: rows},
            "pstate": {
                "raw_shape": tuple(int(value) for value in h.shape),
                "raw_source_dtype": _DTYPE_TO_NAME[h.dtype],
                "raw_wire_dtype": self.dtype_name,
                "raw_token_res": token_res,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        try:
            rows = strings[self.stream_name]
        except KeyError as exc:
            raise KeyError(f"missing {self.stream_name!r} feature stream") from exc

        shape = tuple(int(value) for value in pstate["raw_shape"])
        if len(shape) < 2:
            raise ValueError("raw_shape must include batch and feature dimensions")
        if len(rows) != shape[0]:
            raise ValueError(
                f"raw stream batch mismatch: expected {shape[0]}, got {len(rows)}"
            )

        wire_name, wire_dtype = _resolve_dtype(pstate["raw_wire_dtype"])
        sample_shape = shape[1:]
        sample_numel = 1
        for value in sample_shape:
            sample_numel *= value
        expected_bytes = sample_numel * wire_dtype.itemsize

        samples = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 1:
                raise ValueError(
                    "each raw feature stream row must contain exactly one byte string"
                )
            payload = row[0]
            if not isinstance(payload, bytes):
                raise TypeError("raw feature stream payloads must be bytes")
            if len(payload) != expected_bytes:
                raise ValueError(
                    f"{wire_name} stream size mismatch: "
                    f"expected {expected_bytes} bytes, got {len(payload)}"
                )
            sample = (
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                .view(wire_dtype)
                .reshape(sample_shape)
            )
            samples.append(sample)

        h_hat = torch.stack(samples, dim=0)
        _, output_dtype = _resolve_dtype(pstate["raw_source_dtype"])
        output_device = kwargs.get("device", "cpu")
        h_hat = h_hat.to(device=output_device, dtype=output_dtype)
        return {"h_hat": h_hat}

    @staticmethod
    def _validate_input(h: Any) -> None:
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"RawDtypeCodec expects a tensor, got {type(h)!r}")
        if h.ndim < 2:
            raise ValueError("RawDtypeCodec expects a batched feature tensor")
        if h.dtype not in _DTYPE_TO_NAME:
            raise TypeError(f"unsupported source feature dtype: {h.dtype}")

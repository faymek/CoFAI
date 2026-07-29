"""Raw feature transmission in a native PyTorch floating dtype."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


_WIRE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


class RawDtypeCodec(nn.Module):
    """Cast features to a floating dtype and transmit their raw bytes.

    This is a scalar quantization baseline, not an entropy codec. Decoded values
    are cast back to the encoder input dtype for downstream inference.
    """

    stream_name = "feature"

    def __init__(self, dtype: str = "float16"):
        super().__init__()
        try:
            self.wire_dtype = _WIRE_DTYPES[dtype]
        except (KeyError, TypeError) as exc:
            choices = ", ".join(_WIRE_DTYPES)
            raise ValueError(
                f"unsupported raw wire dtype {dtype!r}; expected one of: {choices}"
            ) from exc
        self.dtype_name = dtype
        self.bits_per_value = int(self.wire_dtype.itemsize * 8)

    def forward(self, h, **kwargs):
        self._validate_input(h)
        return {
            "h_hat": h.to(self.wire_dtype).to(h.dtype),
            "bits": {
                self.stream_name: float(h.numel() * self.bits_per_value),
            },
        }

    def compress(self, h, **kwargs):
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
                "raw_source_dtype": str(h.dtype).removeprefix("torch."),
                "raw_wire_dtype": self.dtype_name,
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

        wire_name = str(pstate["raw_wire_dtype"])
        try:
            wire_dtype = _WIRE_DTYPES[wire_name]
        except KeyError as exc:
            raise ValueError(f"unsupported raw wire dtype: {wire_name!r}") from exc

        sample_shape = shape[1:]
        expected_bytes = math.prod(sample_shape) * wire_dtype.itemsize
        samples = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 1:
                raise ValueError(
                    "each raw feature stream row must contain one byte string"
                )
            payload = row[0]
            if not isinstance(payload, bytes):
                raise TypeError("raw feature stream payloads must be bytes")
            if len(payload) != expected_bytes:
                raise ValueError(
                    f"{wire_name} stream size mismatch: "
                    f"expected {expected_bytes} bytes, got {len(payload)}"
                )
            samples.append(
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                .view(wire_dtype)
                .reshape(sample_shape)
            )

        source_name = str(pstate["raw_source_dtype"])
        output_dtype = getattr(torch, source_name, None)
        if not isinstance(output_dtype, torch.dtype):
            raise ValueError(f"unsupported source feature dtype: {source_name!r}")
        h_hat = torch.stack(samples).to(
            device=kwargs.get("device", "cpu"),
            dtype=output_dtype,
        )
        return {"h_hat": h_hat}

    @staticmethod
    def _validate_input(h) -> None:
        if not isinstance(h, torch.Tensor):
            raise TypeError(f"RawDtypeCodec expects a tensor, got {type(h)!r}")
        if h.ndim < 2:
            raise ValueError("RawDtypeCodec expects a batched feature tensor")
        if not h.is_floating_point():
            raise TypeError(f"unsupported source feature dtype: {h.dtype}")

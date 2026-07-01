"""Unified eval data structures (protocol implementation).

This module is the *engine-level* home for eval contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, TypedDict, runtime_checkable

import torch

__all__ = [
    # ---- codec schema + protocols (merged from cofai/spec/codec.py) ----
    "StringsValue",
    "CodedUnit",
    "CodedData",
    "CodecArgs",
    "DataUnitCodecProtocol",
    "LayeredFrameCodecProtocol",
    "LayeredVideoCodecProtocol",
    # ---- eval contracts ----
    "EvalResultPayload",
    "EvalBatch",
    "StepOutput",
    "_validate_step_output_soft",
    # ---- test_step protocol + mixins ----
    "TestStepProtocol",
]


# =============================================================================
# Codec schema + protocols (proposal-level ABI)
# =============================================================================

StringsValue = list[list[bytes]] | bytes


@dataclass
class CodedUnit:
    strings: dict[str, StringsValue]
    pstate: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodedData:
    type: Literal["frame", "frame_wise_video", "layer_wise_video"]
    data: dict[Any, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodecArgs:
    tasks: list[str] = field(default_factory=list)
    qp: Optional[int] = None
    device: Optional[str] = None
    strict: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DataUnitCodecProtocol(Protocol):
    def compress(self, x: Any, **kwargs: Any) -> Mapping[str, Any]: ...

    def decompress(self, coded_unit: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class LayeredFrameCodecProtocol(Protocol):
    def compress(self, x: Any, **kwargs: Any) -> Mapping[str, Any]: ...

    def decompress(self, coded_data: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class LayeredVideoCodecProtocol(Protocol):
    def compress_video(self, video_reader: Any, codec_args: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]: ...

    def decompress_video(self, coded_data: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any] | list[Any]: ...


# =============================================================================
# Eval contracts
# =============================================================================


class EvalResultPayload(TypedDict):
    name: str
    description: str
    results: Dict[str, Any]
    quality: Optional[str]
    records: List[Dict[str, Any]]


@dataclass(frozen=True)
class EvalBatch:
    """Packed batch ready for model inference (MMEngine-style split).

    Produced by collate+pack logic (typically in `cofai.engine.dataloader`).

    - ``inputs``: shared model inputs; currently contains ``img`` as
      ``FloatTensor[B,C,H,W]`` for every image task.
    - ``samples``: length ``B``; each entry matches the dataset/transform dict
      except ``img`` is omitted. Task-specific inputs and annotations live under
      their task kind (for example ``samples[i]["vqa"]``).
    """

    inputs: Dict[str, Any]
    samples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StepOutput:
    """Model step output consumed by evaluators and output writers."""

    timing: Dict[str, float] = field(default_factory=dict)
    bits: Dict[str, float] = field(default_factory=dict)
    # Keys are plain task labels (e.g. "seg", "edge").
    pred: Dict[str, Any] = field(default_factory=dict)
    gt: Dict[str, Any] = field(default_factory=dict)
    per_sample_records: List[Dict[str, Any]] = field(default_factory=list)
    # artifacts are for optional logging/saving only.
    artifacts: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# test_step protocol + default mixins (compat layer)
# =============================================================================

class TestStepProtocol(Protocol):
    def test_step(self, batch: "EvalBatch", *, ctx: Optional[Dict[str, Any]] = None) -> "StepOutput": ...


def _warn(msg: str) -> None:
    # Soft constraint only: never raise here.
    print(f"[eval-contract][warn] {msg}")


def _validate_step_output_soft(step_output: StepOutput) -> None:
    """Soft contract checks for StepOutput.timing/bits/records (warning-only)."""

    timing = step_output.timing or {}
    bits = step_output.bits or {}

    required_timing = {"total_enc_time", "total_dec_time"}
    missing = [k for k in sorted(required_timing) if k not in timing]
    if missing:
        _warn(f"missing timing keys: {missing}")
    for k, v in timing.items():
        if not isinstance(k, str):
            _warn(f"timing key is not str: {k!r}")
            continue
        if not isinstance(v, (int, float)):
            _warn(f"timing[{k!r}] is not number: {type(v)!r}")

    for k, v in bits.items():
        if not isinstance(k, str):
            _warn(f"bits key is not str: {k!r}")
            continue
        if not isinstance(v, (int, float)):
            _warn(f"bits[{k!r}] is not number: {type(v)!r}")
            continue
        if float(v) < 0:
            _warn(f"bits[{k!r}] is negative: {v}")

    recs = step_output.per_sample_records or []
    for i, r in enumerate(recs[:5]):
        if not isinstance(r, dict):
            _warn(f"record[{i}] is not dict: {type(r)!r}")
            continue
        for need in ("file", "quality", "total_enc_time", "total_dec_time", "bpp"):
            if need not in r:
                _warn(f"record[{i}] missing field {need!r}")

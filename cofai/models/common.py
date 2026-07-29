"""Exploratory feature-codec orchestration for non-DINO backbones."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from compressai.models.base import CompressionModel

from cofai.engine.registry import instantiate_class, register


_CONTEXT_KEY = "common_backbone_context"
_SIDE_STREAMS_KEY = "common_side_streams"


def _build_component(config_or_module: Any, *, name: str) -> nn.Module:
    if isinstance(config_or_module, nn.Module):
        return config_or_module
    if not isinstance(config_or_module, dict):
        raise TypeError(f"{name} must be an nn.Module or a config dict")
    return instantiate_class(config_or_module)


def _merge_unique(target: dict, source: dict, *, kind: str) -> None:
    duplicate = set(target).intersection(source)
    if duplicate:
        raise KeyError(f"duplicate {kind} keys: {sorted(duplicate)!r}")
    target.update(source)


def _bits_from_strings(strings: dict[str, list[list[bytes]]]) -> dict[str, float]:
    bits: dict[str, float] = {}
    for name, rows in strings.items():
        total = 0
        for row in rows:
            if not isinstance(row, list) or len(row) != 1:
                raise ValueError(
                    f"stream {name!r} must use canonical list[list[bytes]] rows"
                )
            if not isinstance(row[0], bytes):
                raise TypeError(f"stream {name!r} payloads must be bytes")
            total += len(row[0]) * 8
        bits[str(name)] = float(total)
    return bits


@register("CommonFeatureCodecModel")
class CommonFeatureCodecModel(CompressionModel):
    """Compose a split backbone, a feature codec, and optional task heads.

    This model intentionally exists in parallel with ``DinoFeatureCodecModel``.
    It is an exploratory integration point for algorithms whose encoder already
    performs feature restructuring, such as GPS token grouping.

    ``post_process`` is deliberately reserved as ``None``. The correct recovery
    operation depends on the downstream task and has not been established by the
    integrated GPS work; decoded compact features therefore pass through unchanged.
    """

    def __init__(
        self,
        backbone,
        codec,
        heads: dict | None = None,
        post_process=None,
        **kwargs,
    ):
        super().__init__()
        if post_process is not None:
            raise ValueError(
                "post_process is a reserved placeholder and must currently be null"
            )
        self.backbone = _build_component(backbone, name="backbone")
        self.codec = _build_component(codec, name="codec")
        self.post_process = None

        self.heads = nn.ModuleDict()
        for task, head in (heads or {}).items():
            self.heads[str(task)] = _build_component(head, name=f"heads.{task}")

    @staticmethod
    def _normalize_encoded(encoded: Any) -> dict[str, Any]:
        if isinstance(encoded, torch.Tensor):
            return {
                "h": encoded,
                "context": {},
                "strings": {},
                "bits": {},
            }
        if not isinstance(encoded, dict) or "h" not in encoded:
            raise TypeError(
                "backbone.encode must return a tensor or a dict containing `h`"
            )
        return {
            "h": encoded["h"],
            "context": dict(encoded.get("context") or {}),
            "strings": dict(encoded.get("strings") or {}),
            "bits": {
                str(name): float(value)
                for name, value in dict(encoded.get("bits") or {}).items()
            },
        }

    def _encode(self, x, **kwargs) -> dict[str, Any]:
        return self._normalize_encoded(self.backbone.encode(x, **kwargs))

    def _decode(self, h_hat, *, context, tasks, **kwargs) -> dict[str, Any]:
        decoded = self.backbone.decode(
            h_hat,
            context=context,
            tasks=list(tasks),
            **kwargs,
        )
        if not isinstance(decoded, dict):
            if len(tasks) != 1:
                raise TypeError(
                    "backbone.decode must return a task dict when multiple tasks "
                    "are requested"
                )
            decoded = {str(tasks[0]): decoded}

        task_outputs = dict(decoded)
        for task, head in self.heads.items():
            if task in task_outputs and task in tasks:
                task_outputs[task] = head(task_outputs[task])
        return task_outputs

    @staticmethod
    def _codec_args(encoded: dict[str, Any], qp: Any) -> dict[str, Any]:
        context = encoded["context"]
        return {
            "token_res": context.get("token_res"),
            "qp": qp,
        }

    def forward(self, x, qp=0, tasks=None, **kwargs):
        _, task_outputs = self.forward_test(
            x,
            qp=qp,
            tasks=list(tasks or []),
            **kwargs,
        )
        return task_outputs

    def forward_test(self, x, qp=0, tasks=None, **kwargs):
        tasks = list(tasks or [])
        encoded = self._encode(x, **kwargs)

        start = time.time()
        codec_out = self.codec(encoded["h"], **self._codec_args(encoded, qp))
        elapsed = time.time() - start
        self._codec_time = {
            "codec_enc_time": elapsed / 2.0,
            "codec_dec_time": elapsed / 2.0,
        }
        if "h_hat" not in codec_out:
            raise KeyError("codec.forward must return `h_hat`")

        side_bits = dict(encoded["bits"])
        if "bits" in codec_out:
            bits = {
                str(name): float(value)
                for name, value in dict(codec_out["bits"]).items()
            }
            _merge_unique(bits, side_bits, kind="rate")
            coded_unit = {"bits": bits}
        elif "likelihoods" in codec_out and not side_bits:
            coded_unit = {"likelihoods": dict(codec_out["likelihoods"])}
        else:
            raise ValueError(
                "mixed side-information bits require a codec.forward result "
                "using the `bits` contract"
            )

        h_hat = codec_out["h_hat"]
        task_outputs = self._decode(
            h_hat,
            context=encoded["context"],
            tasks=tasks,
            **kwargs,
        )
        return coded_unit, task_outputs

    def compress(self, x, qp=0, tasks=None, **kwargs):
        encoded = self._encode(x, **kwargs)
        start = time.time()
        coded_unit = self.codec.compress(
            encoded["h"],
            **self._codec_args(encoded, qp),
        )
        self._codec_time = {"codec_enc_time": time.time() - start}

        strings = dict(coded_unit.get("strings") or {})
        side_strings = dict(encoded["strings"])
        _merge_unique(strings, side_strings, kind="stream")
        # Validate the exact shape consumed by the unmodified eval bit counter.
        _bits_from_strings(strings)

        pstate = dict(coded_unit.get("pstate") or {})
        for reserved in (_CONTEXT_KEY, _SIDE_STREAMS_KEY):
            if reserved in pstate:
                raise KeyError(f"codec pstate uses reserved key {reserved!r}")
        pstate[_CONTEXT_KEY] = encoded["context"]
        pstate[_SIDE_STREAMS_KEY] = tuple(side_strings)
        return {"strings": strings, "pstate": pstate}

    def decompress(self, coded_unit, tasks=None, **kwargs):
        tasks = list(tasks or [])
        strings = dict(coded_unit["strings"])
        pstate = dict(coded_unit["pstate"])
        context = pstate.pop(_CONTEXT_KEY)
        side_streams = tuple(pstate.pop(_SIDE_STREAMS_KEY))
        codec_strings = {
            name: value for name, value in strings.items() if name not in side_streams
        }

        start = time.time()
        decoded = self.codec.decompress(
            strings=codec_strings,
            pstate=pstate,
        )
        elapsed = time.time() - start
        self._codec_time = getattr(self, "_codec_time", {})
        self._codec_time["codec_dec_time"] = elapsed
        if "h_hat" not in decoded:
            raise KeyError("codec.decompress must return `h_hat`")

        h_hat = decoded["h_hat"]
        return self._decode(
            h_hat,
            context=context,
            tasks=tasks,
            **kwargs,
        )

"""Exploratory feature-codec orchestration for non-DINO backbones."""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from compressai.models.base import CompressionModel

from cofai.engine.registry import instantiate_class, register


_CODEC_PSTATE_KEYS = "common_codec_pstate_keys"
_CODEC_STREAM_KEYS = "common_codec_stream_keys"


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

    A standard backbone encoder may return only its feature tensor. An adapter that
    also produces auxiliary DU syntax may return ``h`` together with the documented
    ``strings``, ``pstate``, and optional ``meta`` fields. The public compressed
    result remains a regular CodedUnit; this does not define a second backbone
    protocol.

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
            return {"h": encoded, "strings": {}, "pstate": {}, "meta": {}}
        if not isinstance(encoded, dict) or "h" not in encoded:
            raise TypeError(
                "backbone.encode must return a tensor or a dict containing `h`"
            )
        return {
            "h": encoded["h"],
            "strings": dict(encoded.get("strings") or {}),
            "pstate": dict(encoded.get("pstate") or {}),
            "meta": dict(encoded.get("meta") or {}),
        }

    def _encode(self, x, **kwargs) -> dict[str, Any]:
        return self._normalize_encoded(self.backbone.encode(x, **kwargs))

    def _decode(
        self,
        h_hat,
        *,
        pstate,
        meta,
        tasks,
        **kwargs,
    ) -> dict[str, Any]:
        decode_kwargs = dict(kwargs)
        if pstate:
            decode_kwargs["pstate"] = pstate
        if meta:
            decode_kwargs["meta"] = meta
        decoded = self.backbone.decode(
            h_hat,
            tasks=list(tasks),
            **decode_kwargs,
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
        return {
            "token_res": encoded["pstate"].get("token_res"),
            "qp": qp,
        }

    def _decode_device(self, requested_device=None) -> torch.device:
        if requested_device is not None:
            return torch.device(requested_device)
        for parameter in self.parameters():
            return parameter.device
        for buffer in self.buffers():
            return buffer.device
        return torch.device("cpu")

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

        auxiliary_bits = _bits_from_strings(encoded["strings"])
        if "bits" in codec_out:
            bits = {
                str(name): float(value)
                for name, value in dict(codec_out["bits"]).items()
            }
            _merge_unique(bits, auxiliary_bits, kind="rate")
            coded_unit = {"bits": bits}
        elif "likelihoods" in codec_out and not auxiliary_bits:
            coded_unit = {"likelihoods": dict(codec_out["likelihoods"])}
        else:
            raise ValueError(
                "auxiliary stream bits require a codec.forward result "
                "using the `bits` contract"
            )

        h_hat = codec_out["h_hat"]
        task_outputs = self._decode(
            h_hat,
            pstate=encoded["pstate"],
            meta=encoded["meta"],
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
        codec_stream_keys = tuple(strings)
        _merge_unique(strings, encoded["strings"], kind="stream")
        # Validate the exact shape consumed by the unmodified eval bit counter.
        _bits_from_strings(strings)

        pstate = dict(coded_unit.get("pstate") or {})
        codec_pstate_keys = tuple(pstate)
        for reserved in (_CODEC_PSTATE_KEYS, _CODEC_STREAM_KEYS):
            if reserved in pstate or reserved in encoded["pstate"]:
                raise KeyError(f"component pstate uses reserved key {reserved!r}")
        _merge_unique(pstate, encoded["pstate"], kind="state")
        pstate[_CODEC_PSTATE_KEYS] = codec_pstate_keys
        pstate[_CODEC_STREAM_KEYS] = codec_stream_keys

        result = {"strings": strings, "pstate": pstate}
        if encoded["meta"]:
            result["meta"] = encoded["meta"]
        return result

    def decompress(self, coded_unit, tasks=None, **kwargs):
        tasks = list(tasks or [])
        strings = dict(coded_unit["strings"])
        pstate = dict(coded_unit["pstate"])
        codec_pstate_keys = tuple(pstate.pop(_CODEC_PSTATE_KEYS))
        codec_stream_keys = tuple(pstate.pop(_CODEC_STREAM_KEYS))
        codec_strings = {
            name: strings[name] for name in codec_stream_keys
        }
        codec_pstate = {name: pstate.pop(name) for name in codec_pstate_keys}
        meta = dict(coded_unit.get("meta") or {})

        start = time.time()
        decoded = self.codec.decompress(
            strings=codec_strings,
            pstate=codec_pstate,
            device=self._decode_device(kwargs.get("device")),
        )
        elapsed = time.time() - start
        self._codec_time = getattr(self, "_codec_time", {})
        self._codec_time["codec_dec_time"] = elapsed
        if "h_hat" not in decoded:
            raise KeyError("codec.decompress must return `h_hat`")

        h_hat = decoded["h_hat"]
        return self._decode(
            h_hat,
            pstate=pstate,
            meta=meta,
            tasks=tasks,
            **kwargs,
        )

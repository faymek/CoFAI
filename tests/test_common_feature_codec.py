from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cofai.engine.run_eval import _bits_from_coded_unit
from cofai.latent_codecs import RawDtypeCodec
from cofai.models import CommonFeatureCodecModel


class _SplitBackbone(nn.Module):
    def encode(self, x, **kwargs):
        return {
            "h": x,
            "pstate": {"bias": 0.25},
            "meta": {"order": torch.arange(x.shape[0])},
            "strings": {"selection_map": [[b"map"] for _ in range(x.shape[0])]},
        }

    def decode(self, h, *, pstate, meta, tasks, **kwargs):
        return {
            "reid": h + pstate["bias"],
            "order": meta["order"],
        }


class _PlainBackbone(nn.Module):
    def encode(self, x):
        return x

    def decode(self, h, *, tasks):
        return {"reid": h}


@pytest.mark.parametrize(
    ("wire_dtype", "bytes_per_value"),
    [
        ("float16", 2),
        ("bfloat16", 2),
        ("float8_e4m3fn", 1),
        ("float8_e5m2", 1),
    ],
)
def test_raw_dtype_codec_round_trip_uses_real_wire_bytes(
    wire_dtype,
    bytes_per_value,
):
    h = torch.tensor(
        [[[1.0001, -2.0001], [3.14159, 4.125]]],
        dtype=torch.float32,
    )
    codec = RawDtypeCodec(dtype=wire_dtype)

    coded = codec.compress(h)

    assert set(coded["strings"]) == {"feature"}
    assert "raw_output_device" not in coded["pstate"]
    assert "raw_token_res" not in coded["pstate"]
    assert coded["pstate"]["raw_wire_dtype"] == wire_dtype
    assert isinstance(coded["strings"]["feature"][0][0], bytes)
    assert len(coded["strings"]["feature"][0][0]) == h.numel() * bytes_per_value
    h_hat = codec.decompress(**coded)["h_hat"]
    torch.testing.assert_close(
        h_hat,
        h.to(getattr(torch, wire_dtype)).to(torch.float32),
        rtol=0,
        atol=0,
    )


def test_common_model_flattens_codec_and_selection_streams():
    h = torch.randn(2, 5, 3)
    model = CommonFeatureCodecModel(
        backbone=_SplitBackbone(),
        codec=RawDtypeCodec(dtype="float16"),
        post_process=None,
    )

    coded = model.compress(h, tasks=["reid"])

    assert set(coded) == {"strings", "pstate", "meta"}
    assert set(coded["strings"]) == {"feature", "selection_map"}
    assert "codec" not in coded["strings"]
    assert "pre_process" not in coded["strings"]
    assert set(coded["meta"]) == {"order"}
    assert "bias" in coded["pstate"]
    bits = _bits_from_coded_unit(coded)
    assert bits["feature"] == h.numel() * 16
    assert bits["selection_map"] == h.shape[0] * len(b"map") * 8

    decoded = model.decompress(coded, tasks=["reid"])
    expected = h.to(torch.float16).to(h.dtype) + 0.25
    torch.testing.assert_close(decoded["reid"], expected, rtol=0, atol=0)
    torch.testing.assert_close(decoded["order"], torch.arange(h.shape[0]))


def test_common_model_estimated_path_counts_all_flat_streams():
    h = torch.randn(2, 5, 3)
    model = CommonFeatureCodecModel(
        backbone=_SplitBackbone(),
        codec=RawDtypeCodec(dtype="float16"),
    )

    coded, decoded = model.forward_test(h, tasks=["reid"])

    assert coded["bits"] == {
        "feature": float(h.numel() * 16),
        "selection_map": float(h.shape[0] * len(b"map") * 8),
    }
    assert decoded["reid"].shape == h.shape


def test_common_model_accepts_documented_plain_backbone_protocol():
    h = torch.randn(1, 2, 3)
    model = CommonFeatureCodecModel(
        backbone=_PlainBackbone(),
        codec=RawDtypeCodec(dtype="float16"),
    )

    coded, decoded = model.forward_test(h, tasks=["reid"])

    assert coded["bits"] == {"feature": float(h.numel() * 16)}
    torch.testing.assert_close(
        decoded["reid"],
        h.to(torch.float16).to(h.dtype),
        rtol=0,
        atol=0,
    )


def test_common_model_leaves_post_process_unimplemented():
    with pytest.raises(ValueError, match="reserved placeholder"):
        CommonFeatureCodecModel(
            backbone=_SplitBackbone(),
            codec=RawDtypeCodec(dtype="float16"),
            post_process={"type": "future.Restore"},
        )

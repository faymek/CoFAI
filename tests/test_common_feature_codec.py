from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cofai.engine.run_eval import _bits_from_coded_unit
from cofai.latent_codecs import FP16Codec
from cofai.models import CommonFeatureCodecModel


class _SplitBackbone(nn.Module):
    def encode(self, x, **kwargs):
        return {
            "h": x,
            "context": {"bias": 0.25},
            "strings": {"selection_map": [[b"map"] for _ in range(x.shape[0])]},
            "bits": {"selection_map": float(x.shape[0] * len(b"map") * 8)},
        }

    def decode(self, h, *, context, tasks, **kwargs):
        return {"reid": h + context["bias"]}


def test_fp16_codec_round_trip_uses_real_wire_bytes():
    h = torch.tensor(
        [[[1.0001, -2.0001], [3.14159, 4.125]]],
        dtype=torch.float32,
    )
    codec = FP16Codec()

    coded = codec.compress(h)

    assert set(coded["strings"]) == {"fp16"}
    assert isinstance(coded["strings"]["fp16"][0][0], bytes)
    assert len(coded["strings"]["fp16"][0][0]) == h.numel() * 2
    h_hat = codec.decompress(**coded)["h_hat"]
    torch.testing.assert_close(
        h_hat,
        h.to(torch.float16).to(torch.float32),
        rtol=0,
        atol=0,
    )


def test_common_model_flattens_codec_and_selection_streams():
    h = torch.randn(2, 5, 3)
    model = CommonFeatureCodecModel(
        backbone=_SplitBackbone(),
        codec=FP16Codec(),
        post_process=None,
    )

    coded = model.compress(h, tasks=["reid"])

    assert set(coded["strings"]) == {"fp16", "selection_map"}
    assert "codec" not in coded["strings"]
    assert "pre_process" not in coded["strings"]
    bits = _bits_from_coded_unit(coded)
    assert bits["fp16"] == h.numel() * 16
    assert bits["selection_map"] == h.shape[0] * len(b"map") * 8

    decoded = model.decompress(coded, tasks=["reid"])
    expected = h.to(torch.float16).to(h.dtype) + 0.25
    torch.testing.assert_close(decoded["reid"], expected, rtol=0, atol=0)


def test_common_model_estimated_path_counts_all_flat_streams():
    h = torch.randn(2, 5, 3)
    model = CommonFeatureCodecModel(
        backbone=_SplitBackbone(),
        codec=FP16Codec(),
    )

    coded, decoded = model.forward_test(h, tasks=["reid"])

    assert coded["bits"] == {
        "fp16": float(h.numel() * 16),
        "selection_map": float(h.shape[0] * len(b"map") * 8),
    }
    assert decoded["reid"].shape == h.shape


def test_common_model_leaves_post_process_unimplemented():
    with pytest.raises(ValueError, match="reserved placeholder"):
        CommonFeatureCodecModel(
            backbone=_SplitBackbone(),
            codec=FP16Codec(),
            post_process={"type": "future.Restore"},
        )

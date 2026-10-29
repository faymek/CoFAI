from pathlib import Path
import runpy
import sys
from types import ModuleType

import torch


MODULE = Path(__file__).resolve().parents[1] / "cofai/layers/dcvc_cuda_inference.py"
EXPORTS = (
    "process_with_mask_cuda",
    "combine_for_reading_2x_cuda",
    "restore_y_2x_cuda",
    "restore_y_4x_cuda",
    "build_index_dec_cuda",
    "round_and_to_int8_cuda",
    "clamp_reciprocal_with_quant_cuda",
    "bias_quant_cuda",
    "add_and_multiply_cuda",
    "bias_pixel_shuffle_8_cuda",
    "replicate_pad_cuda",
    "build_index_enc_cuda",
    "DepthConvProxy",
    "SubpelConv2xProxy",
)


def test_imports_installed_top_level_extension(monkeypatch):
    extension = ModuleType("inference_extensions_cuda")
    for name in EXPORTS:
        setattr(extension, name, object())
    monkeypatch.setitem(sys.modules, "inference_extensions_cuda", extension)

    namespace = runpy.run_path(str(MODULE))

    assert namespace["CUSTOMIZED_CUDA_INFERENCE"] is True
    for name in EXPORTS:
        assert namespace[name] is getattr(extension, name)


def test_missing_extension_keeps_pytorch_path(monkeypatch):
    monkeypatch.setitem(sys.modules, "inference_extensions_cuda", None)

    namespace = runpy.run_path(str(MODULE))

    assert namespace["CUSTOMIZED_CUDA_INFERENCE"] is False
    values = torch.tensor([-200.0, 0.6, 200.0])
    _, quantized, reconstructed, _ = namespace["process_with_mask"](
        values, torch.ones_like(values), torch.zeros_like(values),
        torch.ones_like(values), None,
    )
    expected = torch.tensor([-128.0, 1.0, 127.0])
    torch.testing.assert_close(quantized, expected)
    torch.testing.assert_close(reconstructed, expected)

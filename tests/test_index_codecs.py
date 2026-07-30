from __future__ import annotations

import torch

from cofai.engine.registry import instantiate_class
from cofai.index_codecs import UniformIndexCodec
from cofai.latent_codecs import FixedGaussianCodec


def test_uniform_index_codec_reports_uniform_likelihoods():
    indices = torch.tensor([[0, 1], [2, 3]])
    codec = UniformIndexCodec(alphabet_size=4)

    output = codec(indices)

    assert output["tokens"] is indices
    torch.testing.assert_close(
        output["likelihoods"]["t"],
        torch.full(indices.shape, 0.25),
    )


def test_uniform_index_codec_can_be_built_from_qualified_config():
    codec = instantiate_class(
        {
            "type": "cofai.index_codecs.UniformIndexCodec",
            "alphabet_size": 8,
        }
    )

    assert isinstance(codec, UniformIndexCodec)


def test_fixed_gaussian_codec_lives_in_latent_codec_namespace():
    assert FixedGaussianCodec.__module__ == "cofai.latent_codecs.fixed_gaussian"

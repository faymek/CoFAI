from types import SimpleNamespace
import subprocess
import sys

import torch
from torch import nn

from cofai.backbone.vdt import VDTBackbone
from cofai.heads.vdt_reid import VDTReIDHead
from cofai.latent_codecs.raw_dtype import RawDtypeCodec
from cofai.models import CommonFeatureCodecModel
from examples.protrac.calibration import ProTRACStats
from examples.protrac.semseg.model import ProTRACSegBackbone


def test_evaluation_import_preserves_torch_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; "
            "torch.backends.cudnn.deterministic = False; "
            "torch.set_num_threads(2); "
            "import examples.protrac.evaluation; "
            "assert not torch.backends.cudnn.deterministic; "
            "assert torch.get_num_threads() == 2",
        ],
        check=True,
    )


def test_sample_statistics_update_and_reset():
    source = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    stats = ProTRACStats.create(source)
    weights = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    observations = torch.tensor([[2.0, 4.0], [8.0, 10.0]])
    torch.testing.assert_close(stats.shift(weights), torch.zeros(2, 2))
    stats.update(observations, weights)
    torch.testing.assert_close(stats.mean, torch.tensor([[4.0, 6.0], [8.0, 10.0]]))
    torch.testing.assert_close(stats.count, torch.tensor([1.5, 0.5]))
    stats.reset()
    torch.testing.assert_close(stats.mean, source)
    torch.testing.assert_close(stats.count, torch.zeros(2))


def test_image_statistics_use_one_observation_per_region():
    stats = ProTRACStats.create(torch.zeros(2, 2), prior=2)
    observations = torch.tensor([[3.0, 6.0], [3.0, 6.0], [30.0, 60.0]])
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    stats.update(observations, weights, min_mass=2, per_image=True)
    torch.testing.assert_close(stats.mean, torch.tensor([[1.0, 2.0], [0.0, 0.0]]))
    torch.testing.assert_close(stats.count, torch.tensor([3.0, 2.0]))


def test_vdt_applies_view_subtraction_at_each_block():
    backbone = VDTBackbone(image_size=(16, 16), dim=12, slot=11)
    backbone.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
    backbone.norm = nn.Identity()
    with torch.no_grad():
        backbone.patch_embed.proj.weight.zero_()
        backbone.patch_embed.proj.bias.zero_()
        backbone.cls_token.fill_(25)
        backbone.view_token.fill_(2)
        prefix = backbone.encode(torch.zeros(1, 3, 16, 16))
        torch.testing.assert_close(prefix[:, 0], torch.full((1, 12), 3.0))
        final = backbone.final_tokens(prefix.clone())
        torch.testing.assert_close(final[:, 0], torch.ones(1, 12))
        torch.testing.assert_close(final[:, 1], torch.full((1, 12), 2.0))


def test_vdt_common_codec_roundtrip():
    torch.manual_seed(7)
    backbone = VDTBackbone(image_size=(32, 16), dim=12, slot=11)
    model = CommonFeatureCodecModel(
        backbone=backbone,
        codec=RawDtypeCodec(dtype="float16"),
        heads={"reid": VDTReIDHead(dim=12)},
        post_process=None,
    ).eval()
    with torch.inference_mode():
        images = torch.rand(2, 3, 32, 16)
        _, expected = model.forward_test(images, tasks=["reid"])
        coded = model.compress(images, tasks=["reid"])
        assert all(isinstance(row[0], bytes) for row in coded["strings"]["feature"])
        actual = model.decompress(coded, tasks=["reid"])
    torch.testing.assert_close(actual["reid"], expected["reid"], rtol=0, atol=0)


def test_semseg_codec_preserves_grid_and_final_features():
    tokens = torch.arange(44, dtype=torch.float16).reshape(1, 11, 4)
    backbone = ProTRACSegBackbone.__new__(ProTRACSegBackbone)
    nn.Module.__init__(backbone)
    backbone.dino = SimpleNamespace(model=SimpleNamespace(num_prefix_tokens=5))
    backbone.encode = lambda image: {"h": tokens, "pstate": {"patch_hw": (2, 3)}}
    model = CommonFeatureCodecModel(
        backbone=backbone,
        codec=RawDtypeCodec(dtype="float16"),
        heads={},
        post_process=None,
    )
    with torch.inference_mode():
        coded = model.compress(torch.empty(1))
        decoded = model.decompress(coded, tasks=["semseg"])
    expected = tokens[:, 5:].transpose(1, 2).reshape(1, 4, 2, 3)
    torch.testing.assert_close(decoded["semseg"][0], expected, rtol=0, atol=0)

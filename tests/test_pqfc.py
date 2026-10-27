"""PQFC DINOv3 online plans: transform uses ORFC-2446 release; noR uses R=I artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cofai.latent_codecs import OrthoRotationFeatureCodec

PLAN_ROOT = Path(__file__).resolve().parents[1] / "examples/pqfc/plan/dinov3"
RELEASE_NAMES = [
    "blk23_K16_e32.npz",
    "blk23_K256_e32.npz",
    "blk23_K1024_e32.npz",
    "blk23_K512_e16.npz",
    "blk23_K1024_e16.npz",
    "blk23_K64_e8.npz",
    "blk23_K256_e8.npz",
    "blk23_K512_e8.npz",
]


def _plans(suffix: str) -> list[Path]:
    return sorted(PLAN_ROOT.glob(f"*{suffix}*.yaml"))


def test_pqfc_dinov3_plans_exist_and_use_public_codec():
    transform = [p for p in _plans("__PQFC__") if "__PQFC-noR__" not in p.name]
    nor = _plans("__PQFC-noR__")
    assert len(transform) == 2
    assert len(nor) == 2
    names = {p.name for p in transform}
    assert "ade20k-val__dinov3-vitl16-slot24__PQFC__semseg.yaml" in names
    assert "nyuv2-val__dinov3-vitl16-slot24__PQFC__depth.yaml" in names

    for plan_path in [*transform, *nor]:
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        assert plan["model"]["type"] == "DinoFeatureCodecModel"
        assert plan["model"]["dino_codec"]["type"] == (
            "cofai.latent_codecs.OrthoRotationFeatureCodec"
        )
        assert plan["model"]["dino_codec"]["type"].split(".")[-1] == (
            OrthoRotationFeatureCodec.__name__
        )
        expected = 7 if "__PQFC-noR__" in plan_path.name else 8
        assert len(plan["multi_run"]) == expected


def test_pqfc_transform_weights_are_orfc_2446_release():
    for plan_path in _plans("__PQFC__"):
        if "__PQFC-noR__" in plan_path.name:
            continue
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        default = plan["model"]["dino_codec"]["orfc_weights_path"]
        assert "weights/orfc_2446/dinov3_vitl16_ori/" in default
        names = [
            Path(overrides["model"]["dino_codec"]["orfc_weights_path"]).name
            for overrides in plan["multi_run"].values()
        ]
        assert names == RELEASE_NAMES


@pytest.mark.parametrize("task", ["semseg", "depth"])
def test_pqfc_transform_plans_match_m2446_release(task):
    [pqfc_path] = PLAN_ROOT.glob(f"*slot24__PQFC__{task}.yaml")
    orfc_root = Path(__file__).resolve().parents[1] / "examples/orfc_2446/plan/dinov3"
    [orfc_path] = orfc_root.glob(f"*slot24__ORFC__{task}.yaml")
    pqfc = yaml.safe_load(pqfc_path.read_text())
    orfc = yaml.safe_load(orfc_path.read_text())
    for plan in (pqfc, orfc):
        plan.pop("name")
        plan.pop("description")
    assert pqfc == orfc


def test_pqfc_nor_weights_use_canonical_dir():
    for plan_path in _plans("__PQFC-noR__"):
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        default = plan["model"]["dino_codec"]["orfc_weights_path"]
        assert "weights/pqfc/dinov3_vitl16_noR/" in default
        assert default.endswith("_no_transform.npz")
        names = [
            Path(overrides["model"]["dino_codec"]["orfc_weights_path"]).name
            for overrides in plan["multi_run"].values()
        ]
        assert names == [
            name.replace(".npz", "_no_transform.npz") for name in RELEASE_NAMES[:-1]
        ]
        for overrides in plan["multi_run"].values():
            path = overrides["model"]["dino_codec"]["orfc_weights_path"]
            assert "weights/pqfc/dinov3_vitl16_noR/" in path
            assert path.endswith("_no_transform.npz")


def test_pqfc_train_uses_example_local_training_and_public_ops():
    train_src = (
        Path(__file__).resolve().parents[1]
        / "examples/pqfc/offline/train_pqfc_dinov3.py"
    ).read_text(encoding="utf-8")
    assert "from examples.orfc_2446.offline.soft_pq import" in train_src
    assert "from examples.orfc_2446.offline.artifacts import" in train_src
    assert "from cofai.ops.orfc import" in train_src
    assert "cofai.entropy_models.soft_pq" not in train_src
    assert "cofai.entropy_models.orfc_model" not in train_src
    assert "cofai.entropy_models.soft_pq_export" not in train_src

    from examples.orfc_2446.offline.artifacts import save_codec_npz
    from examples.orfc_2446.offline.soft_pq import (
        OrthogonalTransform,
        train_soft_pq,
    )

    assert callable(train_soft_pq)
    assert callable(save_codec_npz)
    assert OrthogonalTransform.__name__ == "OrthogonalTransform"


def _write_pqfc_artifact(path, use_transform):
    rng = np.random.default_rng(7)
    payload = {
        "R": np.eye(4, dtype=np.float32),
        "codebooks": rng.standard_normal((2, 4, 2), dtype=np.float32),
        "pmf": np.full((2, 4), 0.25, dtype=np.float32),
        "norm_mode": np.asarray("split_reg_cls_patch"),
        "n_prefix": np.asarray(3, dtype=np.int32),
    }
    if use_transform is not None:
        payload["use_transform"] = np.bool_(use_transform)
    np.savez(path, **payload)
    return path


@pytest.mark.parametrize("flag", [None, True, False])
def test_pqfc_transform_metadata_controls_runtime(tmp_path, monkeypatch, flag):
    from cofai.latent_codecs.orfc import load_orfc_artifact
    from examples.orfc_2446.offline.artifacts import load_softpq_npz
    from examples.pqfc.run_pqfc_dinov3 import _load_npz_codec

    path = _write_pqfc_artifact(tmp_path / "codec.npz", flag)
    enabled = flag is not False
    assert load_orfc_artifact(path)["use_transform"] is enabled
    assert load_softpq_npz(path)["use_transform"] is enabled
    codec, metadata = _load_npz_codec(path, torch.device("cpu"))
    assert codec.use_transform is enabled
    assert metadata["use_transform"] is enabled
    assert (codec.R is not None) is enabled
    assert ("R" in codec.state_dict()) is enabled

    calls = []
    matmul = torch.Tensor.__matmul__

    def track(left, right):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return matmul(left, right)

    monkeypatch.setattr(torch.Tensor, "__matmul__", track)
    tokens = torch.randn(1, 8, 4)
    coded = codec.compress(tokens)
    decoded = codec.decompress(**coded)
    assert calls == ([((8, 4), (4, 4))] * 2 if enabled else [])
    assert torch.equal(decoded["h_hat"], codec(tokens)["h_hat"])


def test_pqfc_nor_preserves_identity_rotation_streams_and_reconstruction(tmp_path):
    baseline = OrthoRotationFeatureCodec(
        orfc_weights_path=str(_write_pqfc_artifact(tmp_path / "legacy.npz", None))
    )
    nor = OrthoRotationFeatureCodec(
        orfc_weights_path=str(_write_pqfc_artifact(tmp_path / "nor.npz", False))
    )
    tokens = torch.randn(2, 8, 4)
    before = baseline.compress(tokens)
    after = nor.compress(tokens)
    assert before == after
    before_reconstruction = baseline.decompress(**before)["h_hat"]
    after_reconstruction = nor.decompress(**after)["h_hat"]
    assert torch.equal(before_reconstruction, after_reconstruction)
    assert torch.equal(baseline(tokens)["h_hat"], nor(tokens)["h_hat"])

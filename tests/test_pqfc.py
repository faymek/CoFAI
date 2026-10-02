"""PQFC DINOv3 online plans: transform uses ORFC-2446 release; noR uses R=I artifacts."""

from __future__ import annotations

from pathlib import Path

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


def test_pqfc_train_imports_orfc_2446_not_entropy_models():
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

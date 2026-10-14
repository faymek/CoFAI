"""ORFC-2446 runtime, artifact, and proposal-plan invariants."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cofai.entropy_models import StaticCategoricalEntropyModel
from cofai.latent_codecs import OrthoRotationFeatureCodec
from cofai.latent_codecs.orfc_normalization import (
    denormalize_orfc_features,
    normalize_orfc_features,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_ROOT = ROOT / "examples" / "orfc_2446" / "plan"
SCRIPTS = ROOT / "examples" / "orfc_2446" / "dinov2" / "scripts"
sys.path.insert(0, str(SCRIPTS))
collect_all = importlib.import_module("collect_softpq_results").collect_all
build_rows = importlib.import_module("compare_plot_ours_vs_npz").build_rows
discover_plan_results = importlib.import_module("plan_results").discover_plan_results


def _write_artifact(
    path: Path,
    *,
    norm_mode: str = "per_image",
    n_prefix: int = 0,
    include_pmf: bool = True,
) -> Path:
    rng = np.random.default_rng(7)
    payload = {
        "R": np.eye(4, dtype=np.float32),
        "codebooks": rng.standard_normal((2, 4, 2), dtype=np.float32),
        "norm_mode": np.asarray(norm_mode),
        "n_prefix": np.asarray(n_prefix, dtype=np.int32),
    }
    if include_pmf:
        payload["pmf"] = np.full((2, 4), 0.25, dtype=np.float32)
    np.savez(path, **payload)
    return path


def _stream_bytes(coded: dict, name: str) -> int:
    return sum(len(stream[0]) for stream in coded["strings"][name])


def test_orfc_codec_obeys_the_latent_codec_contract(tmp_path):
    codec = OrthoRotationFeatureCodec(
        orfc_weights_path=str(_write_artifact(tmp_path / "orfc.npz"))
    )
    tokens = torch.randn(1, 5, 4)

    estimated = codec(tokens)
    coded = codec.compress(tokens)
    decoded = codec.decompress(**coded)

    assert set(estimated) == {"h_hat", "likelihoods"}
    assert set(estimated["likelihoods"]) == {"orfc", "orfc_stats"}
    assert set(coded) == {"strings", "pstate"}
    assert torch.equal(estimated["h_hat"], decoded["h_hat"])
    assert _stream_bytes(coded, "orfc_stats") == 8
    assert isinstance(codec.entropy_model, StaticCategoricalEntropyModel)


def test_static_categorical_entropy_model_preserves_orfc_streams():
    entropy_model = StaticCategoricalEntropyModel(
        torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
    )
    labels = torch.tensor(
        [
            [0, 1, 0],
            [3, 1, 0],
        ]
    )

    likelihoods = entropy_model(labels)
    streams = entropy_model.compress(labels)
    decoded = entropy_model.decompress(streams, num_samples=labels.shape[1])

    torch.testing.assert_close(
        likelihoods,
        torch.tensor(
            [
                [0.1, 0.2, 0.1],
                [0.1, 0.3, 0.4],
            ]
        ),
    )
    assert [stream.hex() for stream in streams] == [
        "b80223f9f9000000",
        "89f798aa29000000",
    ]
    assert torch.equal(decoded, labels)


def test_orfc_split_normalization_is_part_of_the_real_bitstream(tmp_path):
    codec = OrthoRotationFeatureCodec(
        orfc_weights_path=str(
            _write_artifact(
                tmp_path / "split.npz",
                norm_mode="split_reg_cls_patch",
                n_prefix=3,
            )
        )
    )
    coded = codec.compress(torch.randn(1, 8, 4))

    # Two groups of mean/std, serialized as four float32 values.
    assert _stream_bytes(coded, "orfc_stats") == 16


@pytest.mark.parametrize(
    ("mode", "n_prefix", "stats_per_image"),
    [
        ("per_image", 0, 1),
        ("per_token_ln", 0, 8),
        ("split_cls_patch", 3, 2),
        ("split_reg_cls_patch", 3, 2),
    ],
)
def test_orfc_training_and_runtime_share_normalization(mode, n_prefix, stats_per_image):
    tokens = torch.randn(2, 8, 4)

    normalized, mean, std = normalize_orfc_features(
        tokens,
        mode=mode,
        n_prefix=n_prefix,
    )
    reconstructed = denormalize_orfc_features(
        normalized,
        mean,
        std,
        mode=mode,
        n_prefix=n_prefix,
    )

    torch.testing.assert_close(reconstructed, tokens)
    assert mean.shape == std.shape == (2, stats_per_image)


def test_dinov3_extraction_reuses_plan_center_padding():
    from examples.orfc_2446.dinov3.offline.extract_features_dinov3 import (
        build_ade_transform,
    )

    image = np.ones((3, 5, 3), dtype=np.float32)
    padded = build_ade_transform(4)({"img": image})["img"]

    assert padded.shape == (3, 4, 8)
    torch.testing.assert_close(padded[:, :3, 1:6], torch.ones(3, 3, 5))
    assert torch.count_nonzero(padded[:, :, :1]) == 0
    assert torch.count_nonzero(padded[:, :, 6:]) == 0


def test_orfc_requires_a_complete_release_artifact(tmp_path):
    path = _write_artifact(tmp_path / "missing-pmf.npz", include_pmf=False)
    with pytest.raises(KeyError, match="pmf"):
        OrthoRotationFeatureCodec(orfc_weights_path=str(path))


def test_orfc_reuses_the_existing_public_codec_only():
    import cofai.entropy_models as entropy_models
    import cofai.latent_codecs as latent_codecs
    import cofai.ops as ops

    assert latent_codecs.OrthoRotationFeatureCodec is OrthoRotationFeatureCodec
    assert not hasattr(latent_codecs, "SoftPQFeatureCodec")
    assert entropy_models.StaticCategoricalEntropyModel is StaticCategoricalEntropyModel
    assert not hasattr(entropy_models, "FeatureCodec")
    assert not hasattr(entropy_models, "train_soft_pq")
    assert not hasattr(entropy_models, "batched_assign")
    assert not hasattr(entropy_models, "learn_orfc_rotation")
    assert not hasattr(ops, "batched_assign")


def test_proposal_plans_are_local_and_expand_all_quality_points():
    dinov2_plans = sorted((PLAN_ROOT / "dinov2").glob("*.yaml"))
    dinov3_plans = sorted((PLAN_ROOT / "dinov3").glob("*.yaml"))
    assert len(dinov2_plans) == 14
    assert len(dinov3_plans) == 2

    quality_count = 0
    for plan_path in [*dinov2_plans, *dinov3_plans]:
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        assert "__ORFC__" in plan_path.name
        assert "__ORFC__" in plan["name"]
        assert plan["model"]["dino_codec"]["type"] == (
            "cofai.latent_codecs.OrthoRotationFeatureCodec"
        )
        assert plan["multi_run"]
        assert all(isinstance(quality, int) for quality in plan["multi_run"])
        quality_count += len(plan["multi_run"])
    assert quality_count == 106


def test_dinov3_release_packager_matches_plan_quality_mapping(tmp_path):
    from examples.orfc_2446.dinov3.offline.package_release import (
        RELEASE_IDENTITY,
        RELEASE_NAMES,
        _validate_release_identity,
    )

    expected_names = list(RELEASE_NAMES.values())
    for plan_path in sorted((PLAN_ROOT / "dinov3").glob("*.yaml")):
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        actual_names = [
            Path(overrides["model"]["dino_codec"]["orfc_weights_path"]).name
            for overrides in plan["multi_run"].values()
        ]
        assert actual_names == expected_names

    valid = tmp_path / "valid.npz"
    np.savez(valid, **RELEASE_IDENTITY)
    _validate_release_identity(valid)

    invalid = tmp_path / "missing-provenance.npz"
    np.savez(invalid, norm_mode="split_reg_cls_patch", n_prefix=5)
    with pytest.raises(KeyError, match="identity metadata"):
        _validate_release_identity(invalid)


def test_published_dinov2_plans_cover_every_proposal_reference_point(tmp_path):
    entries = discover_plan_results(tmp_path)
    rows, counts = build_rows(tmp_path)

    assert len(entries) == 90
    assert len(rows) == 74
    assert counts["missing_plan"] == 0
    assert counts["missing_result"] == 74


def test_result_collector_reads_the_engine_output_contract(tmp_path):
    entry = discover_plan_results(tmp_path)[0]
    metrics = {"bpfp": 0.25}
    metrics["cls_top-1" if entry.task == "cls" else "semseg_mIoU"] = 90.0
    entry.result_path.parent.mkdir(parents=True)
    entry.result_path.write_text(json.dumps({"results": metrics}), encoding="utf-8")

    results = collect_all(tmp_path, require_all=False)

    assert len(results) == 1
    assert results[0].ckpt == entry.codec_path
    assert results[0].result_path == str(entry.result_path)

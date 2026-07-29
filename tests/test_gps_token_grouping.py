from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from cofai.backbone import GPSTransReIDBackbone
from cofai.heads import GPSTransReIDHead
from cofai.index_codecs import (
    AdaptiveBitmapIndexCodec,
    EncodedIndexSet,
)
from cofai.token_grouping import GPSTokenGrouper
from examples.gps.config import load_config
from examples.gps.reid.evaluator import _output_order
from examples.gps.run_eval_token_grouping import (
    benchmark_selection_map,
    summarize_run,
)
from examples.gps.token_grouping_eval.model_probe import ProbeRecord

SELECTION_MAP_CODEC = AdaptiveBitmapIndexCodec()


def test_selection_indices_round_trip_from_self_contained_stream():
    encoded = SELECTION_MAP_CODEC.encode([1, 4, 7], universe_size=9)
    stream = encoded.to_bytes()

    assert encoded.stream_bits == len(stream) * 8
    assert EncodedIndexSet.from_bytes(stream) == encoded
    assert SELECTION_MAP_CODEC.decode(stream) == [1, 4, 7]


@pytest.mark.parametrize(
    ("indices", "expected_coding"),
    [
        ([3], "index"),
        (list(range(12)), "bitmap"),
    ],
)
def test_selection_indices_choose_smaller_representation(indices, expected_coding):
    encoded = SELECTION_MAP_CODEC.encode(indices, universe_size=16)

    assert encoded.coding == expected_coding
    assert SELECTION_MAP_CODEC.decode(encoded) == indices


@pytest.mark.parametrize(
    "stream",
    [
        b"",
        b"BAD!" + b"\x00" * 9,
        SELECTION_MAP_CODEC.encode([1], universe_size=9).to_bytes() + b"\x00",
    ],
)
def test_selection_indices_reject_malformed_streams(stream):
    with pytest.raises(ValueError):
        SELECTION_MAP_CODEC.decode(stream)


def test_graph_grouping_across_operating_points():
    torch.manual_seed(10)
    tokens = torch.randn(2, 17, 8)
    attention = torch.softmax(torch.randn(2, 4, 17, 17), dim=-1)
    grouper = GPSTokenGrouper()

    for rho in (0.0, 0.5, 0.9):
        grouped = grouper(tokens, attention, keep_ratio=1.0 - rho)
        assert grouped.tokens.shape == (
            2,
            grouped.kept_indices.shape[1] + 1,
            8,
        )


def test_graph_grouping_rejects_invalid_structural_inputs():
    with pytest.raises(ValueError, match="at least one patch"):
        GPSTokenGrouper()(torch.randn(1, 1, 8), torch.randn(1, 1, 1, 1), 1.0)

    with pytest.raises(ValueError, match="duplicates"):
        GPSTokenGrouper()(
            torch.randn(1, 3, 8),
            torch.randn(1, 1, 3, 3),
            0.5,
            original_indices=torch.tensor([[0, 0]]),
        )


@pytest.mark.parametrize("feature_bit_depth", [8, 16])
def test_summary_counts_complete_map_stream(feature_bit_depth):
    record = ProbeRecord(
        n_tokens=16,
        kept_indices=list(range(0, 16, 2)),
        preprocess_ms=0.1,
    )
    grouping_config = SimpleNamespace(
        feature_dimension=8,
        timing_repeats=1,
        warmup=0,
    )
    encoded = SELECTION_MAP_CODEC.encode(record.kept_indices, record.n_tokens)
    feature_bits = (len(record.kept_indices) + 1) * 8 * feature_bit_depth

    summary = summarize_run(
        [record],
        [encoded.to_bytes()],
        {
            "mAP": 0.5,
            "rank1": 0.6,
            "rank5": 0.7,
            "rank10": 0.8,
            "bits": {
                "feature": float(feature_bits),
                "selection_map": float(encoded.stream_bits),
            },
            "coded_groups": 1,
        },
        grouping_config,
        rho=0.5,
    )

    assert summary["bpfp_map_mean"] == encoded.stream_bits / record.n_tokens
    assert summary["bpfp_feat_mean"] == feature_bits / record.n_tokens
    assert summary["feature_bit_depth"] == feature_bit_depth
    assert summary["map_recovery_accuracy"] == 1.0


def test_map_recovery_checks_the_transmitted_stream_contents():
    record = ProbeRecord(
        n_tokens=16,
        kept_indices=list(range(0, 16, 2)),
        preprocess_ms=0.1,
    )
    wrong_stream = SELECTION_MAP_CODEC.encode(
        list(range(1, 16, 2)),
        universe_size=16,
    ).to_bytes()

    with pytest.raises(RuntimeError, match="round trip failed"):
        benchmark_selection_map(
            record,
            wrong_stream,
            repeats=1,
            warmup=0,
        )


def test_gps_backbone_emits_raw_index_set_and_keeps_cls():
    class DummyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = SimpleNamespace(num_patches=4)
            self.token_grouper = object()

        def forward(self, x, **kwargs):
            return (
                torch.arange(48, dtype=torch.float32).reshape(2, 3, 8),
                torch.arange(2),
                0.0,
                {
                    "indices": torch.tensor([[0, 2], [1, 3]]),
                    "token_count": 4,
                },
            )

    backbone = GPSTransReIDBackbone(
        DummyBase(),
        torch.nn.Identity(),
        torch.nn.Identity(),
        cls_token_num=1,
        shuffle_groups=2,
        shift_num=1,
        divide_length=4,
        rearrange=False,
    )
    head = GPSTransReIDHead(
        torch.nn.Identity(),
        [torch.nn.Identity() for _ in range(4)],
        neck_feature="before",
        cls_token_num=1,
    ).eval()
    encoded = backbone.encode(
        torch.zeros(2, 3, 4, 4),
        label=torch.arange(2),
        multi_view=True,
    )

    assert encoded["h"].shape[1] == encoded["pstate"]["retained_patch_tokens"] + 1
    assert set(encoded["index_sets"]) == {"selection_map"}
    rows = SELECTION_MAP_CODEC.encode_batch(encoded["index_sets"]["selection_map"])
    assert all(SELECTION_MAP_CODEC.decode(row[0]) in ([0, 2], [1, 3]) for row in rows)
    decoded = backbone.decode(
        encoded["h"],
        pstate=encoded["pstate"],
        tasks=["reid"],
    )
    embedding = head(decoded["reid"])
    expected = torch.cat(
        [
            encoded["h"][:, 0],
            encoded["h"][:, 0] / 4,
            encoded["h"][:, 0] / 4,
            encoded["h"][:, 0] / 4,
            encoded["h"][:, 0] / 4,
        ],
        dim=1,
    )
    torch.testing.assert_close(embedding, expected)


def test_gps_config_uses_worktree_dotenv(monkeypatch, tmp_path):
    project_root = tmp_path / "shared-project"
    (tmp_path / ".env").write_text(f"PROJECT_ROOT={project_root}\n", encoding="utf-8")
    config_path = tmp_path / "gps.yml"
    config_path.write_text(
        "data_root: ${project_root:}/data\ncheckpoint: ${project_root:}/weights/model.pth\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)

    config = load_config(config_path)

    assert config.data_root == str(project_root / "data")
    assert config.checkpoint == str(project_root / "weights/model.pth")


def test_reid_output_order_is_derived_from_batch_metadata():
    query_meta = {"pid": (7, 7, 7, 9, 9, 9)}
    gallery_meta = {"pid": (7, 8, 9)}

    assert _output_order(query_meta, "query").tolist() == [0, 3]
    assert _output_order(gallery_meta, "gallery").tolist() == [0, 1, 2]

    with pytest.raises(ValueError, match="one vehicle identity"):
        _output_order({"pid": (7, 8, 7)}, "query")

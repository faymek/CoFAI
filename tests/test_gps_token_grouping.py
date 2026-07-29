from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from cofai.token_codecs import (
    EncodedSelectionMap,
    decode_selection_indices,
    encode_selection_indices,
)
from cofai.token_grouping import GraphTokenGrouper
from examples.gps.config import load_config
from examples.gps.reid.backbone import GPSReIDBackbone
from examples.gps.reid.evaluator import _output_order
from examples.gps.reid.head import GPSReIDHead
from examples.gps.run_eval_token_grouping import summarize_run
from examples.gps.token_grouping_eval.model_probe import ProbeRecord


def test_selection_indices_round_trip_from_self_contained_stream():
    encoded = encode_selection_indices([1, 4, 7], token_count=9)
    stream = encoded.to_bytes()

    assert encoded.stream_bits == len(stream) * 8
    assert EncodedSelectionMap.from_bytes(stream) == encoded
    assert decode_selection_indices(stream) == [1, 4, 7]


@pytest.mark.parametrize(
    "stream",
    [
        b"",
        b"BAD!" + b"\x00" * 9,
        encode_selection_indices([1], token_count=9).to_bytes() + b"\x00",
    ],
)
def test_selection_indices_reject_malformed_streams(stream):
    with pytest.raises(ValueError):
        decode_selection_indices(stream)


def test_graph_grouping_across_operating_points():
    torch.manual_seed(10)
    tokens = torch.randn(2, 17, 8)
    attention = torch.softmax(torch.randn(2, 4, 17, 17), dim=-1)
    grouper = GraphTokenGrouper()

    for rho in (0.0, 0.5, 0.9):
        grouped = grouper(tokens, attention, keep_ratio=1.0 - rho)
        assert grouped.tokens.shape == (
            2,
            grouped.kept_indices.shape[1] + 1,
            8,
        )


def test_graph_grouping_rejects_invalid_structural_inputs():
    with pytest.raises(ValueError, match="at least one patch"):
        GraphTokenGrouper()(torch.randn(1, 1, 8), torch.randn(1, 1, 1, 1), 1.0)

    with pytest.raises(ValueError, match="duplicates"):
        GraphTokenGrouper()(
            torch.randn(1, 3, 8),
            torch.randn(1, 1, 3, 3),
            0.5,
            original_indices=torch.tensor([[0, 0]]),
        )


def test_summary_counts_complete_map_stream():
    record = ProbeRecord(
        n_tokens=16,
        kept_indices=list(range(0, 16, 2)),
        preprocess_ms=0.1,
    )
    grouping_config = SimpleNamespace(
        feature_dimension=8,
        feature_bit_depth=16,
        timing_repeats=1,
        warmup=0,
    )
    encoded = encode_selection_indices(record.kept_indices, record.n_tokens)
    feature_bits = (len(record.kept_indices) + 1) * 8 * 16

    summary = summarize_run(
        [record],
        {
            "mAP": 0.5,
            "rank1": 0.6,
            "rank5": 0.7,
            "rank10": 0.8,
            "bits": {
                "fp16": float(feature_bits),
                "selection_map": float(encoded.stream_bits),
            },
            "coded_groups": 1,
        },
        grouping_config,
        rho=0.5,
    )

    assert summary["bpfp_map_mean"] == encoded.stream_bits / record.n_tokens
    assert summary["bpfp_feat_mean"] == feature_bits / record.n_tokens
    assert summary["map_recovery_accuracy"] == 1.0


def test_gps_backbone_emits_flat_map_side_stream_and_keeps_cls():
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

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = DummyBase()
            self.b1 = torch.nn.Identity()
            self.b2 = torch.nn.Identity()
            self.bottleneck = torch.nn.Identity()
            self.bottleneck_1 = torch.nn.Identity()
            self.bottleneck_2 = torch.nn.Identity()
            self.bottleneck_3 = torch.nn.Identity()
            self.bottleneck_4 = torch.nn.Identity()
            self.cls_token_num = 1
            self.shuffle_groups = 2
            self.shift_num = 1
            self.divide_length = 4
            self.rearrange = False
            self.neck_feat = "before"

    legacy_model = DummyModel()
    backbone = GPSReIDBackbone(legacy_model)
    head = GPSReIDHead(legacy_model).eval()
    encoded = backbone.encode(
        torch.zeros(2, 3, 4, 4),
        label=torch.arange(2),
        multi_view=True,
    )

    assert encoded["h"].shape[1] == encoded["pstate"]["retained_patch_tokens"] + 1
    assert set(encoded["strings"]) == {"selection_map"}
    assert all(
        decode_selection_indices(row[0]) in ([0, 2], [1, 3])
        for row in encoded["strings"]["selection_map"]
    )
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

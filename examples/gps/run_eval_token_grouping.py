"""Run the real GPS rate-performance and token-map recovery sweep."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from cofai.index_codecs import AdaptiveBitmapIndexCodec
from examples.gps.config import load_config
from examples.gps.model import build_codec_model
from examples.gps.reid.dataloader import make_dataloader
from examples.gps.reid.evaluator import evaluate_model
from examples.gps.reid.legacy_config import build_legacy_config
from examples.gps.token_grouping_eval.model_probe import (
    ProbeRecord,
    install_token_grouping_probe,
)
from examples.gps.token_grouping_eval.bitrate import bitrate_record
from examples.gps.token_grouping_eval.result_writer import write_csv, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION_MAP_CODEC = AdaptiveBitmapIndexCodec()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPS rate-performance evaluation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--token-grouping-config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rho", type=float, nargs="+")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def mean(values) -> float:
    values = list(values)
    return 0.0 if not values else float(sum(values) / len(values))


def benchmark_selection_map(
    record: ProbeRecord,
    encoded_stream: bytes | None,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    if len(record.kept_indices) == record.n_tokens:
        if encoded_stream is not None:
            raise RuntimeError("unpruned features must not transmit a selection map")
        decoded = record.kept_indices
    else:
        if encoded_stream is None:
            raise RuntimeError("pruned features are missing a selection-map stream")
        decoded = SELECTION_MAP_CODEC.decode(encoded_stream)
    if decoded != record.kept_indices:
        raise RuntimeError("selection-map round trip failed")

    parse_times = []
    if encoded_stream is not None:
        for _ in range(warmup):
            SELECTION_MAP_CODEC.decode(encoded_stream)
        for _ in range(repeats):
            start = time.perf_counter()
            SELECTION_MAP_CODEC.decode(encoded_stream)
            parse_times.append((time.perf_counter() - start) * 1000.0)

    return {
        "map_recovery": 1.0,
        "map_decode_ms": mean(parse_times),
    }


def summarize_run(
    records: list[ProbeRecord],
    selection_streams: list[bytes | None],
    task: dict,
    tg,
    rho: float,
) -> dict:
    if not records:
        raise RuntimeError("the model probe did not collect any query token maps")
    if len(selection_streams) != len(records):
        raise RuntimeError(
            "transmitted selection maps and probe records have different lengths"
        )

    feature_dim = int(tg.feature_dimension)
    actual_bits = dict(task.get("bits") or {})
    actual_feature_bits = actual_bits.get("feature")
    if actual_feature_bits is None:
        raise RuntimeError("real codec result is missing the `feature` stream")
    feature_values = sum(
        (len(record.kept_indices) + 1) * feature_dim for record in records
    )
    if feature_values <= 0 or not float(actual_feature_bits).is_integer():
        raise RuntimeError("raw feature stream must contain an integer bit count")
    feature_bits_total = int(actual_feature_bits)
    bit_depth, remainder = divmod(feature_bits_total, feature_values)
    if remainder:
        raise RuntimeError(
            "raw feature stream does not have a constant integer bit depth"
        )

    bitrate_rows = []
    for record, selection_stream in zip(records, selection_streams, strict=True):
        kept_count = len(record.kept_indices)
        map_bits = 0 if selection_stream is None else len(selection_stream) * 8
        # The transmitted compact tensor contains one CLS token in addition to
        # every retained patch token.
        feature_bits = (kept_count + 1) * feature_dim * bit_depth
        bitrate_rows.append(
            bitrate_record(
                token_count=record.n_tokens,
                kept_tokens=kept_count,
                feature_bits=feature_bits,
                map_bits=map_bits,
                dense_feature_bits=(record.n_tokens + 1) * feature_dim * bit_depth,
            )
        )

    coded_groups = int(task.get("coded_groups", 0))
    if coded_groups != len(records):
        raise RuntimeError(
            "real codec/query grouping mismatch: "
            f"{coded_groups} coded groups versus {len(records)} selection records"
        )
    expected_feature = sum(int(row["feature_bits"]) for row in bitrate_rows)
    expected_map = sum(int(row["map_bits"]) for row in bitrate_rows)
    if actual_feature_bits != float(expected_feature):
        raise RuntimeError(
            f"raw feature stream mismatch: expected {expected_feature}, got "
            f"{actual_feature_bits}"
        )
    if actual_bits.get("selection_map", 0.0) != float(expected_map):
        raise RuntimeError(
            f"selection-map stream mismatch: expected {expected_map}, got "
            f"{actual_bits.get('selection_map', 0.0)}"
        )

    map_rows = [
        benchmark_selection_map(
            record,
            selection_stream,
            repeats=int(tg.timing_repeats),
            warmup=int(tg.warmup),
        )
        for record, selection_stream in zip(
            records[:256],
            selection_streams[:256],
            strict=True,
        )
    ]
    return {
        "rho": float(rho),
        "mAP": 100.0 * float(task["mAP"]),
        "R1": 100.0 * float(task["rank1"]),
        "R5": 100.0 * float(task["rank5"]),
        "R10": 100.0 * float(task["rank10"]),
        "num_groups": len(records),
        "n_tokens": records[0].n_tokens,
        "feature_bit_depth": bit_depth,
        "kept_count_mean": mean(row["kept_tokens"] for row in bitrate_rows),
        "keep_ratio_mean": mean(row["keep_ratio"] for row in bitrate_rows),
        "bpfp_feat_mean": mean(row["feature_bpfp"] for row in bitrate_rows),
        "bpfp_map_mean": mean(row["map_bpfp"] for row in bitrate_rows),
        "bpfp_total_mean": mean(row["total_bpfp"] for row in bitrate_rows),
        "eta_map_mean": mean(row["side_information_fraction"] for row in bitrate_rows),
        "rate_saving_mean": mean(row["rate_saving"] for row in bitrate_rows),
        "map_recovery_accuracy": mean(row["map_recovery"] for row in map_rows),
        "preprocess_ms_mean": mean(record.preprocess_ms for record in records),
        "map_decode_ms_mean": mean(row["map_decode_ms"] for row in map_rows),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.token_grouping_config)
    if args.data_root is not None:
        cfg.data_root = str(args.data_root.resolve())
    if args.checkpoint is not None:
        cfg.checkpoint = str(args.checkpoint.resolve())
    if args.batch_size is not None:
        if args.batch_size <= 0 or args.batch_size % 3:
            raise ValueError("GPS eval batch size must be a positive multiple of 3")
        cfg.evaluation.batch_size = args.batch_size

    base_legacy_cfg = build_legacy_config(cfg)
    loaders = make_dataloader(base_legacy_cfg)
    rhos = (
        args.rho
        if args.rho is not None
        else [float(value) for value in cfg.token_grouping.rho]
    )
    rows = []
    output = args.output or (
        PROJECT_ROOT / "logs" / "gps" / str(cfg.name) / "token_grouping.csv"
    )

    for rho in rhos:
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        cfg.model.pruning_ratios = [float(rho)] * len(cfg.model.pruning_layers)
        legacy_cfg = build_legacy_config(cfg)
        model = build_codec_model(
            cfg,
            legacy_cfg,
            camera_num=loaders.camera_num,
            view_num=loaders.view_num,
        )
        probe = install_token_grouping_probe(model)
        selection_streams: list[bytes | None] = []

        def collect_selection_streams(coded_unit, group_count: int) -> None:
            rows = coded_unit["strings"].get("selection_map")
            if rows is None:
                selection_streams.extend([None] * group_count)
                return
            if len(rows) != group_count:
                raise RuntimeError(
                    "selection-map stream batch does not match grouped features"
                )
            selection_streams.extend(row[0] for row in rows)

        print(f"\nEvaluating {cfg.name} at rho={rho:.1f}")
        task = evaluate_model(
            legacy_cfg,
            model,
            loaders.query,
            loaders.gallery,
            loaders.num_query,
            real_codec=True,
            on_query_coded_unit=collect_selection_streams,
        )
        row = summarize_run(
            probe.records,
            selection_streams,
            task,
            cfg.token_grouping,
            rho,
        )
        rows.append(row)
        write_csv(output, rows)
        write_json(output.with_suffix(".json"), rows)
        print(
            f"rho={rho:.1f} K={row['kept_count_mean']:.1f} "
            f"BPFP={row['bpfp_total_mean']:.3f} mAP={row['mAP']:.3f} "
            f"R1={row['R1']:.3f} map={row['map_recovery_accuracy']:.1%}"
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

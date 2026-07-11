"""Run the real GPS rate-performance and token-map recovery sweep."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cofai.token_codecs.token_selection_map import (
    decode_selection_indices,
    encode_selection_indices,
)
from examples.gps.config import load_config
from examples.gps.reid.dataloader import make_dataloader
from examples.gps.reid.evaluator import evaluate_model
from examples.gps.reid.legacy_config import build_legacy_config
from examples.gps.reid.model import make_model
from examples.gps.token_grouping_eval.model_probe import (
    ProbeRecord,
    install_token_grouping_probe,
)
from examples.gps.token_grouping_eval.result_writer import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPS rate-performance evaluation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--token-grouping-config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rho", type=float, nargs="+")
    return parser.parse_args()


def mean(values) -> float:
    values = list(values)
    return 0.0 if not values else float(sum(values) / len(values))


def benchmark_restore(
    record: ProbeRecord,
    feature_dim: int,
    device: torch.device,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    if len(record.kept_indices) == record.n_tokens:
        return {
            "map_recovery": 1.0,
            "sparse_decode_restore_ms": 0.0,
            "fixed_decode_restore_ms": 0.0,
        }

    encoded = encode_selection_indices(record.kept_indices, record.n_tokens)
    decoded = decode_selection_indices(encoded)
    if decoded != record.kept_indices:
        raise RuntimeError("selection-map round trip failed")

    indices = torch.tensor(decoded, dtype=torch.long, device=device)
    tokens = torch.randn(
        len(decoded),
        feature_dim,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device=device,
    )

    def restore_once() -> torch.Tensor:
        restored = torch.zeros(record.n_tokens, feature_dim, dtype=tokens.dtype, device=device)
        restored.index_copy_(0, indices, tokens)
        return restored

    for _ in range(warmup):
        decode_selection_indices(encoded)
        restore_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    parse_times = []
    restore_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        decode_selection_indices(encoded)
        parse_times.append((time.perf_counter() - start) * 1000.0)

        if device.type == "cuda":
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            restore_once()
            event_end.record()
            torch.cuda.synchronize(device)
            restore_times.append(float(event_start.elapsed_time(event_end)))
        else:
            start = time.perf_counter()
            restore_once()
            restore_times.append((time.perf_counter() - start) * 1000.0)

    parse_ms = mean(parse_times)
    return {
        "map_recovery": 1.0,
        "sparse_decode_restore_ms": parse_ms,
        "fixed_decode_restore_ms": parse_ms + mean(restore_times),
    }


def summarize_run(records: list[ProbeRecord], task: dict, tg, rho: float) -> dict:
    if not records:
        raise RuntimeError("the model probe did not collect any query token maps")

    feature_dim = int(tg.feature_dimension)
    bit_depth = int(tg.feature_bit_depth)
    metadata_bits = int(tg.metadata_bits)
    bitrate_rows = []
    for record in records:
        kept_count = len(record.kept_indices)
        dense_bits = record.n_tokens * feature_dim * bit_depth + metadata_bits
        if kept_count == record.n_tokens:
            map_bits = 0
        else:
            map_bits = encode_selection_indices(
                record.kept_indices, record.n_tokens
            ).payload_bits
        feature_bits = kept_count * feature_dim * bit_depth
        total_bits = feature_bits + map_bits + metadata_bits
        bitrate_rows.append(
            {
                "kept_count": kept_count,
                "keep_ratio": kept_count / record.n_tokens,
                "bpfp_feat": feature_bits / record.n_tokens,
                "bpfp_map": (map_bits + metadata_bits) / record.n_tokens,
                "bpfp_total": total_bits / record.n_tokens,
                "eta_map": (map_bits + metadata_bits) / total_bits,
                "rate_saving": 1.0 - total_bits / dense_bits,
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restore_rows = [
        benchmark_restore(
            record,
            feature_dim,
            device,
            repeats=int(tg.timing_repeats),
            warmup=int(tg.warmup),
        )
        for record in records[:256]
    ]
    return {
        "rho": float(rho),
        "mAP": 100.0 * float(task["mAP"]),
        "R1": 100.0 * float(task["rank1"]),
        "R5": 100.0 * float(task["rank5"]),
        "R10": 100.0 * float(task["rank10"]),
        "num_groups": len(records),
        "n_tokens": records[0].n_tokens,
        "kept_count_mean": mean(row["kept_count"] for row in bitrate_rows),
        "keep_ratio_mean": mean(row["keep_ratio"] for row in bitrate_rows),
        "bpfp_feat_mean": mean(row["bpfp_feat"] for row in bitrate_rows),
        "bpfp_map_mean": mean(row["bpfp_map"] for row in bitrate_rows),
        "bpfp_total_mean": mean(row["bpfp_total"] for row in bitrate_rows),
        "eta_map_mean": mean(row["eta_map"] for row in bitrate_rows),
        "rate_saving_mean": mean(row["rate_saving"] for row in bitrate_rows),
        "map_recovery_accuracy": mean(row["map_recovery"] for row in restore_rows),
        "preprocess_ms_mean": mean(record.preprocess_ms for record in records),
        "sparse_decode_restore_ms_mean": mean(
            row["sparse_decode_restore_ms"] for row in restore_rows
        ),
        "fixed_decode_restore_ms_mean": mean(
            row["fixed_decode_restore_ms"] for row in restore_rows
        ),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.token_grouping_config)
    if args.data_root is not None:
        cfg.data_root = str(args.data_root.resolve())
    if args.checkpoint is not None:
        cfg.checkpoint = str(args.checkpoint.resolve())

    base_legacy_cfg = build_legacy_config(cfg)
    loaders = make_dataloader(base_legacy_cfg)
    _, _, _, query_loader, gallery_loader, _, num_query, num_classes, camera_num, view_num = loaders
    rhos = args.rho if args.rho is not None else [float(value) for value in cfg.token_grouping.rho]
    rows = []
    output = args.output or (
        PROJECT_ROOT / "logs" / "gps" / str(cfg.dataset.name) / "token_grouping.csv"
    )

    for rho in rhos:
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        cfg.model.pruning_ratios = [float(rho)] * len(cfg.model.pruning_layers)
        legacy_cfg = build_legacy_config(cfg)
        model = make_model(
            legacy_cfg,
            num_class=num_classes,
            camera_num=camera_num,
            view_num=view_num,
        )
        model.load_param(str(cfg.checkpoint))
        probe = install_token_grouping_probe(model)
        print(f"\nEvaluating {cfg.dataset.name} at rho={rho:.1f}")
        task = evaluate_model(legacy_cfg, model, query_loader, gallery_loader, num_query)
        row = summarize_run(probe.records, task, cfg.token_grouping, rho)
        rows.append(row)
        write_csv(output, rows)
        write_json(output.with_suffix(".json"), rows)
        print(
            f"rho={rho:.1f} K={row['kept_count_mean']:.1f} "
            f"BPFP={row['bpfp_total_mean']:.3f} mAP={row['mAP']:.3f} "
            f"R1={row['R1']:.3f} recovery={row['map_recovery_accuracy']:.1%}"
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

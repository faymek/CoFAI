#!/usr/bin/env python3
"""Compare published plan results with the M2446 proposal reference points."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from plan_results import (
    PlanResult,
    discover_plan_results,
    get_project_root,
    load_metrics,
    parse_codec_params,
)


# Reference points transcribed from plot_rate_task.py lines 520-711.
# Each entry is (backbone, task, block, layer, config, BPFP, task metric).
PLOT_OURS: list[tuple[str, str, int, str, str, float, float]] = [
    ("vitl14", "cls", 5, "blk05", "K=4, e32, lambda=0.0, lr=5e-4, ep=300", 0.0569, 93.80),
    ("vitl14", "cls", 5, "blk05", "K=8, e32", 0.087, 95.00),
    ("vitl14", "cls", 5, "blk05", "K=16, e32", 0.1179, 96.00),
    ("vitl14", "cls", 5, "blk05", "K=64, e32, lr=5e-4", 0.1796, 96.80),
    ("vitl14", "cls", 5, "blk05", "K=256, e32", 0.2361, 97.00),
    ("vitl14", "cls", 5, "blk05", "K=64, e16", 0.3534, 97.40),
    ("vitl14", "cls", 5, "blk05", "K=256, e16", 0.4607, 97.80),
    ("vitl14", "cls", 10, "blk10", "K=4, e32", 0.0602, 96.00),
    ("vitl14", "cls", 10, "blk10", "K=8, e32", 0.0913, 97.00),
    ("vitl14", "cls", 10, "blk10", "K=16, e32", 0.1228, 97.80),
    ("vitl14", "cls", 10, "blk10", "K=64, e32, lambda=0.0", 0.1849, 98.00),
    ("vitl14", "cls", 10, "blk10", "K=256, e32, lambda=0.0, lr=5e-4", 0.246, 98.00),
    ("vitl14", "cls", 15, "blk15", "K=4, e32", 0.061, 93.80),
    ("vitl14", "cls", 15, "blk15", "K=8, e32", 0.0922, 95.00),
    ("vitl14", "cls", 15, "blk15", "K=16, e32", 0.123, 96.00),
    ("vitl14", "cls", 15, "blk15", "K=64, e32, lr=5e-4", 0.1849, 96.80),
    ("vitl14", "cls", 15, "blk15", "K=256, e32", 0.2471, 97.00),
    ("vitl14", "cls", 15, "blk15", "K=64, e16", 0.3707, 97.40),
    ("vitl14", "cls", 15, "blk15", "K=256, e16, lambda=0.2", 0.4476, 97.60),
    ("vitl14", "cls", 15, "blk15", "K=256, e16", 0.488, 97.80),
    ("vitl14", "cls", 20, "blk20", "K=8, e32", 0.0928, 87.00),
    ("vitl14", "cls", 20, "blk20", "K=16, e32, lr=5e-4", 0.1239, 92.40),
    ("vitl14", "cls", 20, "blk20", "K=32, e32", 0.1538, 93.00),
    ("vitl14", "cls", 20, "blk20", "K=64, e32, lambda=0.0", 0.1842, 94.00),
    ("vitl14", "cls", 20, "blk20", "K=256, e32, lambda=0.0, lr=5e-4", 0.2469, 95.60),
    ("vitl14", "cls", 20, "blk20", "K=256, e16", 0.4883, 97.40),
    ("vitl14", "seg", 5, "blk05", "K=4, e32, lambda=0.0, lr=5e-4, ep=300", 0.0565, 78.30),
    ("vitl14", "seg", 5, "blk05", "K=8, e32", 0.0864, 78.90),
    ("vitl14", "seg", 5, "blk05", "K=16, e32", 0.1172, 79.40),
    ("vitl14", "seg", 5, "blk05", "K=64, e32, lr=5e-4", 0.1788, 79.70),
    ("vitl14", "seg", 5, "blk05", "K=256, e32", 0.2339, 80.00),
    ("vitl14", "seg", 5, "blk05", "K=64, e16", 0.3497, 80.40),
    ("vitl14", "seg", 10, "blk10", "K=4, e32", 0.0597, 78.40),
    ("vitl14", "seg", 10, "blk10", "K=8, e32", 0.091, 80.10),
    ("vitl14", "seg", 10, "blk10", "K=64, e32, lambda=0.0", 0.1803, 80.40),
    ("vitl14", "seg", 10, "blk10", "K=256, e32, lambda=0.0, lr=5e-4", 0.2389, 80.90),
    ("vitl14", "seg", 10, "blk10", "K=64, e16", 0.3678, 81.20),
    ("vitl14", "seg", 15, "blk15", "K=4, e32", 0.0607, 78.30),
    ("vitl14", "seg", 15, "blk15", "K=8, e32", 0.092, 78.90),
    ("vitl14", "seg", 15, "blk15", "K=16, e32", 0.1227, 79.40),
    ("vitl14", "seg", 15, "blk15", "K=64, e32, lr=5e-4", 0.1849, 79.70),
    ("vitl14", "seg", 15, "blk15", "K=256, e32", 0.2472, 80.00),
    ("vitl14", "seg", 15, "blk15", "K=64, e16", 0.3708, 80.40),
    ("vitl14", "seg", 20, "blk20", "K=8, e32", 0.0942, 68.60),
    ("vitl14", "seg", 20, "blk20", "K=16, e32, lr=5e-4", 0.1244, 71.00),
    ("vitl14", "seg", 20, "blk20", "K=64, e32, lambda=0.0", 0.177, 72.80),
    ("vitl14", "seg", 20, "blk20", "K=256, e16", 0.485, 75.50),
    ("vitg14", "cls", 9, "blk09", "K=4, e32, lambda=0.5", 0.0603, 97.40),
    ("vitg14", "cls", 9, "blk09", "K=8, e32, lambda=0.5", 0.0902, 97.80),
    ("vitg14", "cls", 9, "blk09", "K=16, e32, lambda=0.5", 0.1207, 99.00),
    ("vitg14", "cls", 9, "blk09", "K=64, e32, lambda=0.5", 0.1808, 99.20),
    ("vitg14", "cls", 9, "blk09", "K=256, e32, lambda=0.5", 0.239, 99.20),
    ("vitg14", "cls", 19, "blk19", "K=4, e32, lambda=0.5", 0.0605, 97.60),
    ("vitg14", "cls", 19, "blk19", "K=8, e32, lambda=0.5", 0.092, 98.80),
    ("vitg14", "cls", 19, "blk19", "K=16, e32, lambda=0.5", 0.1231, 99.40),
    ("vitg14", "cls", 19, "blk19", "K=64, e32, lambda=0.5", 0.1849, 99.40),
    ("vitg14", "cls", 19, "blk19", "K=256, e32, lambda=0.5", 0.2442, 99.40),
    ("vitg14", "cls", 29, "blk29", "K=4, e32, lambda=0.5", 0.0616, 95.60),
    ("vitg14", "cls", 29, "blk29", "K=8, e32, lambda=0.5", 0.0929, 96.20),
    ("vitg14", "cls", 29, "blk29", "K=16, e32, lambda=0.5", 0.1241, 97.80),
    ("vitg14", "cls", 29, "blk29", "K=64, e32, lambda=0.0", 0.1853, 98.00),
    ("vitg14", "cls", 29, "blk29", "K=256, e32, lambda=0.0, lr=5e-4", 0.2462, 98.80),
    ("vitg14", "seg", 9, "blk09", "K=4, e32, lambda=0.5", 0.0599, 82.22),
    ("vitg14", "seg", 9, "blk09", "K=8, e32, lambda=0.5", 0.09, 82.92),
    ("vitg14", "seg", 9, "blk09", "K=16, e32, lambda=0.5", 0.1206, 83.06),
    ("vitg14", "seg", 9, "blk09", "K=64, e32, lambda=0.0", 0.1798, 83.25),
    ("vitg14", "seg", 19, "blk19", "K=4, e32, lambda=0.5", 0.0602, 83.13),
    ("vitg14", "seg", 19, "blk19", "K=8, e32, lambda=0.5", 0.0918, 83.43),
    ("vitg14", "seg", 19, "blk19", "K=256, e32, lambda=0.0, lr=5e-4", 0.2448, 83.48),
    ("vitg14", "seg", 29, "blk29", "K=4, e32, lambda=0.5", 0.0616, 79.75),
    ("vitg14", "seg", 29, "blk29", "K=8, e32, lambda=0.5", 0.0928, 80.80),
    ("vitg14", "seg", 29, "blk29", "K=16, e32, lambda=0.5", 0.1242, 81.38),
    ("vitg14", "seg", 29, "blk29", "K=64, e32, lambda=0.0", 0.184, 81.93),
    ("vitg14", "seg", 29, "blk29", "K=256, e32, lambda=0.0, lr=5e-4", 0.2442, 82.50),
]


def parse_reference_config(config: str) -> dict[str, int | float]:
    """Parse the fields that identify a proposal point."""
    k_match = re.search(r"\bK=(\d+)", config)
    emb_match = re.search(r"\be(\d+)", config)
    if not k_match or not emb_match:
        raise ValueError(f"Unsupported reference config: {config}")

    params: dict[str, int | float] = {
        "K": int(k_match.group(1)),
        "emb": int(emb_match.group(1)),
    }
    optional_patterns = {
        "lambda": r"lambda=([\d.]+)",
        "lr": r"lr=([\deE+.-]+)",
        "epochs": r"ep=(\d+)",
    }
    for name, pattern in optional_patterns.items():
        match = re.search(pattern, config)
        if match:
            params[name] = int(match.group(1)) if name == "epochs" else float(match.group(1))
    return params


def find_plan_result(
    entries: list[PlanResult],
    backbone: str,
    task: str,
    layer: str,
    config: str,
) -> PlanResult | None:
    """Find the unique published plan entry matching a proposal point."""
    expected = parse_reference_config(config)
    candidates = [
        entry for entry in entries if entry.backbone == backbone and entry.task == task and entry.layer == layer
    ]
    for name, value in expected.items():
        candidates = [entry for entry in candidates if parse_codec_params(entry.codec_path)[name] == value]
    if len(candidates) > 1 and "lambda" not in expected:
        candidates = [entry for entry in candidates if parse_codec_params(entry.codec_path)["lambda"] == 0.5]
    return candidates[0] if len(candidates) == 1 else None


def build_rows(results_dir: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Join proposal reference points with plan metadata and completed results."""
    entries = discover_plan_results(results_dir)
    rows: list[dict[str, object]] = []
    counts = {"match": 0, "mismatch": 0, "missing_plan": 0, "missing_result": 0}

    for backbone, task, block, layer, config, reference_bpfp, reference_metric in PLOT_OURS:
        entry = find_plan_result(entries, backbone, task, layer, config)
        params = parse_reference_config(config)
        status = "missing_plan"
        measured_bpfp = measured_metric = delta_bpfp = delta_metric = None

        if entry is None:
            counts["missing_plan"] += 1
        elif not entry.result_path.is_file():
            status = "missing_result"
            counts["missing_result"] += 1
            params = parse_codec_params(entry.codec_path)
        else:
            params = parse_codec_params(entry.codec_path)
            measured_bpfp, measured_metric = load_metrics(entry)
            delta_bpfp = measured_bpfp - reference_bpfp
            delta_metric = measured_metric - reference_metric
            status = "match" if abs(delta_bpfp) <= 0.01 and abs(delta_metric) <= 0.2 else "mismatch"
            counts[status] += 1

        rows.append(
            {
                "backbone": backbone,
                "task": task,
                "block": block,
                "layer": layer,
                "config": config,
                "K": params["K"],
                "emb": params["emb"],
                "lambda": params.get("lambda", ""),
                "lr": params.get("lr", ""),
                "epochs": params.get("epochs", ""),
                "checkpoint": Path(entry.codec_path).name if entry else "",
                "result_path": str(entry.result_path) if entry else "",
                "reference_bpfp": f"{reference_bpfp:.4f}",
                "reference_metric": f"{reference_metric:.2f}",
                "measured_bpfp": f"{measured_bpfp:.4f}" if measured_bpfp is not None else "",
                "measured_metric": f"{measured_metric:.2f}" if measured_metric is not None else "",
                "delta_bpfp": f"{delta_bpfp:.4f}" if delta_bpfp is not None else "",
                "delta_metric": f"{delta_metric:.2f}" if delta_metric is not None else "",
                "status": status,
            }
        )
    return rows, counts


def write_report(rows: list[dict[str, object]], counts: dict[str, int], csv_path: Path, text_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"M2446 proposal vs published DINOv2 plan results — {len(rows)} reference points",
        f"match: {counts['match']}",
        f"mismatch: {counts['mismatch']}",
        f"missing_plan: {counts['missing_plan']}",
        f"missing_result: {counts['missing_result']}",
        "",
        "Largest metric gaps:",
    ]
    mismatches = [row for row in rows if row["status"] == "mismatch"]
    mismatches.sort(key=lambda row: -abs(float(row["delta_metric"] or 0)))
    for row in mismatches[:10]:
        lines.append(
            f"  {row['backbone']} {row['layer']} {row['task']} {row['config']}: "
            f"delta_metric={row['delta_metric']} delta_bpfp={row['delta_bpfp']}"
        )

    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {text_path}")
    print("\n".join(lines[:5]))


def main() -> None:
    project_root = get_project_root()
    default_results = project_root / "logs" / "orfc_2446" / "dinov2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=default_results / "ours_plot_vs_npz_comparison.csv",
    )
    parser.add_argument(
        "--out-text",
        type=Path,
        default=default_results / "ours_plot_vs_npz_summary.txt",
    )
    args = parser.parse_args()

    rows, counts = build_rows(args.results_dir.resolve())
    write_report(rows, counts, args.out_csv, args.out_text)


if __name__ == "__main__":
    main()

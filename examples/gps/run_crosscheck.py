"""Print GPS cross-check results in compact tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print current GPS results")
    parser.add_argument("--root", type=Path, default=Path("logs/gps"))
    return parser.parse_args()


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def fmt_ms(seconds: float) -> str:
    return f"{1000.0 * seconds:.3f}"


DISPLAY_NAMES = {"VeRi": "VeRi-776", "MuRI": "MuRI"}
DISPLAY_ORDER = ("VeRi", "MuRI")
DISPLAY_RHOS = (0.1, 0.3, 0.5, 0.7, 0.9)


def load_task_curve(dataset_root: Path) -> dict[float, dict[str, float]]:
    """Load optional per-rho task metrics without requiring them for proxy runs."""
    candidates = (
        dataset_root / "task_rate_curve.csv",
        dataset_root / "task_results.csv",
        dataset_root / "rate_curve.csv",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {}
    curve = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rho = float(row["rho"])
            curve[rho] = {
                key: float(row[key])
                for key in ("mAP", "rank1")
                if row.get(key, "") != ""
            }
    return curve


def available_datasets(root: Path) -> list[str]:
    found = {path.parent.name for path in root.glob("*/result.json")}
    return [dataset for dataset in DISPLAY_ORDER if dataset in found]


def print_m2405_results(root: Path, datasets: list[str]) -> None:
    print("M2405 cross-check")
    print("数据集          mAP    Rank-1    Rank-5   Rank-10    Time(s)")
    print("------------------------------------------------------------")
    for dataset in datasets:
        result = load_json(root / dataset / "result.json")
        print(
            f"{DISPLAY_NAMES[dataset]:<12} {fmt_percent(result['mAP']):>9} "
            f"{fmt_percent(result['rank1']):>9} {fmt_percent(result['rank5']):>9} "
            f"{fmt_percent(result['rank10']):>9} {float(result['elapsed_seconds']):>10.2f}"
        )


def print_rate_table(root: Path, dataset: str) -> None:
    dataset_root = root / dataset
    result = load_json(dataset_root / "result.json")
    task_curve = load_task_curve(dataset_root)
    rows = load_json(dataset_root / "token_grouping.json")
    by_rho = {round(float(row["rho"]), 1): row for row in rows}

    print(
        f"\n表 3-{1 if dataset == 'VeRi' else 2}  {DISPLAY_NAMES[dataset]}  代表性工作点的率-性能结果"
    )
    print("rho       K   R_keep   BPFP_total   码率节省       mAP    Rank-1")
    print("---------------------------------------------------------------------")
    for rho in DISPLAY_RHOS:
        row = by_rho.get(round(rho, 1))
        if row is None:
            continue
        task = task_curve.get(round(rho, 1), {})
        map_value = (
            f"{float(row['mAP']):.3f}"
            if "mAP" in row
            else fmt_percent(task["mAP"])
            if "mAP" in task
            else "--"
        )
        rank1_value = (
            f"{float(row['R1']):.3f}"
            if "R1" in row
            else fmt_percent(task["rank1"])
            if "rank1" in task
            else "--"
        )
        kept_count = float(row.get("kept_count_mean", row.get("kept_tokens")))
        keep_ratio = float(row.get("keep_ratio_mean", row.get("keep_ratio")))
        total_bpfp = float(row.get("bpfp_total_mean", row.get("total_bpfp")))
        saving = float(row.get("rate_saving_mean", row.get("rate_saving")))
        print(
            f"{rho:>3.1f} {kept_count:>7.0f} {keep_ratio:>8.3f} "
            f"{total_bpfp:>12.3f} {fmt_percent(saving):>10} "
            f"{map_value:>9} {rank1_value:>9}"
        )
    print(
        f"dense checkpoint: mAP {fmt_percent(result['mAP'])}, Rank-1 {fmt_percent(result['rank1'])}"
    )


def print_complexity_table(root: Path, datasets: list[str]) -> None:
    print("\n表 3-3  Selection map 一致性与复杂度统计")
    print("数据集       map round-trip   预处理(ms/group)   map 解码(ms/group)")
    print("-------------------------------------------------------------------")
    for dataset in datasets:
        rows = load_json(root / dataset / "token_grouping.json")
        rows = [row for row in rows if float(row["rho"]) > 0.0]
        if not rows:
            print(f"{DISPLAY_NAMES[dataset]:<12} {'--':>14} {'--':>17} {'--':>20}")
            continue
        recovery = min(float(row["map_recovery_accuracy"]) for row in rows)
        preprocess = [float(row["preprocess_ms_mean"]) for row in rows]
        decode = [float(row["map_decode_ms_mean"]) for row in rows]
        preprocess_range = f"{min(preprocess):.3f} 至 {max(preprocess):.3f}"
        decode_range = f"{min(decode):.3f} 至 {max(decode):.3f}"
        print(
            f"{DISPLAY_NAMES[dataset]:<12} {fmt_percent(recovery):>14} "
            f"{preprocess_range:>17} {decode_range:>20}"
        )


def main() -> None:
    args = parse_args()
    datasets = available_datasets(args.root)
    if not datasets:
        raise FileNotFoundError(f"No result files found under {args.root}")
    print_m2405_results(args.root, datasets)
    print("\nM2460 cross-check")
    for dataset in datasets:
        print_rate_table(args.root, dataset)
    print_complexity_table(args.root, datasets)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect Soft-PQ online eval results and compare with ORFC CSV benchmarks."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from plan_results import (
    discover_plan_results,
    format_codec_config,
    get_project_root,
    load_metrics,
)


@dataclass
class JobResult:
    bb: str
    layer: str
    task: str
    ckpt: str
    config: str
    bpfp: float
    metric: float
    result_path: str


def collect_all(results_dir: Path, require_all: bool = True) -> list[JobResult]:
    """Collect result files using the published plans as the only job list."""
    out: list[JobResult] = []
    missing: list[Path] = []
    for entry in discover_plan_results(results_dir):
        if not entry.result_path.is_file():
            missing.append(entry.result_path)
            continue
        bpfp, metric = load_metrics(entry)
        out.append(
            JobResult(
                bb=entry.backbone,
                layer=entry.layer,
                task=entry.task,
                ckpt=entry.codec_path,
                config=format_codec_config(entry.codec_path),
                bpfp=bpfp,
                metric=metric,
                result_path=str(entry.result_path),
            )
        )
    if missing and require_all:
        paths = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(f"{len(missing)} published plan results are missing:\n{paths}")
    return out


def parse_orfc_ours(csv_path: Path, task: str) -> dict[tuple[str, str], dict]:
    """Parse ORFC CSV Ours section -> {(layer, config_norm): {bpfp, metric}}."""
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ours_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Ours"))
    block_nums = [b.strip() for b in lines[ours_idx + 1].split(",") if b.strip().isdigit()]
    blocks = [f"blk{int(b):02d}" for b in block_nums]

    results: dict[tuple[str, str], dict] = {}
    metric_key = "acc" if task == "cls" else "miou"

    for ln in lines[ours_idx + 3 :]:
        if not ln.strip() or ln.startswith("Ours"):
            break
        parts = next(csv.reader([ln]))
        for bi, layer in enumerate(blocks):
            base = bi * 3
            if base + 2 >= len(parts):
                continue
            config = parts[base].strip()
            if not config or config == "config":
                continue
            try:
                bpfp = float(parts[base + 1])
                metric = float(parts[base + 2].replace("%", ""))
            except ValueError:
                continue
            results[(layer, normalize_config(config))] = {
                "bpfp": bpfp,
                metric_key: metric,
                "config": config,
            }
    return results


def normalize_config(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"lambda\s*=\s*[\d.]+", "lambda", s)
    s = re.sub(r"[^\w=,>+\-./]", "", s)  # drop garbled lambda chars
    s = re.sub(r"lambda", "lambda", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("lambda0.0", "lambda0").replace("lambda0", "lambda")
    return s


def find_orfc_match(layer: str, config: str, orfc: dict) -> dict | None:
    nc = normalize_config(config)
    key = (layer, nc)
    if key in orfc:
        return orfc[key]
    for (ly, cfg), val in orfc.items():
        if ly == layer and cfg == nc:
            return val
    # prefix match on K/e tags
    k = re.search(r"k=(\d+)", nc)
    e = re.search(r"e(\d+)", nc)
    if not k or not e:
        return None
    candidates = [
        (cfg, val)
        for (ly, cfg), val in orfc.items()
        if ly == layer and f"k={k.group(1)}" in cfg and f"e{e.group(1)}" in cfg
    ]
    if len(candidates) == 1:
        return candidates[0][1]
    # disambiguate by lambda/lr/ep flags
    flags = []
    if "lambda0.2" in config or "lambda=0.2" in config:
        flags.append("lambda0.2")
    elif "lambda0.0" in config or "lambda=0.0" in config:
        flags.append("lambda0")
    if "lr=5e-4" in config or "lr5e-4" in nc:
        flags.append("lr5e-4")
    if "ep=300" in config:
        flags.append("ep300")
    if not flags and len(candidates) == 1:
        return candidates[0][1]
    for cfg, val in candidates:
        ok = True
        for f in flags:
            if f == "lambda0.2" and "lambda0.2" not in cfg:
                ok = False
            if f == "lambda0" and "lambda0.2" in cfg:
                ok = False
            if f == "lr5e-4" and "lr5e-4" not in cfg and "lr=5e-4" not in val.get("config", "").lower():
                ok = False
            if f == "ep300" and "ep300" not in cfg and "ep=300" not in val.get("config", "").lower():
                ok = False
        if ok:
            return val
    return None


def compare_with_orfc(results: list[JobResult], out_path: Path, orfc_results: Path) -> None:
    csv_map = {
        ("vitl14", "cls"): orfc_results / "dinov2_vitl14_cls.csv",
        ("vitl14", "seg"): orfc_results / "dinov2_vitl14_seg.csv",
        ("vitg14", "cls"): orfc_results / "dinov2_vitg14_cls.csv",
        ("vitg14", "seg"): orfc_results / "dinov2_vitg14_seg.csv",
    }
    orfc_parsed = {k: parse_orfc_ours(v, k[1]) for k, v in csv_map.items()}

    lines = [f"Soft-PQ vs ORFC CSV comparison — {datetime.now().isoformat()}", ""]
    match = mismatch = missing = 0
    tol_metric = 0.2
    tol_bpfp = 0.01

    for r in results:
        key = (r.bb, r.task)
        ref = find_orfc_match(r.layer, r.config, orfc_parsed[key])
        if ref is None:
            missing += 1
            lines.append(f"[MISSING_REF] {r.bb} {r.layer} {r.task} config={r.config}")
            continue
        metric_key = "acc" if r.task == "cls" else "miou"
        d_metric = abs(r.metric - ref[metric_key])
        d_bpfp = abs(r.bpfp - ref["bpfp"])
        ok = d_metric <= tol_metric and d_bpfp <= tol_bpfp
        if ok:
            match += 1
        else:
            mismatch += 1
            lines.append(
                f"[DIFF] {r.bb} {r.layer} {r.task} {r.config}\n"
                f"  ours: BPFP={r.bpfp:.4f} metric={r.metric:.2f}\n"
                f"  orfc: BPFP={ref['bpfp']:.4f} metric={ref[metric_key]:.2f}\n"
                f"  delta: BPFP={d_bpfp:.4f} metric={d_metric:.2f}"
            )

    lines.insert(2, f"Summary: match={match} mismatch={mismatch} missing_ref={missing} total={len(results)}")
    lines.insert(3, f"Tolerance: metric±{tol_metric}% BPFP±{tol_bpfp}")
    lines.insert(4, "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote comparison report: {out_path}")
    print(f"match={match} mismatch={mismatch} missing_ref={missing}")


def write_reproduction_csv(results: list[JobResult], out_path: Path) -> None:
    sections = [
        ("vitl14", "cls", "dinov2 ViT-L/14 | Classification | Acc@1", "Acc@1 (%)"),
        ("vitl14", "seg", "dinov2 ViT-L/14 | Segmentation | mIoU", "mIoU (%)"),
        ("vitg14", "cls", "dinov2 ViT-G/14 | Classification | Acc@1", "Acc@1 (%)"),
        ("vitg14", "seg", "dinov2 ViT-G/14 | Segmentation | mIoU", "mIoU (%)"),
    ]
    layer_order = {
        "vitl14": ["blk05", "blk10", "blk15", "blk20"],
        "vitg14": ["blk09", "blk19", "blk29"],
    }

    lines = [
        "Soft-PQ Online Evaluation Reproduction Results",
        "vitl14: logs/orfc_2446/dinov2/",
        "vitg14: logs/orfc_2446/dinov2/",
        "",
    ]

    for bb, task, title, metric_col in sections:
        lines.append(title)
        lines.append(f"block,config,BPFP,{metric_col}")
        subset = [r for r in results if r.bb == bb and r.task == task]
        order = {ly: i for i, ly in enumerate(layer_order[bb])}
        subset.sort(key=lambda r: (order.get(r.layer, 99), r.config))
        for r in subset:
            lines.append(f'{r.layer},"{r.config}",{r.bpfp:.4f},{r.metric:.2f}')
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote reproduction CSV: {out_path}")


def main() -> None:
    project_root = get_project_root()
    default_results = project_root / "logs" / "orfc_2446" / "dinov2"
    parser = argparse.ArgumentParser(description="Collect published DINOv2 SoftPQ plan results")
    parser.add_argument("--results-dir", type=Path, default=default_results)
    parser.add_argument(
        "--orfc-results",
        type=Path,
        help="Optional directory containing the four legacy ORFC CSV files",
    )
    parser.add_argument(
        "--compare-out",
        type=Path,
        default=default_results / f"orfc_csv_comparison_{datetime.now():%Y%m%d}.txt",
    )
    parser.add_argument(
        "--repro-csv",
        type=Path,
        default=default_results / "soft_pq_reproduction_results.csv",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    results = collect_all(args.results_dir.resolve(), require_all=not args.allow_missing)
    print(f"Collected {len(results)} job results")
    if args.orfc_results:
        compare_with_orfc(results, args.compare_out, args.orfc_results)
    write_reproduction_csv(results, args.repro_csv)


if __name__ == "__main__":
    main()

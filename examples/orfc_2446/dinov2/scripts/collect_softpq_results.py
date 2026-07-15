#!/usr/bin/env python3
"""Collect Soft-PQ online eval results and compare with ORFC CSV benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ORFC2446_DIR = Path(__file__).resolve().parents[1]
ORFC_RESULTS = PROJECT_ROOT.parent / "ORFC" / "coding" / "orfc" / "results"

W_L = "weights/orfc_2446/dinov2_vitl14_ori"
W_G = "weights/orfc_2446/dinov2_vitg14_ori"

SLOT_MAP = {
    "blk05": "slot06", "blk10": "slot11", "blk15": "slot16", "blk20": "slot21",
    "blk09": "slot10", "blk19": "slot20", "blk29": "slot30",
}

# Manual config labels aligned with soft_pq_reproduction_results.csv / ORFC Ours rows.
CONFIG_LABELS: dict[str, str] = {}


def _init_config_labels() -> None:
    def add(ckpt_name: str, label: str) -> None:
        CONFIG_LABELS[ckpt_name] = label

    # vitl14 cls+seg share same ckpt names
    pairs = [
        ("blk05_K4_emb32_bt1024_ws_lmbda0.0_tau0.5_lr0.0005_ep300_n5000_s42.npz",
         "K=4, e32, lambda=0.0, lr=5e-4, ep=300"),
        ("blk05_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=8, e32"),
        ("blk05_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=16, e32"),
        ("blk05_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz", "K=64, e32, lr=5e-4"),
        ("blk05_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e32"),
        ("blk05_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e16"),
        ("blk05_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e16"),
        ("blk10_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=4, e32"),
        ("blk10_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=8, e32"),
        ("blk10_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=16, e32"),
        ("blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e32, lambda=0.0"),
        ("blk10_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz", "K=256, e32, lambda=0.0, lr=5e-4"),
        ("blk10_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e32"),
        ("blk10_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e16"),
        ("blk10_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e16"),
        ("blk15_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=4, e32"),
        ("blk15_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=8, e32"),
        ("blk15_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=16, e32"),
        ("blk15_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz", "K=64, e32, lr=5e-4"),
        ("blk15_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e32"),
        ("blk15_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e16"),
        ("blk15_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e16, lambda=0.2"),
        ("blk15_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e16"),
        ("blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=8, e32"),
        ("blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz", "K=16, e32, lr=5e-4"),
        ("blk20_K32_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=32, e32"),
        ("blk20_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e32, lambda=0.0"),
        ("blk20_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz", "K=256, e32, lambda=0.0, lr=5e-4"),
        ("blk20_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e16"),
        # vitg14
        ("blk09_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=4, e32"),
        ("blk09_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=8, e32"),
        ("blk09_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=16, e32"),
        ("blk09_K64_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e32"),
        ("blk09_K256_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e32"),
        ("blk09_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e16"),
        ("blk19_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=4, e32"),
        ("blk19_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=8, e32"),
        ("blk19_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=16, e32"),
        ("blk19_K64_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e32"),
        ("blk19_K256_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=256, e32"),
        ("blk19_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e16"),
        ("blk29_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=4, e32"),
        ("blk29_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=8, e32"),
        ("blk29_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=16, e32"),
        ("blk29_K64_emb32_bt1536_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e32, lambda=0.0"),
        ("blk29_K256_emb32_bt1536_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz", "K=256, e32, lambda=0.0, lr=5e-4"),
        ("blk29_K64_emb16_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz", "K=64, e16"),
    ]
    for ckpt, label in pairs:
        add(ckpt, label)


_init_config_labels()


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


def parse_jobs() -> list[tuple[str, str, str, str]]:
    text = (ORFC2446_DIR / "scripts" / "run_online_eval_all.sh").read_text()
    jobs = []
    for m in re.finditer(r'JOBS\+\=\("([^"]+)"\)', text):
        parts = m.group(1).split()
        if len(parts) >= 4:
            bb, layer, task, ck = parts[0], parts[1], parts[2], parts[3]
            ck = ck.replace("${W_L}", W_L).replace("${W_G}", W_G)
            jobs.append((bb, layer, task, ck))
    return jobs


def codec_tag(ckpt: str) -> str:
    """Result subdir tag matching run_eval_orfc_2446.resolve_result_subdir."""
    stem = Path(ckpt).stem
    m = re.search(r"_K(\d+)_emb(\d+)_", stem)
    if not m:
        raise ValueError(f"Cannot parse codec tag from {ckpt}")
    tag = f"K{m.group(1)}e{m.group(2)}"
    for key in ("lmbda", "tau", "lr", "ep"):
        mm = re.search(rf"_{key}([0-9.]+)", stem)
        if mm:
            tag += f"_{key}{mm.group(1)}"
    return tag


def default_result_path(bb: str, layer: str, task: str, ckpt: str) -> Path:
    dataset = "imagenet-sel500" if task == "cls" else "voc2012-sel100"
    bb_path = f"dinov2-{bb}-slide" if task == "seg" else f"dinov2-{bb}"
    slot = SLOT_MAP[layer]
    return PROJECT_ROOT / f"eval_results/SoftPQ/{dataset}/{bb_path}/{slot}/{codec_tag(ckpt)}/result.json"


def resolve_result_path(bb: str, layer: str, task: str, ckpt: str) -> Path:
    return default_result_path(bb, layer, task, ckpt)


def load_metrics(path: Path, task: str) -> tuple[float, float]:
    data = json.loads(path.read_text())
    res = data.get("results", data)
    bpfp = float(res["bpfp"])
    if task == "cls":
        metric = float(res["cls_top-1"])
    else:
        miou = float(res["semseg_mIoU"])
        metric = miou * 100.0 if miou <= 1.0 else miou
    return bpfp, metric


def collect_all() -> list[JobResult]:
    out: list[JobResult] = []
    missing = []
    for bb, layer, task, ckpt in parse_jobs():
        rp = resolve_result_path(bb, layer, task, ckpt)
        if not rp.exists():
            missing.append((ckpt, str(rp)))
            continue
        bpfp, metric = load_metrics(rp, task)
        config = CONFIG_LABELS.get(Path(ckpt).name, Path(ckpt).stem)
        out.append(JobResult(bb, layer, task, ckpt, config, bpfp, metric, str(rp)))
    if missing:
        print(f"ERROR: {len(missing)} missing results", file=sys.stderr)
        for ck, p in missing[:10]:
            print(f"  {ck} -> {p}", file=sys.stderr)
        sys.exit(1)
    return out


def parse_orfc_ours(csv_path: Path, task: str) -> dict[tuple[str, str], dict]:
    """Parse ORFC CSV Ours section -> {(layer, config_norm): {bpfp, metric}}."""
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ours_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Ours"))
    block_nums = [b.strip() for b in lines[ours_idx + 1].split(",") if b.strip().isdigit()]
    blocks = [f"blk{int(b):02d}" for b in block_nums]

    results: dict[tuple[str, str], dict] = {}
    metric_key = "acc" if task == "cls" else "miou"

    for ln in lines[ours_idx + 3:]:
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
                "bpfp": bpfp, metric_key: metric, "config": config,
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
        (cfg, val) for (ly, cfg), val in orfc.items()
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


def compare_with_orfc(results: list[JobResult], out_path: Path) -> None:
    csv_map = {
        ("vitl14", "cls"): ORFC_RESULTS / "dinov2_vitl14_cls.csv",
        ("vitl14", "seg"): ORFC_RESULTS / "dinov2_vitl14_seg.csv",
        ("vitg14", "cls"): ORFC_RESULTS / "dinov2_vitg14_cls.csv",
        ("vitg14", "seg"): ORFC_RESULTS / "dinov2_vitg14_seg.csv",
    }
    orfc_parsed = {
        k: parse_orfc_ours(v, k[1]) for k, v in csv_map.items()
    }

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
    out_path.write_text("\n".join(lines))
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
      "vitl14: eval_results/SoftPQ/",
      "vitg14: eval_results/SoftPQ/",
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

  out_path.write_text("\n".join(lines))
  print(f"Wrote reproduction CSV: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-out", type=Path,
                        default=PROJECT_ROOT / "eval_results" / f"orfc_csv_comparison_{datetime.now():%Y%m%d}.txt")
    parser.add_argument("--repro-csv", type=Path,
                        default=PROJECT_ROOT / "eval_results" / "soft_pq_reproduction_results.csv")
    args = parser.parse_args()

    results = collect_all()
    print(f"Collected {len(results)} job results")
    compare_with_orfc(results, args.compare_out)
    write_reproduction_csv(results, args.repro_csv)


if __name__ == "__main__":
    main()

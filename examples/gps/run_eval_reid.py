"""Evaluate GPS multi-view retrieval on one configured dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.gps.config import load_config
from examples.gps.reid.dataloader import make_dataloader
from examples.gps.reid.evaluator import evaluate_model
from examples.gps.reid.legacy_config import build_legacy_config
from examples.gps.reid.model import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPS ReID evaluation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.data_root = str(args.data_root.resolve())
    if args.checkpoint is not None:
        cfg.checkpoint = str(args.checkpoint.resolve())
    if args.subset is not None:
        cfg.dataset.subset = args.subset

    legacy_cfg = build_legacy_config(cfg)
    loaders = make_dataloader(legacy_cfg)
    _, _, _, query_loader, gallery_loader, _, num_query, num_classes, camera_num, view_num = loaders
    model = make_model(legacy_cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(str(cfg.checkpoint))
    result = {
        "dataset": str(cfg.dataset.name),
        "subset": cfg.dataset.subset,
        "checkpoint": str(cfg.checkpoint),
        **evaluate_model(legacy_cfg, model, query_loader, gallery_loader, num_query),
    }
    output = args.output or (PROJECT_ROOT / "logs" / "gps" / str(cfg.dataset.name) / "result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

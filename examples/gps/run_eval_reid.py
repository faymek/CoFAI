"""Evaluate GPS multi-view retrieval on one configured dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.gps.config import load_config
from examples.gps.model import build_codec_model
from examples.gps.reid.dataloader import make_dataloader
from examples.gps.reid.evaluator import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPS ReID evaluation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root is not None:
        cfg.data_root = str(args.data_root.resolve())
    if args.checkpoint is not None:
        cfg.checkpoint = str(args.checkpoint.resolve())
    if args.batch_size is not None:
        if args.batch_size <= 0 or args.batch_size % 3:
            raise ValueError("GPS eval batch size must be a positive multiple of 3")
        cfg.evaluation.batch_size = args.batch_size

    loaders = make_dataloader(cfg)
    model = build_codec_model(
        cfg,
        camera_num=loaders.camera_num,
        view_num=loaders.view_num,
    )
    result = {
        "dataset": str(cfg.name),
        "checkpoint": str(cfg.checkpoint),
        **evaluate_model(
            cfg,
            model,
            loaders.query,
            loaders.gallery,
            loaders.num_query,
        ),
    }
    output = args.output or (
        PROJECT_ROOT / "logs" / "gps" / str(cfg.name) / "result.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

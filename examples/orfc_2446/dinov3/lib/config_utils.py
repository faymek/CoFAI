"""Config loading for the DINOv3 PQFC offline pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml


def resolve_project_root() -> Path:
    import os

    raw = str(os.environ.get("PROJECT_ROOT", "")).strip()
    if raw:
        env = Path(raw)
        if env.is_dir():
            return env.resolve()
    # examples/orfc_2446/dinov3/lib/config_utils.py -> parents[4] = CoFAI root
    return Path(__file__).resolve().parents[4]


def load_config(config_path: Path | None = None) -> dict:
    root = resolve_project_root()
    if config_path is None:
        config_path = root / "examples/orfc_2446/dinov3/configs/dinov3_blk23.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def _resolve(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((root / path).resolve())

    cfg["_project_root"] = str(root)
    paths = cfg["paths"]
    for key in (
        "train_feat_dir",
        "val_feat_root",
        "weights_dir",
        "results_dir",
        "backbone",
        "semseg_head",
        "depth_head",
    ):
        paths[key] = _resolve(paths[key])

    for task in ("semseg", "depth"):
        ds = cfg["datasets"][task]
        ds["root"] = _resolve(ds["root"])

    return cfg


def task_feat_dir(cfg: dict, task: str) -> Path:
    return Path(cfg["paths"]["val_feat_root"]) / task

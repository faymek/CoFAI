"""Dataset / model helpers for the offline VTM pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from cofai.backbone import Dinov3TimmBackbone
from cofai.datasets import NYUDepthDataset, SegmentationDataset
from cofai.engine.builder import build_transforms
from cofai.engine.registry import instantiate_class
from cofai.heads import Dinov3DepthHead, Dinov3SegmentationHead
from torch.utils.data import Dataset


class _TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def resolve_project_root() -> Path:
    env = Path(str(__import__("os").environ.get("PROJECT_ROOT", "")))
    if env.is_dir():
        return env.resolve()
    # examples/vtm/lib -> CoFAI root
    return Path(__file__).resolve().parents[3]


def load_config(config_path: Path | None = None) -> dict:
    root = resolve_project_root()
    if config_path is None:
        config_path = root / "examples/vtm/configs/dinov3_slot24.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    def _resolve(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((root / path).resolve())

    cfg["_project_root"] = str(root)
    cfg["paths"]["backbone"] = _resolve(cfg["paths"]["backbone"])
    cfg["paths"]["semseg_head"] = _resolve(cfg["paths"]["semseg_head"])
    cfg["paths"]["depth_head"] = _resolve(cfg["paths"]["depth_head"])
    cfg["paths"]["feat_root"] = _resolve(cfg["paths"]["feat_root"])
    cfg["paths"]["results_dir"] = _resolve(cfg["paths"]["results_dir"])

    vtm_dir = Path(cfg["vtm"]["dir"])
    if not vtm_dir.is_absolute():
        vtm_dir = (root / vtm_dir).resolve()
    cfg["vtm"]["encoder_path"] = str(vtm_dir / cfg["vtm"]["encoder"])
    cfg["vtm"]["decoder_path"] = str(vtm_dir / cfg["vtm"]["decoder"])
    cfg["vtm"]["cfg_path"] = str(vtm_dir / cfg["vtm"]["cfg"])

    for task in ("semseg", "depth"):
        ds = cfg["datasets"][task]
        ds["root"] = _resolve(ds["root"])
        if task == "semseg":
            ds["bypass_plan"] = _resolve(ds["bypass_plan"])
        else:
            ds["bypass_plan"] = _resolve(ds["bypass_plan"])
    return cfg


def load_subset(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    with open(path) as f:
        stems = {line.strip() for line in f if line.strip()}
    return stems


def sample_stem(sample: dict, task: str, *, nyu_root: str | None = None) -> str:
    meta = sample["meta"]
    if task == "semseg":
        return str(meta["img_name"])
    p = Path(meta["img_path"])
    root = Path(nyu_root) if nyu_root else p.parents[2]
    rel = p.relative_to(root)
    parts = list(rel.parts)
    if parts and parts[0] in ("train", "test"):
        parts = parts[1:]
    return "__".join(parts).replace(".jpg", "")


def build_dataset(task: str, cfg: dict):
    ds_cfg = dict(cfg["datasets"][task])
    test_tf = ds_cfg.pop("test_transforms")
    ds_cfg.pop("meter", None)
    ds_cfg.pop("bypass_plan", None)
    ds_cfg.pop("metric_key", None)
    transform = build_transforms(test_tf)
    ds_type = ds_cfg.pop("type")
    if ds_type == "SegmentationDataset":
        ds = SegmentationDataset(transform=transform, **ds_cfg)
    elif ds_type == "NYUDepthDataset":
        ds = NYUDepthDataset(**ds_cfg)
        ds = _TransformDataset(ds, transform)
    else:
        raise ValueError(f"Unsupported dataset type: {ds_type}")
    return ds


def build_backbone(cfg: dict, device: torch.device) -> Dinov3TimmBackbone:
    return Dinov3TimmBackbone(
        model_size="large",
        img_size=512,
        patch_size=cfg["patch_size"],
        dynamic_size=True,
        slot=cfg["slot"],
        n_last_blocks=1,
        cast_dtype="float32",
        pretrained=False,
        ckpt_path=cfg["paths"]["backbone"],
        device=str(device),
    ).eval().to(device)


def build_head(task: str, cfg: dict):
    if task == "semseg":
        return Dinov3SegmentationHead(
            in_channels=[cfg["embed_dim"]],
            in_index=[0],
            input_transform="resize_concat",
            channels=cfg["embed_dim"],
            num_classes=150,
            patch_size=cfg["patch_size"],
            checkpoint=cfg["paths"]["semseg_head"],
        )
    if task == "depth":
        return Dinov3DepthHead(
            in_channels=[cfg["embed_dim"]],
            min_depth=0.001,
            max_depth=10.0,
            n_output_channels=256,
            use_backbone_norm=True,
            use_batchnorm=True,
            use_cls_token=False,
            bins_strategy="linear",
            norm_strategy="linear",
            checkpoint=cfg["paths"]["depth_head"],
        )
    raise ValueError(task)


def build_meter(task: str, cfg: dict):
    return instantiate_class(cfg["datasets"][task]["meter"])


def task_feat_dir(cfg: dict, task: str) -> Path:
    return Path(cfg["paths"]["feat_root"]) / task

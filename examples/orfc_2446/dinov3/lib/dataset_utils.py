"""Dataset and model builders for proposal-local DINOv3 replay."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from cofai.backbone import Dinov3TimmBackbone
from cofai.datasets import NYUDepthDataset, SegmentationDataset
from cofai.engine.builder import build_transforms
from cofai.engine.registry import instantiate_class
from cofai.heads import Dinov3DepthHead, Dinov3SegmentationHead


class _TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return self.transform(sample) if self.transform is not None else sample


def load_subset(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    with open(path) as subset_file:
        return {line.strip() for line in subset_file if line.strip()}


def sample_stem(sample: dict, task: str, *, nyu_root: str | None = None) -> str:
    meta = sample["meta"]
    if task == "semseg":
        return str(meta["img_name"])

    image_path = Path(meta["img_path"])
    root = Path(nyu_root) if nyu_root else image_path.parents[2]
    relative = image_path.relative_to(root)
    parts = list(relative.parts)
    if parts and parts[0] in {"train", "test"}:
        parts = parts[1:]
    return "__".join(parts).removesuffix(".jpg")


def build_dataset(task: str, cfg: dict):
    dataset_cfg = dict(cfg["datasets"][task])
    transforms = build_transforms(dataset_cfg.pop("test_transforms"))
    for offline_only_key in ("meter", "metric_key"):
        dataset_cfg.pop(offline_only_key, None)

    dataset_type = dataset_cfg.pop("type")
    if dataset_type == "SegmentationDataset":
        return SegmentationDataset(transform=transforms, **dataset_cfg)
    if dataset_type == "NYUDepthDataset":
        return _TransformDataset(NYUDepthDataset(**dataset_cfg), transforms)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def build_backbone(cfg: dict, device: torch.device) -> Dinov3TimmBackbone:
    return (
        Dinov3TimmBackbone(
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
        )
        .eval()
        .to(device)
    )


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
    raise ValueError(f"Unsupported task: {task}")


def build_meter(task: str, cfg: dict):
    return instantiate_class(cfg["datasets"][task]["meter"])

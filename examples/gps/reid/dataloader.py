from __future__ import annotations

from dataclasses import dataclass

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .datasets.bases import ImageDataset
from .datasets.muri import MuRI
from .datasets.sampler_multiview import MultiViewSampler
from .datasets.veri import VeRi

_DATASET_FACTORIES = {
    "veri": VeRi,
    "muri": MuRI,
}


@dataclass(frozen=True)
class GPSDataLoaders:
    """Evaluation inputs and model dimensions for one GPS dataset."""

    query: DataLoader
    gallery: DataLoader
    num_query: int
    num_classes: int
    camera_num: int
    view_num: int


def val_collate_fn(batch):
    imgs, pids, camids, viewids, sceneids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return (
        torch.stack(imgs, dim=0),
        pids,
        camids,
        camids_batch,
        viewids,
        sceneids,
        img_paths,
    )


def make_dataloader(cfg) -> GPSDataLoaders:
    """Build only the multi-view query and gallery loaders used by evaluation."""
    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    try:
        dataset_factory = _DATASET_FACTORIES[cfg.DATASETS.NAMES]
    except KeyError as exc:
        choices = ", ".join(sorted(_DATASET_FACTORIES))
        raise ValueError(
            f"Unsupported GPS dataset {cfg.DATASETS.NAMES!r}; "
            f"expected one of: {choices}"
        ) from exc
    dataset = dataset_factory(root=cfg.DATASETS.ROOT_DIR)

    query_sampler = MultiViewSampler(dataset.query, cfg.TEST.IMS_PER_BATCH, 3)
    query_loader = DataLoader(
        ImageDataset(dataset.query, val_transforms),
        batch_size=cfg.TEST.IMS_PER_BATCH,
        sampler=query_sampler,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn,
    )
    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery, val_transforms),
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn,
    )

    camera_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    if cfg.DATASETS.NAMES == "muri":
        camera_num = 0
        view_num = 0
    return GPSDataLoaders(
        query=query_loader,
        gallery=gallery_loader,
        num_query=len(query_sampler) // query_sampler.num_instances,
        num_classes=dataset.num_train_pids,
        camera_num=camera_num,
        view_num=view_num,
    )

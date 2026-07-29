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
    """Evaluation inputs and SIE dimensions for one GPS dataset."""

    query: DataLoader
    gallery: DataLoader
    num_query: int
    camera_num: int
    view_num: int


def val_collate_fn(batch):
    imgs, pids, camids, viewids, sceneids, img_paths = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        {
            "pid": pids,
            "camid": camids,
            "camera_label": torch.tensor(camids, dtype=torch.int64),
            "view_label": torch.tensor(viewids, dtype=torch.int64),
            "sceneid": sceneids,
            "img_path": img_paths,
        },
    )


def make_dataloader(cfg) -> GPSDataLoaders:
    """Build only the multi-view query and gallery loaders used by evaluation."""
    model_cfg = cfg.model
    evaluation_cfg = cfg.evaluation
    dataset_name = str(cfg.dataset.name)
    val_transforms = T.Compose(
        [
            T.Resize(model_cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=model_cfg.pixel_mean, std=model_cfg.pixel_std),
        ]
    )
    try:
        dataset_factory = _DATASET_FACTORIES[dataset_name]
    except KeyError as exc:
        choices = ", ".join(sorted(_DATASET_FACTORIES))
        raise ValueError(
            f"Unsupported GPS dataset {dataset_name!r}; expected one of: {choices}"
        ) from exc
    dataset = dataset_factory(root=cfg.data_root)

    batch_size = int(evaluation_cfg.batch_size)
    workers = int(evaluation_cfg.workers)
    query_sampler = MultiViewSampler(dataset.query, batch_size, 3)
    query_loader = DataLoader(
        ImageDataset(dataset.query, val_transforms),
        batch_size=batch_size,
        sampler=query_sampler,
        shuffle=False,
        num_workers=workers,
        collate_fn=val_collate_fn,
    )
    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery, val_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=val_collate_fn,
    )

    camera_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    if dataset_name == "muri":
        camera_num = 0
        view_num = 0
    return GPSDataLoaders(
        query=query_loader,
        gallery=gallery_loader,
        num_query=len(query_sampler) // query_sampler.num_instances,
        camera_num=camera_num,
        view_num=view_num,
    )

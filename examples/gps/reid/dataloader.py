import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .datasets.bases import ImageDataset
from timm.data.random_erasing import RandomErasing


from .datasets.sampler_multiview import MultiViewSampler
import torch.distributed as dist
from .datasets.veri import VeRi
import numpy as np
import random
from .datasets.muri import MuRI

from .datasets.veri_wild import VeRIWild
from .datasets.veri_wild2 import VeRIWild2

__factory = {
    'veri': VeRi,
    'muri': MuRI,
    'veri_wild': VeRIWild,
    'veri_wild2': VeRIWild2,
}

def train_collate_fn(batch):
    imgs, pids, camids, viewids, sceneids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    sceneids = torch.tensor(sceneids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, sceneids

def val_collate_fn(batch):
    imgs, pids, camids, viewids, sceneids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, sceneids, img_paths



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def make_dataloader(cfg):

    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset_name = cfg.DATASETS.NAMES
    dataset_kwargs = {"root": cfg.DATASETS.ROOT_DIR}
    if dataset_name in {'veri_wild', 'veri_wild2'} and cfg.DATASETS.SUBSET is not None:
        dataset_kwargs["subset"] = cfg.DATASETS.SUBSET
    dataset = __factory[dataset_name](**dataset_kwargs)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    # Training samplers are intentionally outside this evaluation example.
    train_loader = None


    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    query_set = ImageDataset(dataset.query, val_transforms)
    gallery_set = ImageDataset(dataset.gallery, val_transforms)

    val_loader_multi = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    query_multi_sampler = MultiViewSampler(dataset.query, cfg.TEST.IMS_PER_BATCH, 3)
    query_loader_multi = DataLoader(
        query_set, batch_size=cfg.TEST.IMS_PER_BATCH,
        sampler=query_multi_sampler,
        shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    num_query_single = len(dataset.query)
    num_query_multi = len(query_multi_sampler) // query_multi_sampler.num_instances

    gallery_loader_multi = DataLoader(
        gallery_set, batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    val_loader_normal = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    if 'muri' in cfg.DATASETS.NAMES:
        cam_num = 0
        view_num = 0
    return train_loader, val_loader_normal, val_loader_multi, query_loader_multi, gallery_loader_multi, num_query_single, num_query_multi, num_classes, cam_num, view_num

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
import importlib
import tqdm
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import (
    Compose,
    CenterCrop,
    ToTensor,
    PILToTensor,
    RandomResizedCrop,
    RandomCrop,
)

from cofai.datasets import *
from cofai.heads import *
from cofai.models import *
from cofai.metrics import *
from cofai.utils.debug import extract_shapes, tensor_hash
from dotenv import load_dotenv

load_dotenv()

# Disable Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# comment this to keep same behaviour as training
# torch.backends.cudnn.deterministic = True
# torch.set_num_threads(1)


def get_obj_from_str(string, reload=False):
    if "." in string:
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    else:
        return getattr(sys.modules[__name__], string)


def instantiate_class(config, **kwargs):
    config = config.copy()
    if "type" in config:
        cls = config.pop("type")
        obj = get_obj_from_str(cls)
        return obj(**config, **kwargs)
    else:
        print(config)
        raise KeyError("Expected key `type` to instantiate.")


def instantiate_transforms(config, **kwargs):
    if "type" in config:
        cls = config.pop("type")
        obj = get_obj_from_str(cls)
        if "transforms" in config:
            transforms_list = config.pop("transforms")
            transforms = []
            for t in transforms_list:
                transforms.append(instantiate_class(t))
            return obj(transforms=transforms, **config, **kwargs)
        return obj(**config, **kwargs)
    else:
        raise KeyError("Expected key `type` to instantiate.")


@torch.inference_mode()
def eval_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate_class(cfg.model).to(device)
    model.eval()

    train_transforms = instantiate_transforms(cfg.train_transforms)

    train_dataset = instantiate_class(cfg.train_dataset, transform=train_transforms)
    batch_size = 1 if cfg.extract.batch_size is None else cfg.extract.batch_size
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
    )

    val_dataset = instantiate_class(cfg.val_dataset, transform=train_transforms)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
    )

    # 创建输出目录
    train_dir = f"features/{cfg.extract.name}/train"
    val_dir = f"features/{cfg.extract.name}/val"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_batches = min(cfg.extract.split_size.train, len(train_dataloader))
    val_batches = min(cfg.extract.split_size.val, len(val_dataloader))

    for dataloader, split_dir, split_batches in zip(
        [train_dataloader, val_dataloader],
        [train_dir, val_dir],
        [train_batches, val_batches],
    ):
        print(f"Extracting features to {split_dir}...")
        for i, data in tqdm(enumerate(dataloader), total=split_batches):
            if i >= split_batches:
                break
            x_uint8 = data.to(device)
            x = x_uint8 / 255.0
            return_data = model.extract_feature(x)
            if cfg.extract.save_x_uint8:
                return_data["x_uint8"] = x_uint8
            return_data["x_shape"] = x.shape
            write_data = {}
            if cfg.extract.batch_size is None:
                for k, v in return_data.items():
                    if isinstance(v, torch.Tensor):
                        write_data[k] = v[0]
                    elif isinstance(v, torch.Size):
                        write_data[k] = v[1:]
                    else:
                        write_data[k] = v
            else:
                write_data = return_data
                raise NotImplementedError("Not implemented")

            if i == 0:
                print(extract_shapes(write_data))

            torch.save(write_data, f"{split_dir}/{i:08d}.pt")


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="MPC模型评估脚本")
    parser.add_argument("config", type=str)
    return parser


if __name__ == "__main__":
    parser = setup_args()
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    eval_model(config)


"""
CUDA_VISIBLE_DEVICES=1 python examples/mpc/offline_extract.py --config examples/mpc/train_config/MPC12-v3-large-extract.yaml
"""

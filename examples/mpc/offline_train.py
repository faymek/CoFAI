import os
import re
import sys
import glob
import math
import argparse
import random
import shutil
from datetime import datetime
import fnmatch
import logging
from pathlib import Path
from omegaconf import OmegaConf
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    CenterCrop,
    ToTensor,
    RandomResizedCrop,
    RandomCrop,
)

from compressai.registry import MODELS
from CoFAI.cofai.models import *
from CoFAI.cofai.heads import Dinov2ClassifierHead
from CoFAI.cofai.losses.loss import *
from CoFAI.cofai.datasets import ImageFolder
from CoFAI.cofai.utils.utils import setup_logger
from CoFAI.cofai.utils.tensor_ops import tensor2image
from CoFAI.cofai.utils.utils import rename_key_by_rules
from CoFAI.cofai.datasets.feature import (
    FeatureDictPerSampleFolder,
    feature_dict_collate_fn,
)
from dotenv import load_dotenv

load_dotenv()


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
    if "type" in config:
        cls = config.pop("type")
        obj = get_obj_from_str(cls)
        return obj(**config, **kwargs)
    else:
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


def match_pattern(pattern, name):
    if pattern.startswith("!"):
        return not fnmatch.fnmatch(name, pattern[1:])
    else:
        return fnmatch.fnmatch(name, pattern)


def match_patterns_and(patterns, name):
    for pattern in patterns:
        if not match_pattern(pattern, name):
            return False
    return True


def mse_to_psnr(mse):
    if mse < 1e-10:
        return 100.0
    else:
        return -10 * math.log10(mse)


def rename_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # 处理 "latent_codec.hyper.entropy_bottleneck._bias.0" 到 "latent_codec.hyper.entropy_bottleneck._bias0" 的情况
        if "latent_codec.hyper.entropy_bottleneck._bias." in key:
            new_key = key.replace(".biases.", "_bias")
        # 处理其他类似的模式
        elif "latent_codec.hyper.entropy_bottleneck.matrices." in key:
            new_key = key.replace(".matrices.", "_matrix")
        elif "latent_codec.hyper.entropy_bottleneck._factor." in key:
            new_key = key.replace(".factors.", "_factor")
        else:
            new_key = key
        new_state_dict[new_key] = value
        if new_key != key:
            print(f"Renamed {key} to {new_key}")
            new_state_dict.pop(key)
    return new_state_dict


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MoniterAverageMeter:
    def __init__(self):
        self.monitor = None
        self.count = 0

    def update(self, data, n=1):
        if self.monitor is None:
            self.monitor = data.copy()
            # 确保所有数据值都可以进行数学运算
            for name, item in data.items():
                assert isinstance(item, (int, float)), (
                    f"数据项 '{name}' 的值 {item} 不是数值类型"
                )
        else:
            for name, item in data.items():
                self.monitor[name] += item
        self.count += n

    def average(self):
        monitor_avg = {}
        for name in self.monitor:
            monitor_avg[name] = self.monitor[name] / self.count
        return monitor_avg


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def rename_key(key: str) -> str:
    """Rename state_dict key."""

    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def load_various_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")

    if "network" in checkpoint:
        state_dict = checkpoint["network"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.current_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

    def run(self):
        try:
            self._run()
        except FileExistsError as e:
            raise e
        except KeyboardInterrupt as e:
            self.delete_lock_file()
            raise e
        except Exception as e:
            self.delete_lock_file()
            raise e

    def _run(self):
        self.device = (
            "cuda" if self.cfg.engine.cuda and torch.cuda.is_available() else "cpu"
        )
        self.setup_dirs()
        self.setup_loggers()
        if "offline" in self.cfg.misc and self.cfg.misc.offline:
            self.setup_offline_data()
        else:
            self.setup_data()
        self.setup_models()
        if self.cfg.misc.resume:
            self.resume_checkpoint()
        for epoch in range(self.current_epoch, self.cfg.misc.epochs):
            self.current_epoch = epoch
            self.train_epoch()
            loss = self.test_epoch()
            self.save_checkpoint(loss)

    def setup_dirs(self):  # 创建保存目录，并创建锁文件，保存config.yaml
        save_dir = f"exp/{self.cfg.exp.name}/Q{self.cfg.exp.quality}"
        self.save_dir = save_dir

        config_path = f"{save_dir}/config.yaml"
        ckpt_path = f"{save_dir}/runner.last.pth.tar"
        lock_file = f"{save_dir}/LOCK"
        if self.cfg.misc.resume:
            if os.path.exists(config_path) and os.path.exists(ckpt_path):
                print(f"任务 {lock_file} 未结束，继续训练")
                return
            else:
                raise FileNotFoundError(
                    f"配置文件 {config_path} 或 checkpoint {ckpt_path} 不存在，无法继续训练"
                )
        if os.path.exists(save_dir):
            if os.path.exists(lock_file):
                raise FileExistsError(
                    f"任务 {lock_file} 未结束，请重命名 exp.name 或者删除 LOCK 文件"
                )
            else:  # rename to timestamp
                if os.path.exists(config_path):
                    timestamp = datetime.fromtimestamp(os.path.getctime(config_path))
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                else:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir_timestamp = f"{save_dir}__{timestamp_str}"
                shutil.move(save_dir, save_dir_timestamp)
        os.makedirs(save_dir, exist_ok=True)

        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))

        config_path = os.path.join(self.save_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(self.cfg, f)

    def delete_lock_file(self):
        lock_file = os.path.join(self.save_dir, "LOCK")
        if os.path.exists(lock_file):
            os.remove(lock_file)

    def setup_loggers(self):
        # tb_dir = f"{self.save_dir}/tb_logger"
        # os.makedirs(tb_dir, exist_ok=True)
        # # self.tb_logger = SummaryWriter(log_dir=tb_dir)

        setup_logger(
            "main", self.save_dir, "main", level=logging.INFO, screen=True, tofile=True
        )
        self.logger = logging.getLogger("main")
        self.logger.info(f"EXP: {self.cfg.exp.name}")

    def setup_offline_data(self):
        train_dataset = instantiate_class(self.cfg.train_dataset)
        test_dataset = instantiate_class(self.cfg.val_dataset)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.misc.batch_size,
            num_workers=self.cfg.misc.num_workers,
            shuffle=True,
            collate_fn=feature_dict_collate_fn,
            pin_memory=True,
        )

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.cfg.misc.test_batch_size,
            num_workers=self.cfg.misc.num_workers,
            shuffle=False,
            collate_fn=feature_dict_collate_fn,
            pin_memory=False,
        )

    def setup_models(self):
        self.model: nn.Module = instantiate_class(self.cfg.model).to(self.device)
        if self.cfg.load.path:
            self.load_checkpoint()
        self.criterion = instantiate_class(self.cfg.criterion).to(self.device)
        self.setup_optimizers()

    def load_checkpoint(self):
        checkpoint = torch.load(self.cfg.load.path, map_location="cpu")
        sd_org = checkpoint["state_dict"]
        if self.cfg.load.rules:
            sd_new = {}
            for org_key in sorted(sd_org.keys()):
                new_key = rename_key_by_rules(org_key, self.cfg.load.rules)
                if new_key != "":  # 为空表示删除该key
                    sd_new[new_key] = sd_org[org_key]
            sd_org = sd_new
        self.model.load_state_dict(sd_org, strict=self.cfg.load.strict)

    def setup_optimizers(self):
        main_param_names = set()
        aux_param_names = set()
        no_opt_params = set()
        # 从配置文件中获取匹配模式

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if match_patterns_and(self.cfg.tune.main_optimizer, n):
                main_param_names.add(n)
            elif match_patterns_and(self.cfg.tune.aux_optimizer, n):
                aux_param_names.add(n)
            else:
                no_opt_params.add(n)

        print("main optimizer:", main_param_names)
        print("aux optimizer:", aux_param_names)
        # if len(no_opt_params) > 0:
        #     self.logger.info("以下参数未加入任何优化器:")
        #     for n in sorted(no_opt_params):
        #         self.logger.info(f"  {n}")

        params_dict = dict(self.model.named_parameters())
        main_params = [params_dict[n] for n in sorted(main_param_names)]
        aux_params = [params_dict[n] for n in sorted(aux_param_names)]

        self.main_optimizer = instantiate_class(
            self.cfg.main_optimizer, params=main_params
        )
        self.aux_optimizer = instantiate_class(
            self.cfg.aux_optimizer, params=aux_params
        )
        self.main_scheduler = instantiate_class(
            self.cfg.main_scheduler, optimizer=self.main_optimizer
        )

    def train_batch(self, data):
        self.current_step += 1
        self.main_optimizer.zero_grad()
        self.aux_optimizer.zero_grad()

        # data = data.to(self.device)
        out_net = self.model.offline_forward(data, self.device)
        loss, monitor = self.criterion.forward(out_net, x=None, x_shape=data["x_shape"])
        loss.backward()
        if self.cfg.misc.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.misc.clip_max_norm
            )
        self.main_optimizer.step()

        aux_loss = self.model.aux_loss()
        aux_loss.backward()
        self.aux_optimizer.step()

        monitor["Aux"] = aux_loss.item()
        return monitor

    def train_epoch(self):
        self.model.train()

        pbar = tqdm(self.train_dataloader, desc=f"Train  {self.current_epoch}")
        for idx, data in enumerate(pbar):
            monitor = self.train_batch(data)
            desc = self.monitor_to_str(monitor)
            pbar.set_postfix_str(desc)
            if self.cfg.misc.check and self.current_step % 100 == 0:
                break

    def monitor_to_str(self, monitor):
        item_precision = {"loss": 3, "bpp": 3, "rec": 4}
        item_prints = []
        for name, item in monitor.items():
            prec = item_precision.get(name, 4)
            item_prints.append(f"{name} {item:.{prec}f}")
        return ", ".join(item_prints)

    def log_monitor(self, monitor, pbar=None):
        desc = self.monitor_to_str(monitor)
        if pbar is not None:
            pbar.set_postfix_str(desc)
        # self.tb_logger.add_scalar("Train/Loss", monitor["loss"], self.current_step)

    def val_epoch(self):
        self.model.eval()
        monitor_meter = MoniterAverageMeter()
        pbar = tqdm(self.val_dataloader, desc=f"Val  {self.current_epoch}")
        with torch.inference_mode():
            for idx, data in enumerate(pbar):
                data = data.to(self.device)
                out_net = self.model.offline_forward(data, self.device)
                loss, monitor = self.criterion.forward(out_net, x=None, x_shape=data["x_shape"])
                monitor["Aux"] = self.model.aux_loss().item()
                desc = self.monitor_to_str(monitor)
                pbar.set_postfix_str(desc)
                monitor_meter.update(monitor)

        monitor_avg = monitor_meter.average()
        desc = self.monitor_to_str(monitor_avg)
        self.logger.info(f"Val {self.current_epoch} Average: {desc}")
        return monitor_avg["loss"]

    def test_epoch(self):
        self.model.eval()
        monitor_meter = MoniterAverageMeter()

        # select at most 30 images for visual
        test_len = len(self.test_dataloader)
        if test_len > 30:
            idx_for_visual = list(range(0, test_len, test_len // 30))[:30]
        else:
            idx_for_visual = list(range(test_len))

        pbar = tqdm(self.test_dataloader, desc=f"Test  {self.current_epoch}")
        with torch.inference_mode():
            for idx, data in enumerate(pbar):
                out_net = self.model.offline_forward(data, self.device)
                loss, monitor = self.criterion.forward(out_net, x=None, x_shape=data["x_shape"])
                monitor["Aux"] = self.model.aux_loss().item()
                desc = self.monitor_to_str(monitor)
                pbar.set_postfix_str(desc)
                monitor_meter.update(monitor)
                if self.cfg.misc.visual and "x_hat" in out_net and "x" in data:
                    if idx in idx_for_visual:
                        self.log_images(idx, data["x"], out_net["x_hat"])

        monitor_avg = monitor_meter.average()
        desc = self.monitor_to_str(monitor_avg)
        self.logger.info(f"Test {self.current_epoch} Average: {desc}")
        return monitor_avg["loss"]

    def eval_dino(self):
        if self.model.num_out_layers == 4:
            clf_result = run_linear_s_4(self.model.dino_eval_linear)
        elif self.model.num_out_layers == 1:
            clf_result = run_linear_s_1(self.model.dino_eval_linear)
        else:
            raise ValueError("model must have .num_out_layers equal 1 or 4")
        self.logger.info(f"DINO {self.current_epoch}: {clf_result}")
        return clf_result

    def log_images(self, idx, org_tensor, rec_tensor):
        image_dir = f"{self.save_dir}/image/{self.current_epoch:03d}"
        # ori_path = os.path.join(image_dir, f"{idx:03d}-ori.png")
        # rec_path = os.path.join(image_dir, f"{idx:03d}-rec.png")
        os.makedirs(image_dir, exist_ok=True)

        if org_tensor.dim() == 3:
            org_tensor = org_tensor.unsqueeze(0)
            rec_tensor = rec_tensor.unsqueeze(0)

        for i in range(org_tensor.shape[0]):
            ori_img = tensor2image(org_tensor[i])
            rec_img = tensor2image(rec_tensor[i])
            rec_img.save(os.path.join(image_dir, f"{idx:03d}-rec-{i:03d}.png"))
            ori_img.save(os.path.join(image_dir, f"{idx:03d}-ori-{i:03d}.png"))

    def save_checkpoint(self, loss):
        state = {
            "epoch": self.current_epoch,
            "state_dict": self.model.state_dict(),
            "main_optimizer": self.main_optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
            "main_scheduler": self.main_scheduler.state_dict(),
        }
        filename = os.path.join(self.save_dir, "runner.last.pth.tar")
        best_filename = os.path.join(self.save_dir, "runner.best.pth.tar")
        torch.save(state, filename)
        if loss < self.best_loss:
            self.best_loss = loss
            shutil.copyfile(filename, best_filename)

    def resume_checkpoint(self, load_best=False):
        last_ckpt_file = os.path.join(self.save_dir, "runner.last.pth.tar")
        best_ckpt_file = os.path.join(self.save_dir, "runner.best.pth.tar")
        ckpt_file = best_ckpt_file if load_best else last_ckpt_file
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"Checkpoint file {ckpt_file} not found")

        checkpoint = torch.load(ckpt_file, map_location="cpu")
        self.current_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["state_dict"])
        self.main_optimizer.load_state_dict(checkpoint["main_optimizer"])
        self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        self.main_scheduler.load_state_dict(checkpoint["main_scheduler"])


def main(argv):
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练脚本")
    parser.add_argument("config", type=str, help="配置文件路径")
    parser.add_argument(
        "-m", "--modify", nargs="*", help="覆盖配置参数，格式为 key=value"
    )
    args = parser.parse_args(argv)

    # 加载基础配置
    cfg = OmegaConf.load(args.config)

    # 如果有命令行覆盖参数，则更新配置
    if args.modify:
        cli_conf = OmegaConf.from_cli(args.modify)
        cfg = OmegaConf.merge(cfg, cli_conf)

    # 设置随机种子
    if cfg.engine.seed is not None:
        torch.manual_seed(cfg.engine.seed)
        random.seed(cfg.engine.seed)
        if cfg.engine.deterministic:
            torch.backends.cudnn.deterministic = True
        if cfg.engine.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main(sys.argv[1:])

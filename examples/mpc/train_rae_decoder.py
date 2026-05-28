#!/usr/bin/env python3
"""Train a RAE GeneralDecoder on frozen CoFAI DINO/MPC features.

This is intended for the shallow-RAE experiment:

1. raw:        image -> DINO blocks up to ``slot`` -> GeneralDecoder
2. compressed: image -> DINO blocks up to ``slot`` -> codec -> GeneralDecoder

For ``slot=-3`` and ``--rae-decode-blocks 0``, the decoder sees the feature after
block 8 directly and DINO blocks 9-11 are not run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROJECT_ROOT", str(_REPO_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cofai.backbone import *  # noqa: F401,F403
from cofai.heads import *  # noqa: F401,F403
from cofai.models import *  # noqa: F401,F403
from cofai.engine.registry import instantiate_class
from cofai.utils.utils import rename_key_by_rules

_EPOCH_CKPT_RE = re.compile(r"^epoch_(\d+)\.pt$")
_STEP_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")
LOG_EVERY_STEPS = 10
SAVE_EVERY_EPOCHS = 1
CKPT_EVERY_STEPS = 0
CHECKPOINT_KEEP = 3
SAMPLE_EVERY_STEPS = 1000
EVAL_EVERY_EPOCHS = 2
EVAL_BATCH_SIZE = 1
EVAL_MAX_SAMPLES = 0  # 0 means full eval dir.
EVAL_SAMPLE_LIMIT = 4
GRADIENT_LOSS_WEIGHT = 0.1
LATENT_NOISE_STD = 0.02
EMA_DECAY = 0.999
USE_TQDM = sys.stderr.isatty()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class MLPTokenAdapter(nn.Module):
    """Small channel-mixing adapter on tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor, token_res: Tuple[int, int]) -> torch.Tensor:
        return self.net(x)


class ConvResidualBlock(nn.Module):
    """Spatial residual block on the token grid using full 3x3 convolutions."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvTokenAdapter(nn.Module):
    """Residual 2D conv adapter for image reconstruction from patch tokens."""

    def __init__(self, dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.hidden_dim)
        self.blocks = nn.Sequential(*[ConvResidualBlock(self.hidden_dim) for _ in range(self.depth)])
        self.out_proj = nn.Linear(self.hidden_dim, dim)

    def forward(self, x: torch.Tensor, token_res: Tuple[int, int]) -> torch.Tensor:
        h, w = token_res
        if h * w != x.shape[1]:
            raise ValueError(f"token_res {token_res} does not match token count {x.shape[1]}")
        residual = x
        x = self.in_proj(self.norm(x))
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], h, w)
        x = self.blocks(x)
        x = x.flatten(2).transpose(1, 2)
        return residual + self.out_proj(x)


def adapter_scale_from_decoder(decoder: nn.Module) -> Tuple[int, int]:
    decoder_hidden = int(getattr(decoder.config, "decoder_hidden_size", 768))
    if decoder_hidden <= 384:
        return 384, 2
    if decoder_hidden <= 768:
        return 768, 2
    return 1024, 3


def build_adapter(dim: int, adapter_type: str, decoder: nn.Module) -> nn.Module:
    if adapter_type == "mlp":
        return MLPTokenAdapter(dim)
    if adapter_type == "conv":
        hidden_dim, depth = adapter_scale_from_decoder(decoder)
        return ConvTokenAdapter(dim, hidden_dim=hidden_dim, depth=depth)
    raise ValueError(f"Unsupported adapter_type: {adapter_type}")


class TrainableRAEDecoder(nn.Module):
    def __init__(self, decoder: nn.Module, adapter: Optional[nn.Module] = None):
        super().__init__()
        self.decoder = decoder
        self.adapter = adapter

    def forward(self, tokens: torch.Tensor, token_res: Tuple[int, int]) -> torch.Tensor:
        if self.adapter is not None:
            tokens = self.adapter(tokens, token_res)
        return self.decoder.predict(tokens, token_res=token_res, token_format="patch", clamp=False)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for key, value in current.items():
            if key not in self.shadow:
                self.shadow[key] = value.detach().clone()
                continue
            if torch.is_floating_point(value):
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {key: value.detach().clone() for key, value in state_dict.items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


class FlatImageDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"--eval-dir is not a directory: {self.root}")
        self.paths = sorted(
            path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found under --eval-dir: {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        return transforms.functional.to_tensor(img), str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="YAML configs to merge, e.g. eval_base.yaml eval_MPC2-v3-base-vbr-reg4.yaml",
    )
    parser.add_argument("--head", default="rae_dinov2_base_reg4_512px", help="cfg.heads key for GeneralDecoder")
    parser.add_argument("--train-dir", required=True, help="torchvision ImageFolder root, e.g. ImageNet/train")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and samples")
    parser.add_argument("--image-size", type=int, default=512, help="Training crop size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--loss", choices=["l1", "l2", "charbonnier"], default="l1")
    parser.add_argument(
        "--feature-source",
        choices=["raw", "compressed", "mixed"],
        default="compressed",
        help="raw uses encoder features directly; compressed uses codec output; mixed trains raw first, then compressed",
    )
    parser.add_argument("--mpc-qp", type=int, default=32, help="QP for compressed features")
    parser.add_argument(
        "--mpc-qp-random",
        action="store_true",
        help="Uniformly sample QP from [--mpc-qp-min, --mpc-qp-max] each step",
    )
    parser.add_argument("--mpc-qp-min", type=int, default=0)
    parser.add_argument("--mpc-qp-max", type=int, default=64)
    parser.add_argument(
        "--rae-decode-blocks",
        type=int,
        default=0,
        help="Override backbone. 0 means use the slot feature directly; omit by setting -1 to keep config/default",
    )
    parser.add_argument(
        "--init",
        choices=["pretrained", "random"],
        default="pretrained",
        help="Initialize decoder from head.pretrained_path or train from random init",
    )
    parser.add_argument("--use-adapter", action="store_true", help="Train a token adapter before decoder")
    parser.add_argument(
        "--adapter-type",
        choices=["conv", "mlp"],
        default="conv",
        help="Adapter architecture when --use-adapter is set",
    )
    parser.add_argument("--lpips-weight", type=float, default=0.0)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--use-gradient-loss", action="store_true", help="Add image gradient loss")
    parser.add_argument("--use-ema", action="store_true", help="Track EMA weights for eval/checkpointing")
    parser.add_argument("--use-latent-noise", action="store_true", help="Add Gaussian noise to RAE tokens during training")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="", help="Path to last.pt/epoch_XXXX.pt to resume training")
    parser.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load decoder/adapter weights only and start a fresh optimizer/schedule",
    )
    parser.add_argument("--eval-dir", default="", help="Optional image folder for periodic PSNR/SSIM/LPIPS eval")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="Debug cap; 0 means full epoch")
    return parser.parse_args()


def load_merged_config(paths: list[str]) -> Any:
    cfg = OmegaConf.load(paths[0])
    for path in paths[1:]:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    return cfg


def instantiate_model_from_cfg(cfg: Any, device: torch.device) -> nn.Module:
    model = instantiate_class(OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    load_cfg = cfg.get("load")
    if load_cfg:
        ckpt = torch.load(load_cfg.path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)
        if load_cfg.get("rules"):
            remapped = {}
            for key in sorted(state_dict.keys()):
                new_key = rename_key_by_rules(key, load_cfg.rules)
                if new_key != "":
                    remapped[new_key] = state_dict[key]
            state_dict = remapped
        model.load_state_dict(state_dict, strict=bool(load_cfg.get("strict", False)))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if hasattr(model, "update"):
        model.update()
    return model


def build_decoder(cfg: Any, args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.head not in cfg.heads:
        raise KeyError(f"Head {args.head!r} not found. Available: {sorted(cfg.heads.keys())}")
    head_cfg = OmegaConf.to_container(cfg.heads[args.head], resolve=True)
    if args.init == "random":
        head_cfg.pop("pretrained_path", None)
    decoder = instantiate_class(head_cfg, device=device)
    decoder.train()
    for param in decoder.parameters():
        param.requires_grad_(True)
    return decoder


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_epochs: int,
    epochs: int,
    steps_per_epoch: int,
    min_lr: float,
) -> SequentialLR:
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    total_steps = max(1, epochs * steps_per_epoch)
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])


def pixel_loss(x_hat: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "l1":
        return F.l1_loss(x_hat, target)
    if name == "l2":
        return F.mse_loss(x_hat, target)
    return torch.sqrt((x_hat - target) ** 2 + 1e-6).mean()


def gradient_loss(x_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_hat = x_hat[..., :, 1:] - x_hat[..., :, :-1]
    dx = target[..., :, 1:] - target[..., :, :-1]
    dy_hat = x_hat[..., 1:, :] - x_hat[..., :-1, :]
    dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_hat, dx) + F.l1_loss(dy_hat, dy)


def build_lpips(net: str, device: torch.device) -> nn.Module:
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("Install lpips or set --lpips-weight 0") from exc
    model = lpips.LPIPS(net=net, verbose=False).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_optional_lpips(net: str, device: torch.device) -> Optional[nn.Module]:
    try:
        return build_lpips(net, device)
    except ImportError:
        print("LPIPS package not found; periodic eval will report lpips=None.", flush=True)
        return None


def set_rae_decode_blocks(model: nn.Module, value: int) -> None:
    if value < 0:
        return
    dino = getattr(model, "dino", None)
    if dino is None:
        raise AttributeError("Model has no .dino backbone; cannot set rae_decode_blocks")
    dino.rae_decode_blocks = int(value)


@torch.no_grad()
def extract_rae_tokens(
    model: nn.Module,
    x: torch.Tensor,
    *,
    source: str,
    qp: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    dino = model.dino
    h = dino.encode(x)
    token_res = (x.shape[2] // model.patch_size, x.shape[3] // model.patch_size)
    if source == "raw":
        h_for_rae = h
    else:
        if not hasattr(model, "dino_codec"):
            raise AttributeError("Compressed feature source requires model.dino_codec")
        h_for_rae = model.dino_codec(h, token_res, qp=int(qp))["h_hat"]
    tokens = dino.decode_rae(h_for_rae, token_res)
    return tokens, token_res


def source_for_epoch(args: argparse.Namespace, epoch: int) -> str:
    if args.feature_source == "mixed":
        return "raw" if epoch < args.epochs // 2 else "compressed"
    return args.feature_source


def sample_qp(args: argparse.Namespace, device: torch.device) -> int:
    if args.mpc_qp_random:
        return int(torch.randint(args.mpc_qp_min, args.mpc_qp_max + 1, (), device=device).item())
    return int(args.mpc_qp)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (h, w)


def crop_to_size(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    h, w = size
    return x[..., :h, :w]


def psnr_value(x_hat: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(x_hat, target).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def _ssim_window(channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=dtype) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).view(1, 1, 11, 11)
    return window.expand(channels, 1, 11, 11).contiguous()


def ssim_value(x_hat: torch.Tensor, target: torch.Tensor) -> float:
    x_hat = x_hat.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    channels = x_hat.shape[1]
    window = _ssim_window(channels, x_hat.device, x_hat.dtype)
    mu_x = F.conv2d(x_hat, window, padding=5, groups=channels)
    mu_y = F.conv2d(target, window, padding=5, groups=channels)
    sigma_x = F.conv2d(x_hat * x_hat, window, padding=5, groups=channels) - mu_x * mu_x
    sigma_y = F.conv2d(target * target, window, padding=5, groups=channels) - mu_y * mu_y
    sigma_xy = F.conv2d(x_hat * target, window, padding=5, groups=channels) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return float(ssim_map.mean().item())


@torch.no_grad()
def evaluate_reconstruction(
    *,
    model: nn.Module,
    trainable: TrainableRAEDecoder,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    lpips_net: Optional[nn.Module],
    epoch: int,
    out_dir: Path,
) -> Dict[str, Any]:
    trainable.eval()
    psnrs = []
    ssims = []
    lpips_values = []
    eval_source = "compressed" if hasattr(model, "dino_codec") and args.feature_source != "raw" else "raw"
    eval_qp = int(args.mpc_qp)
    sample_dir = out_dir / "eval_samples" / f"epoch_{epoch:04d}"

    for idx, (x, _path) in enumerate(loader):
        if EVAL_MAX_SAMPLES > 0 and idx >= EVAL_MAX_SAMPLES:
            break
        x = x.to(device, non_blocking=True)
        x_pad, orig_size = pad_to_multiple(x, int(model.patch_size))
        tokens, token_res = extract_rae_tokens(model, x_pad, source=eval_source, qp=eval_qp)
        x_hat = trainable(tokens, token_res).clamp(0, 1)
        x_hat = crop_to_size(x_hat, orig_size)
        target = crop_to_size(x, orig_size).clamp(0, 1)
        psnrs.append(psnr_value(x_hat, target))
        ssims.append(ssim_value(x_hat, target))
        if lpips_net is not None:
            lp = lpips_net(x_hat * 2.0 - 1.0, target * 2.0 - 1.0).mean()
            lpips_values.append(float(lp.item()))
        if idx < EVAL_SAMPLE_LIMIT:
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_image(
                torch.cat([target.detach().cpu(), x_hat.detach().cpu()], dim=0),
                sample_dir / f"sample_{idx:03d}.png",
                nrow=1,
            )

    trainable.train()
    result = {
        "epoch": epoch,
        "source": eval_source,
        "qp": eval_qp if eval_source == "compressed" else None,
        "psnr": sum(psnrs) / max(len(psnrs), 1),
        "ssim": sum(ssims) / max(len(ssims), 1),
        "lpips": sum(lpips_values) / len(lpips_values) if lpips_values else None,
        "num_samples": len(psnrs),
    }
    return result


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    trainable: TrainableRAEDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    meta: Dict[str, Any],
    ema: Optional[ModelEMA] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "decoder_state_dict": trainable.decoder.state_dict(),
            "adapter_state_dict": trainable.adapter.state_dict() if trainable.adapter else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "meta": meta,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    trainable: TrainableRAEDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    weights_only: bool,
    ema: Optional[ModelEMA] = None,
) -> Tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "decoder_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing decoder_state_dict: {path}")

    trainable.decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    adapter_state = ckpt.get("adapter_state_dict")
    if trainable.adapter is not None:
        if adapter_state is None:
            raise ValueError("Current run uses --use-adapter, but checkpoint has no adapter_state_dict")
        trainable.adapter.load_state_dict(adapter_state, strict=True)
    elif adapter_state is not None:
        raise ValueError("Checkpoint has adapter_state_dict; resume with --use-adapter and matching --adapter-type")
    if ema is not None and ckpt.get("ema_state_dict") is not None:
        ema.load_state_dict(ckpt["ema_state_dict"])

    if weights_only:
        print(f"loaded weights from {path}; optimizer/scheduler start fresh")
        return 0, 0

    if "optimizer" not in ckpt or "scheduler" not in ckpt:
        raise KeyError(f"Checkpoint lacks optimizer/scheduler state; use --resume-weights-only: {path}")
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler.is_enabled() and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    print(f"resumed from {path}: next_epoch={start_epoch + 1}, global_step={global_step}")
    return start_epoch, global_step


def prune_checkpoints(out_dir: Path, pattern: re.Pattern[str], keep: int) -> None:
    if keep <= 0:
        return
    numbered: list[tuple[int, Path]] = []
    for path in out_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if len(numbered) <= keep:
        return
    numbered.sort(key=lambda item: item[0])
    for _, path in numbered[:-keep]:
        path.unlink(missing_ok=True)


def append_metric_row(csv_path: Path, jsonl_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def maybe_save_loss_curve(metrics: list[Dict[str, Any]], path: Path) -> None:
    if len(metrics) < 2:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = [int(row["step"]) for row in metrics]
    losses = [float(row["loss"]) for row in metrics]
    rec_losses = [float(row["loss_rec"]) for row in metrics]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label="loss")
    plt.plot(steps, rec_losses, label="loss_rec")
    if any(row.get("loss_lpips") is not None for row in metrics):
        lpips_steps = [int(row["step"]) for row in metrics if row.get("loss_lpips") is not None]
        lpips_values = [float(row["loss_lpips"]) for row in metrics if row.get("loss_lpips") is not None]
        if lpips_steps:
            plt.plot(lpips_steps, lpips_values, label="loss_lpips")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_trainable_weights(trainable: TrainableRAEDecoder, path: Path) -> None:
    payload = {
        "decoder_state_dict": trainable.decoder.state_dict(),
        "adapter_state_dict": trainable.adapter.state_dict() if trainable.adapter else None,
    }
    torch.save(payload, path)


def save_ema_weights(trainable: TrainableRAEDecoder, ema: ModelEMA, path: Path) -> None:
    backup = {
        key: value.detach().clone()
        for key, value in trainable.state_dict().items()
    }
    try:
        ema.copy_to(trainable)
        save_trainable_weights(trainable, path)
    finally:
        trainable.load_state_dict(backup, strict=True)


def evaluate_with_optional_ema(
    *,
    ema: Optional[ModelEMA],
    trainable: TrainableRAEDecoder,
    **kwargs,
) -> Dict[str, Any]:
    if ema is None:
        return evaluate_reconstruction(trainable=trainable, **kwargs)
    backup = {
        key: value.detach().clone()
        for key, value in trainable.state_dict().items()
    }
    try:
        ema.copy_to(trainable)
        return evaluate_reconstruction(trainable=trainable, **kwargs)
    finally:
        trainable.load_state_dict(backup, strict=True)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    cfg = load_merged_config(args.config)
    model = instantiate_model_from_cfg(cfg, device)
    set_rae_decode_blocks(model, args.rae_decode_blocks)

    decoder = build_decoder(cfg, args, device)
    token_dim = int(OmegaConf.select(cfg, f"heads.{args.head}.encoder_hidden_size", default=768))
    adapter = build_adapter(token_dim, args.adapter_type, decoder).to(device) if args.use_adapter else None
    trainable = TrainableRAEDecoder(decoder, adapter).to(device).train()
    ema = ModelEMA(trainable, EMA_DECAY) if args.use_ema else None

    params = [p for p in trainable.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    train_root = Path(args.train_dir).expanduser()
    if not train_root.is_dir():
        raise FileNotFoundError(f"--train-dir is not a directory: {train_root}")
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.image_size,
                scale=(0.5, 1.0),
                ratio=(0.75, 1.333),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    dataset = ImageFolder(str(train_root), transform=transform)
    loader_kwargs: Dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    if len(loader) == 0:
        raise RuntimeError("Training loader is empty; reduce --batch-size or check --train-dir")

    steps_per_epoch = len(loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        min_lr=args.min_lr,
    )

    lpips_net = build_lpips(args.lpips_net, device) if args.lpips_weight > 0 else None
    eval_lpips_net = build_optional_lpips(args.lpips_net, device) if args.eval_dir else None
    resume_path = Path(args.resume).expanduser() if args.resume else None
    if resume_path is not None and not args.resume_weights_only:
        out_dir = resume_path.parent
    else:
        out_dir = Path(args.output_dir).expanduser() / datetime.now().strftime("%y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": args.config,
        "head": args.head,
        "train_dir": str(train_root),
        "image_size": args.image_size,
        "feature_source": args.feature_source,
        "mixed_schedule": "first_half_raw_second_half_compressed" if args.feature_source == "mixed" else None,
        "mpc_qp": args.mpc_qp,
        "mpc_qp_random": args.mpc_qp_random,
        "mpc_qp_min": args.mpc_qp_min if args.mpc_qp_random else None,
        "mpc_qp_max": args.mpc_qp_max if args.mpc_qp_random else None,
        "rae_decode_blocks": args.rae_decode_blocks,
        "init": args.init,
        "use_adapter": args.use_adapter,
        "adapter_type": args.adapter_type if args.use_adapter else None,
        "adapter_hidden_dim": getattr(adapter, "hidden_dim", None) if adapter is not None else None,
        "adapter_depth": getattr(adapter, "depth", None) if adapter is not None else None,
        "use_gradient_loss": args.use_gradient_loss,
        "gradient_loss_weight": GRADIENT_LOSS_WEIGHT if args.use_gradient_loss else None,
        "use_latent_noise": args.use_latent_noise,
        "latent_noise_std": LATENT_NOISE_STD if args.use_latent_noise else None,
        "use_ema": args.use_ema,
        "ema_decay": EMA_DECAY if args.use_ema else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "loss": args.loss,
        "log_every": LOG_EVERY_STEPS,
        "save_every": SAVE_EVERY_EPOCHS,
        "ckpt_every_steps": CKPT_EVERY_STEPS,
        "checkpoint_keep": CHECKPOINT_KEEP,
        "sample_every": SAMPLE_EVERY_STEPS,
        "eval_dir": args.eval_dir or None,
        "eval_every_epochs": EVAL_EVERY_EPOCHS,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "eval_max_samples": EVAL_MAX_SAMPLES if EVAL_MAX_SAMPLES > 0 else None,
        "resume": str(resume_path) if resume_path is not None else None,
        "resume_weights_only": bool(args.resume_weights_only),
    }
    with open(out_dir / ("train_meta_resume.json" if resume_path is not None else "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    metrics_csv = out_dir / "metrics.csv"
    metrics_jsonl = out_dir / "metrics.jsonl"
    eval_metrics_csv = out_dir / "eval_metrics.csv"
    eval_metrics_jsonl = out_dir / "eval_metrics.jsonl"
    loss_curve_path = out_dir / "loss_curve.png"
    metric_rows: list[Dict[str, Any]] = []
    eval_loader = None
    if args.eval_dir:
        eval_loader = DataLoader(
            FlatImageDataset(args.eval_dir),
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    print(f"Training RAE decoder -> {out_dir}")
    print(
        f"feature_source={args.feature_source}, rae_decode_blocks={model.dino.rae_decode_blocks}, "
        f"init={args.init}, trainable_params={sum(p.numel() for p in params):,}",
        flush=True,
    )

    start_epoch = 0
    global_step = 0
    if resume_path is not None:
        start_epoch, global_step = load_checkpoint(
            resume_path,
            trainable=trainable,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            weights_only=bool(args.resume_weights_only),
            ema=ema,
        )
        if start_epoch >= args.epochs and not args.resume_weights_only:
            print(f"Checkpoint already reached epoch {start_epoch}; target --epochs is {args.epochs}. Nothing to do.")
            return

    for epoch in range(start_epoch, args.epochs):
        trainable.train()
        epoch_source = source_for_epoch(args, epoch)
        iterator = tqdm(
            loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            total=steps_per_epoch,
            dynamic_ncols=True,
        ) if USE_TQDM else loader
        for step, (x, _label) in enumerate(iterator):
            if step >= steps_per_epoch:
                break
            global_step += 1
            x = x.to(device, non_blocking=True)
            source = epoch_source
            qp = sample_qp(args, device)
            with torch.no_grad():
                tokens, token_res = extract_rae_tokens(model, x, source=source, qp=qp)
                if args.use_latent_noise:
                    tokens = tokens + LATENT_NOISE_STD * torch.randn_like(tokens)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                x_hat = trainable(tokens, token_res)
                loss_rec = pixel_loss(x_hat, x, args.loss)
                loss = loss_rec
                if args.use_gradient_loss:
                    loss_grad = gradient_loss(x_hat, x)
                    loss = loss + GRADIENT_LOSS_WEIGHT * loss_grad
                else:
                    loss_grad = None
            if lpips_net is not None:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    loss_lpips = lpips_net(
                        x_hat.float().clamp(0, 1) * 2.0 - 1.0,
                        x.float().clamp(0, 1) * 2.0 - 1.0,
                    ).mean()
                loss = loss + args.lpips_weight * loss_lpips
            else:
                loss_lpips = None

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(trainable)

            postfix = {
                "loss": f"{loss.item():.4f}",
                "rec": f"{loss_rec.item():.4f}",
                "src": source,
                "qp": qp,
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
            if loss_lpips is not None:
                postfix["lpips"] = f"{loss_lpips.item():.4f}"
            if loss_grad is not None:
                postfix["grad"] = f"{loss_grad.item():.4f}"
            if USE_TQDM:
                iterator.set_postfix(postfix)

            if LOG_EVERY_STEPS > 0 and (global_step == 1 or global_step % LOG_EVERY_STEPS == 0):
                row = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "source": source,
                    "qp": qp,
                    "loss": float(loss.item()),
                    "loss_rec": float(loss_rec.item()),
                    "loss_grad": float(loss_grad.item()) if loss_grad is not None else None,
                    "loss_lpips": float(loss_lpips.item()) if loss_lpips is not None else None,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
                metric_rows.append(row)
                append_metric_row(metrics_csv, metrics_jsonl, row)
                maybe_save_loss_curve(metric_rows, loss_curve_path)
                if not USE_TQDM:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} "
                        f"step={global_step} "
                        f"source={source} qp={qp} "
                        f"loss={loss.item():.6f} "
                        f"loss_rec={loss_rec.item():.6f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )

            if SAMPLE_EVERY_STEPS > 0 and global_step % SAMPLE_EVERY_STEPS == 0:
                sample = torch.cat(
                    [x[:4].detach().cpu().clamp(0, 1), x_hat[:4].detach().cpu().clamp(0, 1)],
                    dim=0,
                )
                sample_path = out_dir / "samples" / f"step_{global_step:07d}.png"
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(sample, sample_path, nrow=4)

            if CKPT_EVERY_STEPS > 0 and global_step % CKPT_EVERY_STEPS == 0:
                step_ckpt_path = out_dir / f"ckpt_step_{global_step:07d}.pt"
                save_checkpoint(
                    step_ckpt_path,
                    epoch=epoch + 1,
                    global_step=global_step,
                    trainable=trainable,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    meta=meta,
                    ema=ema,
                )
                torch.save(trainable.decoder.state_dict(), out_dir / "model_latest.pt")
                save_trainable_weights(trainable, out_dir / "trainable_latest.pt")
                if ema is not None:
                    save_ema_weights(trainable, ema, out_dir / "trainable_ema_latest.pt")
                print(f"saved {step_ckpt_path}")
                prune_checkpoints(out_dir, _STEP_CKPT_RE, CHECKPOINT_KEEP)

        if (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
            ckpt_path = out_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                ckpt_path,
                epoch=epoch + 1,
                global_step=global_step,
                trainable=trainable,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                meta=meta,
                ema=ema,
            )
            torch.save(trainable.decoder.state_dict(), out_dir / "model.pt")
            save_trainable_weights(trainable, out_dir / "trainable.pt")
            if ema is not None:
                save_ema_weights(trainable, ema, out_dir / "trainable_ema.pt")
            print(f"saved {ckpt_path}")
            prune_checkpoints(out_dir, _EPOCH_CKPT_RE, CHECKPOINT_KEEP)

        if eval_loader is not None and (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
            eval_result = evaluate_with_optional_ema(
                ema=ema,
                model=model,
                trainable=trainable,
                loader=eval_loader,
                device=device,
                args=args,
                lpips_net=eval_lpips_net,
                epoch=epoch + 1,
                out_dir=out_dir,
            )
            append_metric_row(eval_metrics_csv, eval_metrics_jsonl, eval_result)
            lpips_text = "None" if eval_result["lpips"] is None else f"{eval_result['lpips']:.6f}"
            print(
                f"[eval epoch={epoch + 1}] "
                f"source={eval_result['source']} qp={eval_result['qp']} "
                f"psnr={eval_result['psnr']:.4f} "
                f"ssim={eval_result['ssim']:.6f} "
                f"lpips={lpips_text} "
                f"n={eval_result['num_samples']}",
                flush=True,
            )

    save_checkpoint(
        out_dir / "last.pt",
        epoch=args.epochs,
        global_step=global_step,
        trainable=trainable,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        meta=meta,
        ema=ema,
    )
    torch.save(trainable.decoder.state_dict(), out_dir / "model.pt")
    save_trainable_weights(trainable, out_dir / "trainable.pt")
    if ema is not None:
        save_ema_weights(trainable, ema, out_dir / "trainable_ema.pt")
    print(f"saved final decoder weights: {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()

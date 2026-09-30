#!/usr/bin/env python3
"""Train VQ-UFC directly on pre-extracted DINOv3 ``.npy`` features.

The validation split is taken only from the supplied training-feature folder.
Validation reports feature reconstruction/rate objectives and never invokes a
DINOv3 backbone or a downstream task head.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from cofai.entropy_models.vq_ufc_model import VQUFCModel
from cofai.utils.utils_vqufc import load_rms, split_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-features", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-name", choices=("ade_1w", "nyu_1w"), required=True)
    p.add_argument("--num-embeddings", type=int, default=2048)
    p.add_argument("--embedding-dim", type=int, default=8)
    p.add_argument("--num-chunks", type=int, default=32)
    p.add_argument("--lmbda", type=float, default=5.0)
    p.add_argument("--commit-weight", type=float, default=0.25)
    p.add_argument("--soft-rate-weight", type=float, default=1.0)
    p.add_argument("--usage-weight", type=float, default=0.0)
    p.add_argument("--soft-temperature", type=float, default=1.0)
    p.add_argument("--soft-temperature-min", type=float, default=0.1)
    p.add_argument("--soft-temperature-decay", type=float, default=0.95)
    p.add_argument(
        "--use-soft-assignment",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--use-transform",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--transform-input-tokens", type=int, default=256)
    p.add_argument("--transform-tokens", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8, help="Number of feature files accumulated per optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rms-path", required=True)
    p.add_argument(
        "--grad-max-norm",
        type=float,
        default=1.0,
        help="Optimizer gradient max norm; 0 disables gradient clipping",
    )
    p.add_argument("--max-files", type=int, default=0, help="Limit files before splitting; 0 uses all files")
    p.add_argument(
        "--max-train-files",
        type=int,
        default=0,
        help="Limit training files after the full split/RMS hash check; 0 uses all",
    )
    p.add_argument(
        "--max-val-files", type=int, default=0, help="Limit validation files after splitting; 0 uses the full split"
    )
    p.add_argument("--init-checkpoint", default="")
    p.add_argument(
        "--freeze-codebook",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the VQ codebook after checkpoint init (transform fine-tuning)",
    )
    p.add_argument(
        "--freeze-entropy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the entropy prior/logits after checkpoint init (transform fine-tuning)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write TensorBoard events directly into --output-dir",
    )
    p.add_argument(
        "--tensorboard-log-interval",
        type=int,
        default=10,
        help="Log train batch metrics every N optimizer steps",
    )
    args = p.parse_args()
    args.vector_mode = "legacy_sequence"
    args.clip_value = 0.0
    args.num_prefix_tokens = 5
    args.normalization_mode = "prefix_position_patch_channel_rms"
    args.rms_eps = 1e-6
    return args


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_and_split(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    _root, train_files, val_files = split_features(
        args.train_features,
        args.seed,
        args.val_ratio,
        args.max_files,
    )
    if args.max_val_files > 0:
        val_files = val_files[: args.max_val_files]
    return train_files, val_files


def load_feature(
    path: Path,
    device: torch.device,
    args: argparse.Namespace,
    patch_channel_rms: torch.Tensor,
    prefix_position_channel_rms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"Expected [tokens, channels] in {path}, got {array.shape}")
    raw_feature = torch.as_tensor(np.asarray(array, dtype=np.float32), device=device).unsqueeze(0)
    if raw_feature.shape[1] <= args.num_prefix_tokens:
        raise ValueError(f"Not enough tokens for all-token RMS in {path}")
    channels = raw_feature.shape[-1]
    if prefix_position_channel_rms.shape != (args.num_prefix_tokens, channels):
        raise ValueError(f"Prefix RMS shape does not match {tuple(raw_feature.shape)}")
    if patch_channel_rms.numel() != channels:
        raise ValueError(f"Patch RMS shape does not match {tuple(raw_feature.shape)}")
    prefix = raw_feature[:, : args.num_prefix_tokens] / prefix_position_channel_rms.view(
        1, args.num_prefix_tokens, channels
    )
    patch = raw_feature[:, args.num_prefix_tokens :] / patch_channel_rms.view(1, 1, channels)
    normalized = torch.cat((prefix, patch), dim=1)
    if not torch.isfinite(normalized).all():
        raise FloatingPointError(f"Non-finite normalized feature in {path}")
    return raw_feature, normalized


def model_config(args: argparse.Namespace) -> dict:
    return {
        "num_embeddings": args.num_embeddings,
        "embedding_dim": args.embedding_dim,
        "num_chunks": args.num_chunks,
        "lmbda": args.lmbda,
        "commit_weight": args.commit_weight,
        "soft_rate_weight": args.soft_rate_weight,
        "usage_weight": args.usage_weight,
        "use_soft_assignment": args.use_soft_assignment,
        "soft_temperature": args.soft_temperature,
        "soft_temperature_min": args.soft_temperature_min,
        "soft_temperature_decay": args.soft_temperature_decay,
        "vector_mode": args.vector_mode,
        "use_transform": args.use_transform,
        "transform_input_tokens": args.transform_input_tokens,
        "transform_tokens": args.transform_tokens,
        "clip_value": args.clip_value,
        "num_prefix_tokens": args.num_prefix_tokens,
        "normalization_mode": args.normalization_mode,
        "rms_eps": args.rms_eps,
    }


def build_model(args: argparse.Namespace, device: torch.device) -> VQUFCModel:
    model = VQUFCModel(**model_config(args)).to(device)
    if args.init_checkpoint:
        model.load_compatible_checkpoint(args.init_checkpoint, map_location="cpu")
    if args.freeze_codebook or args.freeze_entropy:
        model.set_codec_trainability(
            freeze_codebook=args.freeze_codebook,
            freeze_entropy=args.freeze_entropy,
        )
    return model


METRICS = (
    "rd_loss",
    "recon_mse",
    "vq_mse",
    "commitment",
    "soft_rate",
    "hard_rate",
    "usage_loss",
    "raw_recon_mse",
    "prefix_raw_recon_mse",
    "patch_raw_recon_mse",
)


def output_metrics(
    model: VQUFCModel,
    output: tuple,
    raw_feature: torch.Tensor,
    patch_channel_rms: torch.Tensor,
    prefix_position_channel_rms: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    reconstructed = output[0]
    channels = reconstructed.shape[-1]
    prefix_slice = slice(0, args.num_prefix_tokens)
    patch_slice = slice(args.num_prefix_tokens, None)
    prefix_hat = reconstructed[:, prefix_slice] * prefix_position_channel_rms.view(1, args.num_prefix_tokens, channels)
    patch_hat = reconstructed[:, patch_slice] * patch_channel_rms.view(1, 1, channels)
    raw_reconstructed = torch.cat((prefix_hat, patch_hat), dim=1)
    return {
        "recon_mse": output[1],
        "vq_mse": output[2],
        "commitment": output[3],
        "rd_loss": output[4],
        "soft_rate": output[5],
        "hard_rate": output[6],
        "usage_loss": model.last_usage_loss,
        "raw_recon_mse": F.mse_loss(raw_reconstructed, raw_feature),
        "prefix_raw_recon_mse": F.mse_loss(raw_reconstructed[:, prefix_slice], raw_feature[:, prefix_slice]),
        "patch_raw_recon_mse": F.mse_loss(raw_reconstructed[:, patch_slice], raw_feature[:, patch_slice]),
    }


def mean_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {key: total[key] / max(count, 1) for key in METRICS}


def log_scalars(
    writer: SummaryWriter,
    prefix: str,
    metrics: dict[str, float],
    step: int,
) -> None:
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def configure_tensorboard(writer: SummaryWriter, config: dict) -> None:
    writer.add_text("run/config", f"```json\n{json.dumps(config, indent=2)}\n```", 0)
    writer.add_custom_scalars(
        {
            "Train vs validation": {
                metric: [
                    "Multiline",
                    [f"epoch/train/{metric}", f"epoch/val/{metric}"],
                ]
                for metric in METRICS
            }
        }
    )


def train_epoch(
    model: VQUFCModel,
    files: list[Path],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    writer: SummaryWriter | None,
    patch_channel_rms: torch.Tensor | None,
    prefix_position_channel_rms: torch.Tensor | None,
) -> dict[str, float]:
    model.train()
    temperature = model.update_soft_temperature(epoch)
    order = list(files)
    random.Random(args.seed + epoch).shuffle(order)
    total = {key: 0.0 for key in METRICS}
    optimizer.zero_grad(set_to_none=True)
    batches = math.ceil(len(order) / args.batch_size)
    progress = tqdm(range(batches), desc=f"train {epoch}/{args.epochs}")
    seen = 0
    gradient_norm_total = 0.0
    for batch_index in progress:
        batch = order[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
        batch_total = {key: 0.0 for key in METRICS}
        for path in batch:
            raw_feature, feature = load_feature(path, device, args, patch_channel_rms, prefix_position_channel_rms)
            values = output_metrics(
                model,
                model(feature),
                raw_feature,
                patch_channel_rms,
                prefix_position_channel_rms,
                args,
            )
            (values["rd_loss"] / len(batch)).backward()
            for key in METRICS:
                value = float(values[key].detach().item())
                total[key] += value
                batch_total[key] += value
            seen += 1
        if args.grad_max_norm > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_max_norm)
            gradient_norm_value = float(gradient_norm.detach().item())
        else:
            gradient_norm_value = 0.0
        gradient_norm_total += gradient_norm_value
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step = (epoch - 1) * batches + batch_index + 1
        if writer is not None and (global_step % args.tensorboard_log_interval == 0 or batch_index + 1 == batches):
            log_scalars(writer, "train_step", mean_metrics(batch_total, len(batch)), global_step)
            writer.add_scalar("train_step/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train_step/temperature", temperature, global_step)
            writer.add_scalar("train_step/gradient_norm", gradient_norm_value, global_step)
        current = mean_metrics(total, seen)
        progress.set_postfix(rd=f"{current['rd_loss']:.5f}", temp=f"{temperature:.4f}")
    result = mean_metrics(total, seen)
    result["temperature"] = float(temperature)
    result["gradient_norm"] = gradient_norm_total / max(batches, 1)
    return result


@torch.inference_mode()
def validate(
    model: VQUFCModel,
    files: list[Path],
    device: torch.device,
    args: argparse.Namespace,
    patch_channel_rms: torch.Tensor | None,
    prefix_position_channel_rms: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    total = {key: 0.0 for key in METRICS}
    for path in tqdm(files, desc="validate"):
        raw_feature, feature = load_feature(path, device, args, patch_channel_rms, prefix_position_channel_rms)
        values = output_metrics(
            model,
            model(feature),
            raw_feature,
            patch_channel_rms,
            prefix_position_channel_rms,
            args,
        )
        for key in METRICS:
            total[key] += float(values[key].item())
    return mean_metrics(total, len(files))


def save_checkpoint(
    path: Path,
    model: VQUFCModel,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    normalization: dict,
) -> None:
    torch.save(
        {
            "format_version": 2,
            "model_name": "VQ-UFC",
            "feature_backbone": "DINOv3-ViT-L/16-slot24",
            "dataset_name": args.dataset_name,
            "model_config": model_config(args),
            "train_config": vars(args),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "normalization": normalization,
            "vqvae_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.tensorboard_log_interval < 1:
        raise ValueError("--tensorboard-log-interval must be at least 1")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_files = list_and_split(args)
    root = Path(args.train_features).expanduser().resolve()
    prefix_array, patch_array, rms_metadata = load_rms(args.rms_path, train_files, root, args.num_prefix_tokens)
    prefix_position_channel_rms = torch.from_numpy(prefix_array).to(device=device)
    patch_channel_rms = torch.from_numpy(patch_array).to(device=device)
    if args.max_train_files > 0:
        train_files = train_files[: args.max_train_files]
    normalization = {
        "mode": args.normalization_mode,
        "num_prefix_tokens": args.num_prefix_tokens,
        "clip_value": args.clip_value,
        "rms_eps": args.rms_eps,
        "patch_channel_rms": patch_channel_rms.detach().cpu(),
        "prefix_position_channel_rms": prefix_position_channel_rms.detach().cpu(),
        "metadata": rms_metadata,
    }
    split = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train": [str(path.relative_to(root)) for path in train_files],
        "validation": [str(path.relative_to(root)) for path in val_files],
    }
    (output_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    run_config = {
        "model_config": model_config(args),
        "train_config": vars(args),
        "normalization": {
            key: value
            for key, value in normalization.items()
            if key not in {"patch_channel_rms", "prefix_position_channel_rms"}
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"dataset={args.dataset_name}")
    print(f"train_files={len(train_files)} validation_files={len(val_files)}")
    print(
        f"coded_tokens=prefix+patch num_prefix_tokens={args.num_prefix_tokens} "
        f"normalization={args.normalization_mode}"
    )
    print(f"feature_clip={'off' if args.clip_value == 0 else args.clip_value}")
    print("validation=coded feature RD only; no downstream task evaluation")

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    tb_writer = SummaryWriter(log_dir=str(output_dir)) if args.tensorboard else None
    if tb_writer is not None:
        configure_tensorboard(tb_writer, run_config)
        tb_writer.add_scalar("run/train_files", len(train_files), 0)
        tb_writer.add_scalar("run/validation_files", len(val_files), 0)
        tb_writer.add_scalar(
            "run/trainable_parameters",
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            0,
        )
        print(f"tensorboard_logdir={output_dir}")
    history_path = output_dir / "history.csv"
    best_val = float("inf")
    fieldnames = ["epoch", "seconds"] + [
        f"{split_name}_{metric}" for split_name in ("train", "val") for metric in METRICS
    ] + ["train_temperature", "train_gradient_norm"]
    try:
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
            csv_writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                start = time.time()
                train_metrics = train_epoch(
                    model,
                    train_files,
                    optimizer,
                    device,
                    args,
                    epoch,
                    tb_writer,
                    patch_channel_rms,
                    prefix_position_channel_rms,
                )
                val_metrics = validate(
                    model,
                    val_files,
                    device,
                    args,
                    patch_channel_rms,
                    prefix_position_channel_rms,
                )
                seconds = time.time() - start
                row = {"epoch": epoch, "seconds": seconds}
                row.update({f"train_{key}": value for key, value in train_metrics.items()})
                row.update({f"val_{key}": value for key, value in val_metrics.items()})
                csv_writer.writerow(row)
                handle.flush()
                print(json.dumps(row, sort_keys=True))

                if tb_writer is not None:
                    log_scalars(tb_writer, "epoch/train", train_metrics, epoch)
                    log_scalars(tb_writer, "epoch/val", val_metrics, epoch)
                    tb_writer.add_scalar("epoch/time_seconds", seconds, epoch)

                save_checkpoint(
                    output_dir / "last.pth.tar",
                    model,
                    optimizer,
                    args,
                    epoch,
                    train_metrics,
                    val_metrics,
                    normalization,
                )
                if val_metrics["rd_loss"] < best_val:
                    best_val = val_metrics["rd_loss"]
                    save_checkpoint(
                        output_dir / "best.pth.tar",
                        model,
                        optimizer,
                        args,
                        epoch,
                        train_metrics,
                        val_metrics,
                        normalization,
                    )
                    print(f"saved best.pth.tar: val_rd_loss={best_val:.8f}")
                if tb_writer is not None:
                    tb_writer.add_scalar("epoch/val/best_rd_loss", best_val, epoch)
                    tb_writer.flush()
    finally:
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()

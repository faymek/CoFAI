#!/usr/bin/env python3
"""Train PQFCv2 stage 1: orthogonal transform and one PQ codebook.

Features stay at their native token length. A step accumulates gradients over
``batch_size`` images, each forwarded alone, following VQ-UFC stage 1.
Normalization is the fixed VQ-UFC RMS (prefix-position plus shared patch
channels). PQFC per-image mean/std is not used.

``distortion: vector`` trains the sum of squares of the RMS-normalized
quantizer vectors. ``distortion: post_ln`` trains the sum of squares of the
frozen final LayerNorm, the earliest stage-1 objective: whole-image bits
plus ``lmbda`` times that sum. ``rate_cap_bits_per_token`` adds a hinge on bits
above that cap so a larger codebook cannot spend past the rate budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from cofai.backbone import Dinov3TimmBackbone
from cofai.ops.orfc import batched_assign, batched_kmeans
from cofai.utils.utils_vqufc import load_rms
from examples.orfc_2446.offline.artifacts import save_codec_npz
from examples.orfc_2446.offline.soft_pq import (
    FeatureCodec,
    OrthogonalTransform,
    SoftPQ,
    compute_perplexity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_transform", action="store_true", help="Drop the orthogonal transform; train PQ codebooks only")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(os.environ["PROJECT_ROOT"]).expanduser().resolve()
    config["_project_root"] = str(root)
    for key, value in config["paths"].items():
        candidate = Path(value)
        config["paths"][key] = str(candidate if candidate.is_absolute() else (root / candidate).resolve())
    return config


def apply_rms(
    features: torch.Tensor,
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
) -> torch.Tensor:
    channels = features.shape[-1]
    prefix = features[:, :n_prefix] / prefix_rms.view(1, n_prefix, channels)
    patch = features[:, n_prefix:] / patch_rms.view(1, 1, channels)
    return torch.cat((prefix, patch), dim=1)


def invert_rms(
    normalized: torch.Tensor,
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
    token_offset: int,
) -> torch.Tensor:
    length = normalized.shape[1]
    positions = torch.arange(token_offset, token_offset + length, device=normalized.device)
    scale = patch_rms.view(1, 1, -1).expand(1, length, -1).clone()
    prefix_positions = positions < n_prefix
    if bool(prefix_positions.any()):
        scale[0, prefix_positions] = prefix_rms[positions[prefix_positions]]
    return normalized * scale


def load_feature(path: Path, device: torch.device) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 1024 or array.shape[0] <= 5:
        raise ValueError(f"Expected [tokens>5, 1024] in {path}, got {array.shape}")
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).unsqueeze(0).to(device)


def reservoir_sample(
    files: list[Path],
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
    capacity: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    buffer = np.empty((capacity, 1024), dtype=np.float32)
    filled = 0
    seen = 0
    for path in tqdm(files, desc="opq-sample"):
        raw = load_feature(path, device)
        vectors = apply_rms(raw, prefix_rms, patch_rms, n_prefix).reshape(-1, 1024).cpu().numpy()
        del raw
        count = int(vectors.shape[0])
        if filled < capacity:
            take = min(capacity - filled, count)
            buffer[filled : filled + take] = vectors[:take]
            filled += take
            vectors = vectors[take:]
            seen += take
            count = int(vectors.shape[0])
        if count:
            ranks = np.arange(seen + 1, seen + count + 1)
            chosen = vectors[rng.rand(count) < (capacity / ranks)]
            if len(chosen):
                buffer[rng.randint(0, capacity, size=len(chosen))] = chosen
            seen += count
    if filled < capacity:
        return buffer[:filled]
    return buffer


@torch.no_grad()
def opq_usage_counts(
    files: list[Path],
    rotation: np.ndarray,
    codebooks: list[np.ndarray],
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
    device: torch.device,
) -> np.ndarray:
    groups = len(codebooks)
    counts = np.zeros((groups, codebooks[0].shape[0]), dtype=np.float64)
    rotation_t = torch.from_numpy(rotation).float().to(device)
    codebook_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    for path in tqdm(files, desc="opq-usage"):
        raw = load_feature(path, device)
        flat = apply_rms(raw, prefix_rms, patch_rms, n_prefix).reshape(-1, 1024)
        rotated = flat @ rotation_t
        subspace = rotated.reshape(-1, groups, codebooks[0].shape[1]).permute(1, 0, 2).contiguous()
        labels = batched_assign(subspace, codebook_t, device=device)[1].cpu().numpy()
        for group in range(groups):
            np.add.at(counts[group], labels[group], 1)
        del raw, flat, rotated, subspace
    return counts


def finalize_opq(rotation: np.ndarray, codebooks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Match the M2488 sign flip so det(R) is positive before Cayley init."""
    rotation = np.asarray(rotation, dtype=np.float32).copy()
    codebooks = [np.asarray(item, dtype=np.float32).copy() for item in codebooks]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
        codebooks[-1][:, -1] *= -1
    return rotation, codebooks


def save_opq_init(
    path: Path,
    rotation: np.ndarray,
    codebooks: list[np.ndarray],
    usage: np.ndarray,
    history: list[tuple[float, float]],
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        R=np.asarray(rotation, dtype=np.float32),
        codebooks=np.stack(codebooks).astype(np.float32),
        usage=np.asarray(usage, dtype=np.float64),
        history_mse=np.asarray([item[0] for item in history], dtype=np.float64),
        history_delta=np.asarray([item[1] for item in history], dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_opq_init(
    path: Path,
    *,
    split_sha256: str,
    groups: int,
    codebook_size: int,
    embedding_dim: int,
    allow_codebook_mismatch: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        rotation = np.asarray(data["R"], dtype=np.float32)
        codebooks = np.asarray(data["codebooks"], dtype=np.float32)
        usage = np.asarray(data["usage"], dtype=np.float64)
    if metadata.get("split_sha256") != split_sha256:
        raise ValueError(f"OPQ init {path} was built for a different training split")
    if rotation.shape != (groups * embedding_dim, groups * embedding_dim):
        raise ValueError(f"OPQ rotation shape {rotation.shape} does not match D={groups * embedding_dim}")
    if codebooks.shape != (groups, codebook_size, embedding_dim):
        if allow_codebook_mismatch:
            print(f"  note: OPQ codebook K={codebooks.shape[1]} != config K={codebook_size}, using rotation only")
        else:
            raise ValueError(f"OPQ codebooks shape {codebooks.shape} does not match {(groups, codebook_size, embedding_dim)}")
    if usage.shape != (groups, codebook_size):
        if not allow_codebook_mismatch:
            raise ValueError(f"OPQ usage shape {usage.shape} does not match {(groups, codebook_size)}")
    if not np.isfinite(rotation).all() or not np.isfinite(codebooks).all():
        raise ValueError(f"OPQ init {path} contains non-finite values")
    return rotation, [codebooks[group] for group in range(groups)], usage


def temperature(epoch: int, epochs: int, start: float, end: float, schedule: str) -> float:
    if epochs <= 1:
        return start
    progress = epoch / (epochs - 1)
    if schedule == "linear":
        return start + (end - start) * progress
    return start * (end / start) ** progress


def iter_token_chunks(length: int, chunk_tokens: int):
    for start in range(0, length, chunk_tokens):
        yield start, min(start + chunk_tokens, length)


def accumulate_image(
    raw: torch.Tensor,
    codec: FeatureCodec,
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
    chunk_tokens: int,
    lmbda: float,
    images_per_step: int,
    commit_weight: float = 0.0,
    tail=None,
    distortion_mode: str = "vector",
    rate_cap_bits_per_token: float | None = None,
    rate_cap_weight: float = 0.0,
) -> tuple[float, float, float, torch.Tensor]:
    """Whole-image bits plus ``lmbda`` times a sum of squares.

    ``vector`` sums the RMS-normalized quantizer error. ``post_ln`` sums the
    frozen final-LayerNorm error, with the teacher side stopped. Neither sum
    is divided by the token count. Commitment, when enabled, is the sum of
    squares on the rotated vectors with the codebook side stopped.
    """
    if distortion_mode not in ("vector", "post_ln"):
        raise ValueError(f"distortion must be 'vector' or 'post_ln', got {distortion_mode}")
    if distortion_mode == "post_ln" and tail is None:
        raise ValueError("post_ln distortion needs the frozen final LayerNorm")
    normalized = apply_rms(raw, prefix_rms, patch_rms, n_prefix)
    usage_total = None
    distortion_value = 0.0
    rate_value = 0.0
    commit_value = 0.0
    for start, end in iter_token_chunks(normalized.shape[1], chunk_tokens):
        restored, usage = codec(normalized[:, start:end])
        length = end - start
        if distortion_mode == "post_ln":
            decoded = invert_rms(restored, prefix_rms, patch_rms, n_prefix, start)
            with torch.no_grad():
                teacher = tail.forward_nograd(raw[:, start:end])
            student = tail(decoded)
            distortion = (teacher - student).square().sum()
            del decoded, teacher, student
        else:
            distortion = (normalized[:, start:end] - restored).square().sum()
        rate_bits = codec._last_rate * length
        loss = rate_bits + lmbda * distortion
        if rate_cap_bits_per_token is not None and rate_cap_weight > 0:
            excess = (rate_bits - float(rate_cap_bits_per_token) * length).clamp_min(0)
            loss = loss + float(rate_cap_weight) * excess
        if commit_weight > 0:
            commit = (codec._last_z - codec._last_z_hat.detach()).square().sum()
            loss = loss + commit_weight * commit
            commit_value += float(commit.detach())
        (loss / images_per_step).backward()
        distortion_value += float(distortion.detach())
        rate_value += float(rate_bits.detach())
        usage_total = usage.detach() if usage_total is None else usage_total + usage.detach()
        del restored, rate_bits, distortion
    return distortion_value, rate_value, commit_value, usage_total


class _NullTail:
    """No-op tail used when ``--no_transform`` skips backbone loading.

    ``evaluate`` calls ``tail.forward_nograd`` on the raw and decoded tensors
    to measure post-LayerNorm distortion. Without a transform that path is
    irrelevant, so this stub returns zeros and makes those metrics zero.
    """

    def forward_nograd(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


@torch.inference_mode()
def evaluate(
    files: list[Path],
    codec: FeatureCodec,
    tail,
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
    chunk_tokens: int,
    device: torch.device,
) -> dict[str, float]:
    codec.eval()
    sum_all = 0.0
    sum_patch = 0.0
    sum_raw = 0.0
    sum_vector = 0.0
    count_all = 0
    count_patch = 0
    count_raw = 0
    count_vector = 0
    for path in files:
        raw = load_feature(path, device)
        normalized = apply_rms(raw, prefix_rms, patch_rms, n_prefix)
        pieces = []
        for start, end in iter_token_chunks(normalized.shape[1], chunk_tokens):
            pieces.append(codec(normalized[:, start:end])[0])
        restored = torch.cat(pieces, dim=1)
        vector_diff = normalized - restored
        sum_vector += float(vector_diff.square().sum())
        count_vector += int(vector_diff.numel())
        decoded = invert_rms(restored, prefix_rms, patch_rms, n_prefix, 0)
        teacher = tail.forward_nograd(raw)
        student = tail.forward_nograd(decoded)
        diff = teacher - student
        sum_all += float(diff.square().sum())
        count_all += int(diff.numel())
        patch = diff[:, n_prefix:]
        sum_patch += float(patch.square().sum())
        count_patch += int(patch.numel())
        raw_diff = raw - decoded
        sum_raw += float(raw_diff.square().sum())
        count_raw += int(raw_diff.numel())
    return {
        "vector_mse": sum_vector / max(count_vector, 1),
        "post_ln_all_mse": sum_all / max(count_all, 1),
        "post_ln_patch_mse": sum_patch / max(count_patch, 1),
        "raw_mse": sum_raw / max(count_raw, 1),
    }


def histogram_pmf(
    files: list[Path],
    codec: FeatureCodec,
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
    chunk_tokens: int,
    device: torch.device,
) -> np.ndarray:
    codec.eval()
    groups, codebook_size = codec.pq.G, codec.pq.K
    counts = np.zeros((groups, codebook_size), dtype=np.int64)
    for path in tqdm(files, desc="pmf"):
        raw = load_feature(path, device)
        normalized = apply_rms(raw, prefix_rms, patch_rms, n_prefix)
        for start, end in iter_token_chunks(normalized.shape[1], chunk_tokens):
            codec(normalized[:, start:end])
            labels = codec.pq._last_labels.cpu().numpy()
            for group in range(groups):
                np.add.at(counts[group], labels[group], 1)
    pmf = np.maximum(counts.astype(np.float64), 1e-7)
    pmf /= pmf.sum(axis=1, keepdims=True)
    return pmf.astype(np.float32)


def save_checkpoint(
    path: Path,
    codec: FeatureCodec,
    optimizer,
    scheduler,
    epoch: int,
    history: list[dict],
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": "PQFCv2",
            "epoch": epoch,
            "config": config,
            "history": history,
            "codec": codec.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    feature_root = Path(config["paths"]["feature_root"])
    split = json.loads(Path(config["paths"]["split_path"]).read_text(encoding="utf-8"))
    train_files = [feature_root / name for name in split["train"]]
    val_files = [feature_root / name for name in split["validation"]]
    prefix_np, patch_np, rms_meta = load_rms(
        config["paths"]["rms_path"],
        train_files,
        feature_root,
        int(config["n_prefix"]),
    )
    prefix_rms = torch.from_numpy(prefix_np).to(device)
    patch_rms = torch.from_numpy(patch_np).to(device)

    embed_dim = int(config["embed_dim"])
    embedding_dim = int(config["embedding_dim"])
    if embed_dim % embedding_dim != 0:
        raise ValueError(f"embed_dim {embed_dim} is not divisible by embedding_dim {embedding_dim}")
    groups = embed_dim // embedding_dim
    codebook_size = int(config["K"])
    layer_idx = int(str(config["layer"])[-2:])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"train={len(train_files)} validation={len(val_files)} K={codebook_size} e={embedding_dim} G={groups}")
    if args.no_transform:
        tail = None
        print("--no_transform: skipping DINOv3 backbone/tail load")
    else:
        backbone = Dinov3TimmBackbone(
            model_size="large",
            img_size=512,
            patch_size=16,
            dynamic_size=True,
            slot=int(config["slot"]),
            n_last_blocks=1,
            cast_dtype="float32",
            pretrained=False,
            ckpt_path=config["paths"]["backbone"],
            device=str(device),
        ).eval()
        tail = backbone.build_frozen_tail(layer_idx, (14, 14), device)
        if len(getattr(tail, "blocks", [])) != 0:
            raise RuntimeError("Stage 1 expects the final DINOv3 block, whose tail is LayerNorm only")
        del backbone

    rot_lr = config.get("rot_lr")
    quantizer = SoftPQ(
        groups,
        codebook_size,
        embedding_dim,
        lmbda=float(config["lmbda"]),
        prior_floor=float(config["prior_floor"]),
        shared=bool(config.get("shared_codebook", False)),
    ).to(device)
    transform = None if args.no_transform else OrthogonalTransform(embed_dim).to(device)
    codec = FeatureCodec(quantizer, transform).to(device)
    trainable = [parameter for parameter in codec.parameters() if parameter.requires_grad]
    if rot_lr is not None and transform is not None:
        rotation_ids = {id(parameter) for parameter in transform.parameters()}
        optimizer = torch.optim.Adam(
            [
                {"params": [parameter for parameter in trainable if id(parameter) not in rotation_ids], "lr": float(config["lr"])},
                {"params": [parameter for parameter in trainable if id(parameter) in rotation_ids], "lr": float(rot_lr)},
            ]
        )
        print(f"split lr codebook={float(config['lr'])} rotation={float(rot_lr)}")
    else:
        optimizer = torch.optim.Adam(trainable, lr=float(config["lr"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
        eta_min=float(config["lr"]) * 0.01,
    )

    history_path = output_dir / "history.jsonl"
    start_epoch = 0
    history: list[dict] = []
    last_path = output_dir / "last.pt"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        codec.load_state_dict(checkpoint["codec"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        print(f"resumed at epoch {start_epoch}")
    else:
        if args.no_transform:
            print("no orthogonal transform; initialising codebooks with k-means on RMS-normalised features")
            samples = reservoir_sample(
                train_files,
                prefix_rms,
                patch_rms,
                int(config["n_prefix"]),
                int(config["kmeans_max_samples"]),
                int(config["seed"]),
                device,
            )
            sub = samples.reshape(-1, groups, embedding_dim).transpose(1, 0, 2)
            centroids = batched_kmeans(
                sub, codebook_size, max_iter=100, device=device, verbose=True
            )
            codebooks_init = centroids.cpu().numpy()
            labels = batched_assign(sub, centroids, device=device)[1].cpu().numpy()
            usage = np.zeros((groups, codebook_size), dtype=np.float64)
            for g in range(groups):
                counts = np.bincount(labels[g], minlength=codebook_size).astype(np.float64)
                usage[g] = counts
            quantizer.init_codebooks([codebooks_init[g] for g in range(groups)])
            quantizer.init_prior_from_freq(usage)
            history_path.write_text("", encoding="utf-8")
            del samples, sub, centroids, labels
            torch.cuda.empty_cache()
            print(f"k-means init done; total samples used={np.sum(usage):.0f}")
        else:
            opq_path = Path(config["paths"]["opq_init"])
            if not opq_path.is_file():
                raise FileNotFoundError(
                    f"OPQ init not found: {opq_path}. Run examples/pqfcv2/init_opq.py first."
                )
            rotation, codebooks, usage = load_opq_init(
                opq_path,
                split_sha256=str(rms_meta["split_sha256"]),
                groups=groups,
                codebook_size=codebook_size,
                embedding_dim=embedding_dim,
                allow_codebook_mismatch=quantizer.shared,
            )
            transform.init_from_opq(rotation)
            if quantizer.shared:
                # Pool all groups' subvectors and run k-means for a single shared codebook
                print("shared codebook: pooling all groups for k-means init")
                samples = reservoir_sample(
                    train_files,
                    prefix_rms,
                    patch_rms,
                    int(config["n_prefix"]),
                    int(config["kmeans_max_samples"]),
                    int(config["seed"]),
                    device,
                )
                with torch.no_grad():
                    Z = torch.from_numpy(samples).to(device).float() @ transform.get_rotation()
                    sub_all = Z.reshape(-1, embedding_dim)  # (N*128, 8)
                    cb_shared = batched_kmeans(
                        sub_all.unsqueeze(0), codebook_size, max_iter=100, device=device, verbose=True
                    )[0]
                    labels_shared = batched_assign(
                        sub_all.unsqueeze(0), cb_shared.unsqueeze(0), device=device
                    )[1][0]
                    counts = np.bincount(
                        labels_shared.detach().cpu().numpy(), minlength=codebook_size
                    ).astype(np.float64)
                    cb_init = cb_shared.detach().cpu().numpy()
                usage_shared = np.broadcast_to(counts, (groups, codebook_size)).copy()
                quantizer.init_codebooks([cb_init])
                quantizer.init_prior_from_freq(usage_shared)
                del samples, Z, sub_all, cb_shared, labels_shared
                torch.cuda.empty_cache()
            else:
                quantizer.init_codebooks(codebooks)
                quantizer.init_prior_from_freq(usage)
            history_path.write_text("", encoding="utf-8")
            print(f"loaded OPQ init {opq_path}")

    distortion_mode = str(config.get("distortion", "vector"))
    selection_key = "val_post_ln_patch_mse" if distortion_mode == "post_ln" else "val_vector_mse"
    print(f"distortion={distortion_mode} lmbda={float(config['lmbda'])} select={selection_key}")
    if config.get("rate_cap_bits_per_token") is not None:
        print(
            f"rate cap={float(config['rate_cap_bits_per_token'])} bits/token "
            f"weight={float(config.get('rate_cap_weight', 0.0))}"
        )
    best_val = min((item.get(selection_key, float("inf")) for item in history), default=float("inf"))
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    for epoch in range(start_epoch, epochs):
        codec.train()
        tau = temperature(
            epoch,
            epochs,
            float(config["tau_start"]),
            float(config["tau_end"]),
            str(config["tau_schedule"]),
        )
        codec.pq.temperature = tau
        order = list(train_files)
        random.Random(int(config["seed"]) + epoch).shuffle(order)
        distortion_total = 0.0
        rate_total = 0.0
        commit_total = 0.0
        tokens_total = 0
        usage_epoch = torch.zeros(groups, codebook_size, device=device)
        started = time.time()
        steps = math.ceil(len(order) / batch_size)
        progress = tqdm(range(steps), desc=f"train {epoch + 1}/{epochs}")
        for step in progress:
            batch = order[step * batch_size : (step + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)
            for path in batch:
                raw = load_feature(path, device)
                distortion, rate_bits, commit_value, usage_step = accumulate_image(
                    raw,
                    codec,
                    prefix_rms,
                    patch_rms,
                    int(config["n_prefix"]),
                    int(config["chunk_tokens"]),
                    float(config["lmbda"]),
                    len(batch),
                    float(config.get("commit_weight", 0.0)),
                    tail,
                    distortion_mode,
                    None if config.get("rate_cap_bits_per_token") is None else float(config["rate_cap_bits_per_token"]),
                    float(config.get("rate_cap_weight", 0.0)),
                )
                distortion_total += distortion
                rate_total += rate_bits
                commit_total += commit_value
                tokens_total += int(raw.shape[1])
                usage_epoch += usage_step
                del raw
            torch.nn.utils.clip_grad_norm_(codec.parameters(), float(config["grad_clip"]))
            optimizer.step()
            progress.set_postfix(tau=f"{tau:.4f}")
        scheduler.step()
        val_metrics = evaluate(
            val_files,
            codec,
            tail if tail is not None else _NullTail(),
            prefix_rms,
            patch_rms,
            int(config["n_prefix"]),
            int(config["chunk_tokens"]),
            device,
        )
        element_count = max(tokens_total * embed_dim, 1)
        row = {
            "epoch": epoch + 1,
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "temperature": tau,
            "commitment_mse": commit_total / element_count,
            "distortion_sum": distortion_total / max(len(order), 1),
            "rate_bits_per_image": rate_total / max(len(order), 1),
            "rate_bits_per_token": rate_total / max(tokens_total, 1),
            "perplexity": compute_perplexity(usage_epoch),
            "dead_entries": int((usage_epoch == 0).sum().item()),
            "orth_error": transform.orth_error() if transform is not None else 0.0,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        if distortion_mode == "post_ln":
            row["train_post_ln_mse"] = distortion_total / element_count
        else:
            row["vector_mse"] = distortion_total / element_count
        history.append(row)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        save_checkpoint(last_path, codec, optimizer, scheduler, epoch + 1, history, config)
        if row[selection_key] < best_val:
            best_val = row[selection_key]
            save_checkpoint(output_dir / "best.pt", codec, optimizer, scheduler, epoch + 1, history, config)

    best_path = output_dir / "best.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    codec.load_state_dict(best_checkpoint["codec"])
    print(
        f"export best epoch {best_checkpoint['epoch']} "
        f"{selection_key}={best_val:.8f}",
        flush=True,
    )
    codec.eval()
    pmf = histogram_pmf(
        train_files,
        codec,
        prefix_rms,
        patch_rms,
        int(config["n_prefix"]),
        int(config["chunk_tokens"]),
        device,
    )
    npz_path = output_dir / f"stage1_K{codebook_size}_e{embedding_dim}.npz"
    save_codec_npz(
        codec,
        npz_path,
        pmf,
        norm_mode="prefix_position_patch_channel_rms",
        n_prefix=int(config["n_prefix"]),
        metadata={
            "proposal": "PQFCv2",
            "backbone": "dinov3_vitl16",
            "layer": config["layer"],
            "slot": int(config["slot"]),
            "K": codebook_size,
            "embedding_dim": embedding_dim,
            "prefix_position_channel_rms": prefix_np,
            "patch_channel_rms": patch_np,
            "rms_eps": float(config["rms_eps"]),
        },
    )
    print(f"npz={npz_path}")


if __name__ == "__main__":
    main()

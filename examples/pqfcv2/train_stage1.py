#!/usr/bin/env python3
"""Train PQFCv2 stage 1: orthogonal transform and one PQ codebook.

Features stay at their native token length. Without a grid index, a step
accumulates gradients over ``batch_size`` images, each forwarded alone,
following VQ-UFC stage 1. With ``paths.grids``, images that share a padded
token count are stacked into one forward. RoPE is still built from each
image's own grid, so the stacked tail matches running those images alone.
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
from cofai.backbone.frozen_tail import Qwen3VLVisionFrozenTail
from cofai.ops.orfc import batched_assign, batched_kmeans
from cofai.utils.utils_vqufc import load_rms
from examples.orfc_2446.offline.artifacts import save_codec_npz
from examples.orfc_2446.offline.soft_pq import (
    FeatureCodec,
    OrthogonalTransform,
    SoftPQ,
    compute_perplexity,
)
from examples.pqfcv2.grids import feature_key, load_grids, make_token_batches


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
    backbone_type = str(config.get("backbone_type", "dinov3"))
    for key, value in config["paths"].items():
        expanded = os.path.expandvars(str(value))
        candidate = Path(expanded).expanduser()
        if key == "backbone" and backbone_type == "qwen3vl":
            if candidate.is_absolute():
                config["paths"][key] = str(candidate)
            elif (root / candidate).exists():
                config["paths"][key] = str((root / candidate).resolve())
            else:
                config["paths"][key] = expanded
        else:
            config["paths"][key] = str(candidate if candidate.is_absolute() else (root / candidate).resolve())
    return config


def oversample_train_domain(files: list[Path], feature_root: Path, domain: str, fraction: float, seed: int) -> list[Path]:
    """Repeat a deterministic subset of one training domain, keeping all files."""
    if not 0 <= fraction <= 1:
        raise ValueError("domain oversampling fraction must be in [0, 1]")
    matching = [path for path in files if path.relative_to(feature_root).parts[0] == domain]
    if not matching:
        raise ValueError(f"No training files found for domain {domain!r}")
    return list(files) + random.Random(seed).sample(matching, int(len(matching) * fraction))


def apply_rms(
    features: torch.Tensor,
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
) -> torch.Tensor:
    channels = features.shape[-1]
    if n_prefix == 0:
        return features / patch_rms.view(1, 1, channels)
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
    if n_prefix == 0:
        return normalized * patch_rms.view(1, 1, -1)
    length = normalized.shape[1]
    positions = torch.arange(token_offset, token_offset + length, device=normalized.device)
    scale = patch_rms.view(1, 1, -1).expand(1, length, -1).clone()
    prefix_positions = positions < n_prefix
    if bool(prefix_positions.any()):
        scale[0, prefix_positions] = prefix_rms[positions[prefix_positions]]
    return normalized * scale


def load_feature(
    path: Path,
    device: torch.device,
    embed_dim: int = 1024,
    n_prefix: int = 5,
) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != embed_dim or array.shape[0] <= n_prefix:
        raise ValueError(f"Expected [tokens>{n_prefix}, {embed_dim}] in {path}, got {array.shape}")
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
    embed_dim = int(patch_rms.numel())
    rng = np.random.RandomState(seed)
    buffer = np.empty((capacity, embed_dim), dtype=np.float32)
    filled = 0
    seen = 0
    for path in tqdm(files, desc="opq-sample"):
        raw = load_feature(path, device, embed_dim=embed_dim, n_prefix=n_prefix)
        vectors = apply_rms(raw, prefix_rms, patch_rms, n_prefix).reshape(-1, embed_dim).cpu().numpy()
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
    embed_dim = int(patch_rms.numel())
    groups = len(codebooks)
    counts = np.zeros((groups, codebooks[0].shape[0]), dtype=np.float64)
    rotation_t = torch.from_numpy(rotation).float().to(device)
    codebook_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    for path in tqdm(files, desc="opq-usage"):
        raw = load_feature(path, device, embed_dim=embed_dim, n_prefix=n_prefix)
        flat = apply_rms(raw, prefix_rms, patch_rms, n_prefix).reshape(-1, embed_dim)
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


def assign_batch_rope(tail, grids: list[tuple[int, int]]) -> None:
    """Point the tail at one RoPE per sample.

    A shared grid keeps the single ``[tokens, dim]`` embedding the backbone
    cached. Mixed grids at the same token count are stacked as
    ``[batch, 1, tokens, dim]`` so each row matches its own forward.
    """
    if hasattr(tail, "set_grids"):
        tail.set_grids(grids)
        return
    module = getattr(tail, "rope_module", None)
    if module is None:
        raise RuntimeError("a multi-block tail needs the DINOv3 RoPE module")
    if getattr(tail, "rope_mixed", False):
        raise RuntimeError("mixed RoPE cannot be packed across a token-count batch")
    embeds = [module.get_embed(shape=(int(height), int(width))) for height, width in grids]
    if len(embeds) == 1:
        tail.rope = embeds[0]
        return
    packed = torch.stack(embeds, dim=0)
    if packed.ndim != 3:
        raise RuntimeError(f"expected RoPE [batch, tokens, dim], got {tuple(packed.shape)}")
    tail.rope = packed.unsqueeze(1)


def accumulate_batch(
    raw: torch.Tensor,
    codec: FeatureCodec,
    prefix_rms: torch.Tensor,
    patch_rms: torch.Tensor,
    n_prefix: int,
    lmbda: float,
    commit_weight: float = 0.0,
    tail=None,
    distortion_mode: str = "vector",
    rate_cap_bits_per_token: float | None = None,
    rate_cap_weight: float = 0.0,
    tail_dtype: torch.dtype | None = None,
) -> tuple[float, float, float, torch.Tensor]:
    """Mean of the per-image losses for a same-length stack ``[batch, tokens, dim]``."""
    if distortion_mode not in ("vector", "post_ln"):
        raise ValueError(f"distortion must be 'vector' or 'post_ln', got {distortion_mode}")
    if distortion_mode == "post_ln" and tail is None:
        raise ValueError("post_ln distortion needs the frozen tail")
    batch, length, _channels = raw.shape
    normalized = apply_rms(raw, prefix_rms, patch_rms, n_prefix)
    restored, usage = codec(normalized)
    autocast = tail_dtype is not None
    if distortion_mode == "post_ln":
        decoded = invert_rms(restored, prefix_rms, patch_rms, n_prefix, 0)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=tail_dtype, enabled=autocast):
                teacher = tail(raw)
        with torch.autocast("cuda", dtype=tail_dtype, enabled=autocast):
            student = tail(decoded)
        distortion = (teacher.float() - student.float()).square().sum()
        del decoded, teacher, student
    else:
        distortion = (normalized - restored).square().sum()
    rate_bits = codec._last_rate * length * batch
    loss = rate_bits + lmbda * distortion
    commit_value = 0.0
    if rate_cap_bits_per_token is not None and rate_cap_weight > 0:
        excess = (rate_bits - float(rate_cap_bits_per_token) * length * batch).clamp_min(0)
        loss = loss + float(rate_cap_weight) * excess
    if commit_weight > 0:
        commit = (codec._last_z - codec._last_z_hat.detach()).square().sum()
        loss = loss + commit_weight * commit
        commit_value = float(commit.detach())
    (loss / batch).backward()
    return float(distortion.detach()), float(rate_bits.detach()), commit_value, usage.detach()


def accumulate_token_normalized_batch(
    raw, codec, prefix_rms, patch_rms, n_prefix, lmbda, vector_weight,
    tail, microbatch_tokens, relative_vector=False, rate_weight=1.0,
):
    """Backpropagate an equal-image token-normalized loss in bounded token chunks.

    This path is for the final DINO block, whose frozen tail is tokenwise LN.
    Each chunk contributes its exact fraction of the full image objective.
    """
    if tail is None or len(getattr(tail, "blocks", [])):
        raise ValueError("token microbatching requires a tokenwise final LayerNorm tail")
    batch, length, _ = raw.shape
    if microbatch_tokens < batch:
        raise ValueError("microbatch_tokens must be at least the image batch size")
    normalized = apply_rms(raw, prefix_rms, patch_rms, n_prefix)
    step = max(1, microbatch_tokens // batch)
    usage_total = None
    distortion_total = rate_total = vector_total = 0.0
    for start, end in iter_token_chunks(length, step):
        restored, usage = codec(normalized[:, start:end])
        decoded = invert_rms(restored, prefix_rms, patch_rms, n_prefix, start)
        with torch.no_grad():
            teacher = tail.forward_nograd(raw[:, start:end])
        student = tail(decoded)
        distortion = (teacher.float() - student.float()).square().sum()
        vector_error = (normalized[:, start:end] - restored).square()
        if relative_vector:
            # Give each token a comparable vector penalty despite energy variation.
            energy = normalized[:, start:end].square().mean(dim=-1, keepdim=True).clamp_min(1e-4)
            vector = (vector_error / energy).sum()
        else:
            vector = vector_error.sum()
        rate_bits = codec._last_rate * ((end - start) * batch)
        loss = (rate_weight * rate_bits + lmbda * distortion + vector_weight * vector) / (batch * length)
        loss.backward()
        distortion_total += float(distortion.detach())
        vector_total += float(vector.detach())
        rate_total += float(rate_bits.detach())
        usage_total = usage.detach() if usage_total is None else usage_total + usage.detach()
    return distortion_total, rate_total, vector_total, usage_total


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
    grids: dict[str, tuple[int, int]] | None = None,
    feature_root: Path | None = None,
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
    domains: dict[str, dict[str, float]] = {}
    embed_dim = int(patch_rms.numel())
    for path in files:
        raw = load_feature(path, device, embed_dim=embed_dim, n_prefix=n_prefix)
        if grids is not None and len(getattr(tail, "blocks", [])) > 0:
            assign_batch_rope(tail, [grids[feature_key(path, feature_root)]])
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
        diff = teacher.float() - student.float()
        sum_all += float(diff.square().sum())
        count_all += int(diff.numel())
        patch = diff[:, n_prefix:]
        sum_patch += float(patch.square().sum())
        count_patch += int(patch.numel())
        domain = path.parent.name
        if domain in ("ade", "nyu"):
            stats = domains.setdefault(domain, {"patch_sum": 0.0, "patch_count": 0, "vector_sum": 0.0, "vector_count": 0})
            stats["patch_sum"] += float(patch.square().sum())
            stats["patch_count"] += int(patch.numel())
            stats["vector_sum"] += float(vector_diff.square().sum())
            stats["vector_count"] += int(vector_diff.numel())
        raw_diff = raw - decoded
        sum_raw += float(raw_diff.square().sum())
        count_raw += int(raw_diff.numel())
    results = {
        "vector_mse": sum_vector / max(count_vector, 1),
        "post_ln_all_mse": sum_all / max(count_all, 1),
        "post_ln_patch_mse": sum_patch / max(count_patch, 1),
        "raw_mse": sum_raw / max(count_raw, 1),
    }
    for domain, stats in domains.items():
        results[f"{domain}_post_ln_patch_mse"] = stats["patch_sum"] / max(stats["patch_count"], 1)
        results[f"{domain}_vector_mse"] = stats["vector_sum"] / max(stats["vector_count"], 1)
    return results


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
    embed_dim = int(patch_rms.numel())
    groups, codebook_size = codec.pq.G, codec.pq.K
    counts = np.zeros((groups, codebook_size), dtype=np.int64)
    for path in tqdm(files, desc="pmf"):
        raw = load_feature(path, device, embed_dim=embed_dim, n_prefix=n_prefix)
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
    oversample_domain = config.get("oversample_domain")
    if oversample_domain is not None:
        train_files = oversample_train_domain(
            train_files, feature_root, str(oversample_domain),
            float(config.get("oversample_fraction", 0.0)), int(config["seed"]),
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

    backbone_type = str(config.get("backbone_type", "dinov3"))
    print(
        f"train={len(train_files)} validation={len(val_files)} K={codebook_size} "
        f"e={embedding_dim} G={groups} backbone={backbone_type}"
    )
    if args.no_transform:
        tail = None
        print("--no_transform: skipping backbone/tail load")
    elif backbone_type == "qwen3vl":
        tail_dtype_name = str(config.get("tail_dtype", "bfloat16"))
        tail_weight_dtype = torch.bfloat16 if tail_dtype_name in ("bfloat16", "bf16") else torch.float32
        tail = Qwen3VLVisionFrozenTail(
            model_path=config["paths"]["backbone"],
            slot=int(config["slot"]),
            device=device,
            dtype=tail_weight_dtype,
            attn_implementation=str(config.get("attn_implementation", "sdpa")),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        if not config["paths"].get("grids"):
            raise RuntimeError("Qwen3-VL visual tail requires paths.grids")
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
        if len(getattr(tail, "blocks", [])) != 0 and not config["paths"].get("grids"):
            raise RuntimeError("Stage 1 expects the final DINOv3 block, whose tail is LayerNorm only")
        del backbone

    rot_lr = config.get("rot_lr")
    quantizer = SoftPQ(
        groups,
        codebook_size,
        embedding_dim,
        lmbda=float(config.get("assignment_lmbda", config["lmbda"])),
        prior_floor=float(config["prior_floor"]),
        shared=bool(config.get("shared_codebook", False)),
        soft_rate=bool(config.get("soft_rate", False)),
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
    grids = load_grids(config["paths"]["grids"]) if config["paths"].get("grids") else None
    if config.get("token_normalized_loss", False):
        if grids is None or distortion_mode != "post_ln":
            raise ValueError("token_normalized_loss requires grids and post_ln distortion")
        if config.get("commit_weight", 0) or config.get("rate_cap_weight", 0):
            raise ValueError("token_normalized_loss does not support commitment or rate-cap terms")
    tail_dtype_name = str(config.get("tail_dtype", "float32"))
    if tail_dtype_name in ("float32", "fp32"):
        tail_dtype = None
    elif tail_dtype_name in ("bfloat16", "bf16"):
        tail_dtype = torch.bfloat16
    else:
        raise ValueError(f"tail_dtype must be float32 or bfloat16, got {tail_dtype_name}")
    if grids is not None:
        print(
            f"token-count batches size={batch_size} "
            f"max_batch_tokens={config.get('max_batch_tokens')} tail_dtype={tail_dtype_name}"
        )

    def export_training_snapshot(epoch: int, step: int | None = None) -> None:
        """Keep an immutable, directly evaluable artifact while training continues."""
        pmf = codec.get_prior_pmf()
        floor = float(config["prior_floor"])
        pmf = (1.0 - floor) * pmf + floor / codebook_size
        name = f"epoch_{epoch:02d}" if step is None else f"epoch_{epoch:02d}_step_{step:05d}"
        temporary = output_dir / f".{name}.tmp.npz"
        destination = output_dir / f"{name}.npz"
        save_codec_npz(
            codec, temporary, pmf,
            norm_mode="prefix_position_patch_channel_rms" if int(config["n_prefix"]) else "patch_channel_rms",
            n_prefix=int(config["n_prefix"]),
            metadata={
                "K": codebook_size, "embedding_dim": embedding_dim,
                "lmbda": float(config.get("assignment_lmbda", config["lmbda"])),
                "epoch": epoch,
                "step": 0 if step is None else step,
            },
        )
        os.replace(temporary, destination)

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
        if grids is None:
            random.Random(int(config["seed"]) + epoch).shuffle(order)
            batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        else:
            batches = make_token_batches(
                order,
                feature_root,
                grids,
                int(config["n_prefix"]),
                batch_size,
                int(config["seed"]) + epoch,
                None if config.get("max_batch_tokens") is None else int(config["max_batch_tokens"]),
            )
        distortion_total = 0.0
        rate_total = 0.0
        commit_total = 0.0
        tokens_total = 0
        usage_epoch = torch.zeros(groups, codebook_size, device=device)
        started = time.time()
        progress = tqdm(batches, desc=f"train {epoch + 1}/{epochs}")
        rate_cap = None if config.get("rate_cap_bits_per_token") is None else float(config["rate_cap_bits_per_token"])
        for batch_index, batch in enumerate(progress):
            optimizer.zero_grad(set_to_none=True)
            if grids is None:
                for path in batch:
                    raw = load_feature(path, device, embed_dim=embed_dim, n_prefix=int(config["n_prefix"]))
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
                        rate_cap,
                        float(config.get("rate_cap_weight", 0.0)),
                    )
                    distortion_total += distortion
                    rate_total += rate_bits
                    commit_total += commit_value
                    tokens_total += int(raw.shape[1])
                    usage_epoch += usage_step
                    del raw
            else:
                raw = torch.cat(
                    [
                        load_feature(path, device, embed_dim=embed_dim, n_prefix=int(config["n_prefix"]))
                        for path in batch
                    ],
                    dim=0,
                )
                if tail is not None and len(getattr(tail, "blocks", [])) > 0:
                    assign_batch_rope(tail, [grids[feature_key(path, feature_root)] for path in batch])
                if config.get("token_normalized_loss", False):
                    distortion, rate_bits, commit_value, usage_step = accumulate_token_normalized_batch(
                        raw, codec, prefix_rms, patch_rms, int(config["n_prefix"]),
                        float(config["lmbda"]), float(config.get("vector_weight", 0.0)), tail,
                        int(config.get("microbatch_tokens", config["chunk_tokens"])),
                        bool(config.get("relative_vector", False)),
                        float(config.get("rate_weight", 1.0)),
                    )
                else:
                    distortion, rate_bits, commit_value, usage_step = accumulate_batch(
                        raw, codec, prefix_rms, patch_rms, int(config["n_prefix"]),
                        float(config["lmbda"]), float(config.get("commit_weight", 0.0)),
                        tail, distortion_mode, rate_cap, float(config.get("rate_cap_weight", 0.0)),
                        tail_dtype,
                    )
                distortion_total += distortion
                rate_total += rate_bits
                commit_total += commit_value
                tokens_total += int(raw.shape[0]) * int(raw.shape[1])
                usage_epoch += usage_step
                del raw
            torch.nn.utils.clip_grad_norm_(codec.parameters(), float(config["grad_clip"]))
            optimizer.step()
            interval = int(config.get("snapshot_interval_steps", 0))
            if epoch == 0 and interval > 0 and (batch_index + 1) % interval == 0:
                export_training_snapshot(epoch, batch_index + 1)
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
            grids,
            feature_root,
        )
        element_count = max(tokens_total * embed_dim, 1)
        row = {
            "epoch": epoch + 1,
            "seconds": time.time() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "temperature": tau,
            ("train_relative_vector_mse" if config.get("relative_vector", False) else
             "train_vector_mse" if config.get("token_normalized_loss", False) else "commitment_mse"): commit_total / element_count,
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
        if config.get("export_each_epoch", False):
            export_training_snapshot(epoch + 1)

    best_path = output_dir / "best.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    codec.load_state_dict(best_checkpoint["codec"])
    print(
        f"export best epoch {best_checkpoint['epoch']} "
        f"{selection_key}={best_val:.8f}",
        flush=True,
    )
    codec.eval()
    if config.get("export_learned_prior", False):
        pmf = codec.get_prior_pmf()
        floor = float(config["prior_floor"])
        pmf = (1.0 - floor) * pmf + floor / codebook_size
    else:
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
    n_prefix = int(config["n_prefix"])
    default_backbone_name = "qwen3vl_8b_visual" if backbone_type == "qwen3vl" else "dinov3_vitl16"
    norm_mode = "patch_channel_rms" if n_prefix == 0 else "prefix_position_patch_channel_rms"
    export_meta = {
        "proposal": "PQFCv2",
        "backbone": str(config.get("backbone_name", default_backbone_name)),
        "layer": config["layer"],
        "slot": int(config["slot"]),
        "K": codebook_size,
        "embedding_dim": embedding_dim,
        "lmbda": float(config.get("assignment_lmbda", config["lmbda"])),
        "patch_channel_rms": patch_np,
        "rms_eps": float(config["rms_eps"]),
    }
    if n_prefix > 0:
        export_meta["prefix_position_channel_rms"] = prefix_np
    save_codec_npz(
        codec,
        npz_path,
        pmf,
        norm_mode=norm_mode,
        n_prefix=n_prefix,
        metadata=export_meta,
    )
    print(f"npz={npz_path}")


if __name__ == "__main__":
    main()

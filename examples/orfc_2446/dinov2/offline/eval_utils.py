"""Shared ORFC artifact and rate helpers for DINOv2 offline diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from cofai.engine.bitrate import bits_from_coded_unit
from cofai.latent_codecs import OrthoRotationFeatureCodec

from weights_paths import codec_weights_dir


_FEATURE_DIMENSIONS = {
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def reconstruct_features(features, codec, device, batch_size: int = 32):
    """Reconstruct a feature collection with the runtime ORFC codec."""
    codec.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(np.stack(features[start : start + batch_size])).float().to(device)
            reconstruction = codec(batch)["h_hat"]
            outputs.extend(item.cpu().numpy() for item in reconstruction)
    return outputs


def real_rate_bits(features, codec, device) -> tuple[float, int]:
    """Count the emitted rANS and normalization streams."""
    codec.eval()
    total_bits = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for feature in features:
            batch = torch.from_numpy(feature).float().unsqueeze(0).to(device)
            coded = codec.compress(batch)
            total_bits += sum(bits_from_coded_unit(coded).values())
            total_tokens += int(feature.shape[0])
    return total_bits, total_tokens


def resolve_artifact_path(args) -> str:
    """Resolve exactly one released ``.npz`` from explicit or training options."""
    if args.ckpt_path:
        path = Path(args.ckpt_path).expanduser()
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Offline ORFC evaluation requires a .npz artifact: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"ORFC artifact not found: {path}")
        return str(path)

    weights_dir = Path(codec_weights_dir(args.weights_dir, args.backbone))
    feature_dim = _FEATURE_DIMENSIONS[args.backbone]
    bottleneck_dim = args.bottleneck_dim if args.bottleneck_dim > 0 else feature_dim
    transform_tag = f"bt{bottleneck_dim}"
    init_tag = "ws" if args.warm_start_opq else "km"
    mse_tag = "_mse" if args.mse_loss else ""
    rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    filename = (
        f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
        f"_{transform_tag}_{init_tag}{mse_tag}{rate_tag}{tau_tag}"
        f"_lr{args.lr}_ep{args.epochs}"
        f"_n{args.max_train_images}_s{args.seed}.npz"
    )
    exact = weights_dir / filename
    if exact.is_file():
        return str(exact)

    pattern = f"{args.layer}_K{args.K}_emb{args.embedding_dim}_*.npz"
    matches = sorted(weights_dir.glob(pattern))
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(
            f"No ORFC artifact matching {pattern} under {weights_dir}; " "pass an explicit --ckpt_path"
        )
    raise FileNotFoundError(
        "Ambiguous ORFC artifacts; pass --ckpt_path. Candidates:\n  - "
        + "\n  - ".join(os.fspath(path) for path in matches)
    )


def load_eval_codec(
    artifact_path: str,
    device,
    *,
    norm_mode: str | None = None,
) -> OrthoRotationFeatureCodec:
    """Instantiate the same runtime codec used by formal plans."""
    return OrthoRotationFeatureCodec(
        orfc_weights_path=artifact_path,
        norm_mode=norm_mode,
    ).to(device)

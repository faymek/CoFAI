"""Training / eval distortion metrics aligned with ORFC delta_L_ref."""

from __future__ import annotations

import numpy as np
import torch

from cofai.latent_codecs.orfc_normalization import (
    denormalize_orfc_features,
    normalize_orfc_features,
)
from examples.orfc_2446.offline.soft_pq import codec_forward


@torch.inference_mode()
def eval_delta_l_ref(
    features: list[np.ndarray],
    codec,
    tail,
    *,
    norm_mode: str,
    n_prefix: int,
    device: torch.device,
    batch_size: int = 4,
    patch_only: bool = True,
) -> dict:
    if tail is None:
        raise ValueError("tail is required for delta_L_ref evaluation")

    codec.eval()
    sum_all = 0.0
    sum_patch = 0.0
    n_all = 0
    n_patch = 0

    same_shape = len({tuple(f.shape) for f in features}) == 1
    if same_shape:
        batches = [
            features[start : min(start + batch_size, len(features))]
            for start in range(0, len(features), batch_size)
        ]
    else:
        batches = [[f] for f in features]

    for batch_feats in batches:
        x = torch.from_numpy(np.stack(batch_feats)).float().to(device)
        y_teacher = tail.forward_nograd(x)
        y, mu, std = normalize_orfc_features(x, mode=norm_mode, n_prefix=n_prefix)
        y_hat, _ = codec_forward(y, codec)
        x_hat = denormalize_orfc_features(
            y_hat, mu, std, mode=norm_mode, n_prefix=n_prefix
        )
        y_student = tail.forward_nograd(x_hat)

        diff = y_teacher - y_student
        sum_all += float((diff**2).sum().item())
        n_all += diff.numel()
        if n_prefix > 0 and n_prefix < diff.shape[1]:
            diff_p = diff[:, n_prefix:, :]
            sum_patch += float((diff_p**2).sum().item())
            n_patch += diff_p.numel()

    out = {
        "post_ln_all_mse": sum_all / max(n_all, 1),
        "post_ln_patch_mse": (
            sum_patch / max(n_patch, 1) if n_patch else sum_all / max(n_all, 1)
        ),
    }
    if patch_only:
        out["delta_l_ref_mse"] = out["post_ln_patch_mse"]
    else:
        out["delta_l_ref_mse"] = out["post_ln_all_mse"]
    return out


def eval_reconstruction_mse(
    features: list[np.ndarray],
    codec,
    *,
    norm_mode: str,
    n_prefix: int,
    device: torch.device,
) -> dict:
    from lib.orfc_codec import encode_decode_single

    codec.eval()
    sq = []
    for feat in features:
        recon = encode_decode_single(
            feat, codec, norm_mode, device, n_prefix=n_prefix
        )
        sq.append(float(np.mean((recon - feat) ** 2)))
    return {"raw_mse": float(np.mean(sq))}


def eval_val_distortion(
    val_features: list[np.ndarray],
    codec,
    tail,
    *,
    norm_mode: str,
    n_prefix: int,
    device: torch.device,
    batch_size: int = 4,
) -> dict:
    raw = eval_reconstruction_mse(
        val_features,
        codec,
        norm_mode=norm_mode,
        n_prefix=n_prefix,
        device=device,
    )
    if tail is None:
        return {
            **raw,
            "post_ln_patch_mse": None,
            "post_ln_all_mse": None,
            "delta_l_ref_mse": None,
        }
    post_ln = eval_delta_l_ref(
        val_features,
        codec,
        tail,
        norm_mode=norm_mode,
        n_prefix=n_prefix,
        device=device,
        batch_size=batch_size,
        patch_only=True,
    )
    return {**raw, **post_ln}

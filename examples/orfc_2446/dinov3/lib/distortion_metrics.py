"""Training / eval distortion metrics aligned with ORFC delta_L_ref."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from cofai.entropy_models.orfc_model import batch_inv_normalize_gpu, batch_normalize_gpu
from cofai.entropy_models.soft_pq import codec_forward

DEBUG_LOG = Path("/data4/workspace/zlt/.cursor/debug-3d5c7c.log")
DEBUG_SESSION = "3d5c7c"


def _dbg(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "post-fix") -> None:
    # #region agent log
    payload = {
        "sessionId": DEBUG_SESSION,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG, "a") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


@torch.inference_mode()
def eval_delta_l_ref(
    features: list[np.ndarray],
    codec,
    tail,
    *,
    norm_mode: str,
    n_prefix: int,
    prefix_bypass: bool,
    device: torch.device,
    batch_size: int = 4,
    patch_only: bool = True,
) -> dict:
    """ORFC evaluate_delta_l_ref on a feature list (hard PQ eval).

    Returns mean squared error in post-LN space (per-element mean).
    Supports variable token lengths (evaluated per image).
    """
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
            features[start:min(start + batch_size, len(features))]
            for start in range(0, len(features), batch_size)
        ]
    else:
        batches = [[f] for f in features]

    for batch_feats in batches:
        x = torch.from_numpy(np.stack(batch_feats)).float().to(device)
        y_teacher = tail.forward_nograd(x)
        y, mu, std = batch_normalize_gpu(x, mode=norm_mode, n_prefix=n_prefix)
        y_hat, _ = codec_forward(y, codec, n_prefix, prefix_bypass=prefix_bypass)
        x_hat = batch_inv_normalize_gpu(y_hat, mu, std)
        y_student = tail.forward_nograd(x_hat)

        diff = y_teacher - y_student
        sum_all += float((diff ** 2).sum().item())
        n_all += diff.numel()
        if n_prefix > 0 and n_prefix < diff.shape[1]:
            diff_p = diff[:, n_prefix:, :]
            sum_patch += float((diff_p ** 2).sum().item())
            n_patch += diff_p.numel()

    out = {
        "post_ln_all_mse": sum_all / max(n_all, 1),
        "post_ln_patch_mse": sum_patch / max(n_patch, 1) if n_patch else sum_all / max(n_all, 1),
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
    prefix_bypass: bool,
    device: torch.device,
) -> dict:
    """Pre-LN raw feature MSE after encode/decode (legacy Val MSE)."""
    from lib.orfc_codec import encode_decode_single

    codec.eval()
    sq = []
    for feat in features:
        recon = encode_decode_single(
            feat, codec, norm_mode, device,
            n_prefix=n_prefix, prefix_bypass=prefix_bypass,
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
    prefix_bypass: bool,
    device: torch.device,
    batch_size: int = 4,
) -> dict:
    """Combined validation metrics for training end-of-run reporting."""
    raw = eval_reconstruction_mse(
        val_features, codec,
        norm_mode=norm_mode, n_prefix=n_prefix,
        prefix_bypass=prefix_bypass, device=device,
    )
    if tail is None:
        metrics = {**raw, "post_ln_patch_mse": None, "post_ln_all_mse": None, "delta_l_ref_mse": None}
    else:
        post_ln = eval_delta_l_ref(
            val_features, codec, tail,
            norm_mode=norm_mode, n_prefix=n_prefix,
            prefix_bypass=prefix_bypass, device=device,
            batch_size=batch_size, patch_only=True,
        )
        metrics = {**raw, **post_ln}

    _dbg("FIX", "distortion_metrics:eval_val", "val distortion metrics", metrics)
    return metrics

"""Per-token-group MSE / relative-MSE metrics for codec replay diagnosis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from tqdm import tqdm

from cofai.entropy_models.soft_pq import load_codec, load_codec_meta

from lib.dataset_utils import build_backbone
from lib.orfc_codec import encode_decode_single, resolve_n_prefix

GROUPS = ("cls", "reg", "patch", "prefix", "all")


def split_groups(x: np.ndarray, n_prefix: int) -> dict[str, np.ndarray]:
    return {
        "cls": x[:1],
        "reg": x[1:n_prefix],
        "patch": x[n_prefix:],
        "prefix": x[:n_prefix],
        "all": x,
    }


@torch.inference_mode()
def apply_ln(
    tokens: np.ndarray, backbone, meta: dict, device: torch.device,
) -> np.ndarray:
    h = torch.from_numpy(tokens).unsqueeze(0).to(device=device, dtype=torch.float32)
    h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
    img = torch.zeros(1, 3, h_p * 16, w_p * 16, device=device)
    backbone.encode(img)
    out = backbone.model.norm(h)
    return out.squeeze(0).float().cpu().numpy()


def resolve_prefix_bypass(
    ckpt_path: str | Path,
    prefix_bypass: Optional[bool],
) -> bool:
    if prefix_bypass is not None:
        return prefix_bypass
    meta = load_codec_meta(str(ckpt_path))
    return meta.get("train_tokens", "all") == "patch"


def analyze_codec_mse(
    ckpt_path: str | Path,
    *,
    token_dir: Path,
    meta_dir: Path,
    norm_mode: str,
    n_prefix: int,
    device: torch.device,
    stems: Iterable[str],
    prefix_bypass: Optional[bool] = None,
    backbone=None,
    verbose: bool = True,
) -> dict:
    """Compute all/patch/prefix MSE and pre/post-LN relative MSE per group."""
    ckpt_path = Path(ckpt_path)
    prefix_bypass = resolve_prefix_bypass(ckpt_path, prefix_bypass)
    stems = list(stems)
    own_backbone = backbone is None
    if own_backbone:
        from lib.config_utils import load_config
        backbone = build_backbone(load_config(), device)

    codec = load_codec(str(ckpt_path), device=device)
    sum_rel = {stage: {g: 0.0 for g in GROUPS} for stage in ("pre_ln", "post_ln")}
    sum_mse = {stage: {g: 0.0 for g in GROUPS} for stage in ("pre_ln", "post_ln")}
    sum_eng = {stage: {g: 0.0 for g in GROUPS} for stage in ("pre_ln", "post_ln")}
    n = 0
    t0 = time.time()

    iterator = tqdm(stems, desc=ckpt_path.stem) if verbose else stems
    for stem in iterator:
        tokens = np.load(token_dir / f"{stem}.npy")
        meta = dict(np.load(meta_dir / f"{stem}.npz", allow_pickle=True))
        recon = encode_decode_single(
            tokens, codec, norm_mode, device,
            n_prefix=n_prefix, prefix_bypass=prefix_bypass,
        )
        ref_ln = apply_ln(tokens, backbone, meta, device)
        hat_ln = apply_ln(recon, backbone, meta, device)

        for stage, ref, hat in (
            ("pre_ln", tokens, recon),
            ("post_ln", ref_ln, hat_ln),
        ):
            rg = split_groups(ref, n_prefix)
            hg = split_groups(hat, n_prefix)
            for g in GROUPS:
                r, h = rg[g], hg[g]
                mse = float(np.mean((h - r) ** 2))
                eng = float(np.mean(r ** 2))
                sum_mse[stage][g] += mse
                sum_eng[stage][g] += eng
                sum_rel[stage][g] += mse / max(eng, 1e-12)
        n += 1

    out = {
        "ckpt_path": str(ckpt_path),
        "prefix_bypass": prefix_bypass,
        "norm_mode": norm_mode,
        "n_prefix": n_prefix,
        "n_samples": n,
        "elapsed_s": time.time() - t0,
        "pre_ln": {},
        "post_ln": {},
    }
    for stage in ("pre_ln", "post_ln"):
        for g in GROUPS:
            mean_mse = sum_mse[stage][g] / n
            mean_eng = sum_eng[stage][g] / n
            pooled_rel = mean_mse / max(mean_eng, 1e-12)
            out[stage][g] = {
                "mean_rel_mse": sum_rel[stage][g] / n,
                "pooled_rel_mse": pooled_rel,
                "mean_abs_mse": mean_mse,
                "mean_energy": mean_eng,
            }

    if own_backbone:
        del backbone
    del codec
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out

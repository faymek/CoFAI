#!/usr/bin/env python3
"""Diagnose train distortion vs eval post-LN MSE / ORFC reference alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_COFAI_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.entropy_models.orfc_model import batch_inv_normalize_gpu, batch_normalize_gpu
from cofai.entropy_models.soft_pq import codec_forward, load_codec

from lib.config_utils import load_config, resolve_project_root, task_feat_dir
from lib.dataset_utils import build_backbone
from lib.dinov3_frozen_tail import build_dinov3_tail
from lib.orfc_codec import encode_decode_single, resolve_n_prefix
from lib.token_mse_metrics import apply_ln, split_groups

DEBUG_LOG = Path("/data4/workspace/zlt/.cursor/debug-3d5c7c.log")
SESSION = "3d5c7c"


def _dbg(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    # #region agent log
    payload = {
        "sessionId": SESSION,
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
def _delta_l_ref_batch(
    x: torch.Tensor,
    codec,
    tail,
    *,
    norm_mode: str,
    n_prefix: int,
    prefix_bypass: bool,
    patch_only_dist: bool,
    hard_eval: bool,
) -> tuple[float, float]:
    """ORFC-style delta_L_ref; returns (all-token mean sq, patch-only mean sq)."""
    B = x.shape[0]
    y_teacher = tail.forward_nograd(x)
    y, mu, std = batch_normalize_gpu(x, mode=norm_mode, n_prefix=n_prefix)
    if hard_eval:
        codec.eval()
        y_hat, _ = codec_forward(y, codec, n_prefix, prefix_bypass=prefix_bypass)
    else:
        codec.train()
        y_hat, _ = codec_forward(y, codec, n_prefix, prefix_bypass=prefix_bypass)
    x_hat = batch_inv_normalize_gpu(y_hat, mu, std)
    y_student = tail.forward_nograd(x_hat)
    diff = y_teacher - y_student
    if patch_only_dist and n_prefix > 0:
        diff_use = diff[:, n_prefix:, :]
    else:
        diff_use = diff
    mse = float((diff_use ** 2).mean().item())
    mse_all = float((diff ** 2).mean().item())
    return mse_all, mse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="weights/orfc_2446_dinov3/blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42.pt",
    )
    ap.add_argument("--norm_mode", default="split_cls_patch")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    cfg = load_config()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    n_prefix = resolve_n_prefix(args.norm_mode, 0, cfg.get("n_prefix", 5))

    feat_dir = task_feat_dir(cfg, "semseg")
    stems = sorted(p.stem for p in (feat_dir / "tokens").glob("*.npy"))[: args.n_samples]

    backbone = build_backbone(cfg, device)
    tail = build_dinov3_tail(backbone, cfg["layer_idx"], tuple(cfg["coco_token_hw"]), device)
    codec = load_codec(args.ckpt, device=device)
    meta_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    lmbda = float(meta_ckpt.get("lmbda", 0.0))
    train_tokens = meta_ckpt.get("train_tokens", "all")

    _dbg("B", "main:setup", "checkpoint meta", {
        "lmbda": lmbda, "train_tokens": train_tokens,
        "has_transform": meta_ckpt.get("has_transform"),
        "use_rate": lmbda > 0,
    })

    raw_mse_list = []
    postln_mse_list = []
    delta_all_list = []
    delta_patch_list = []
    delta_hard_patch_list = []
    prefix_postln_list = []

    for stem in stems:
        tokens = np.load(feat_dir / "tokens" / f"{stem}.npy")
        meta = dict(np.load(feat_dir / "meta" / f"{stem}.npz", allow_pickle=True))
        prefix_bypass = train_tokens == "patch" and n_prefix > 0

        recon = encode_decode_single(
            tokens, codec, args.norm_mode, device,
            n_prefix=n_prefix, prefix_bypass=prefix_bypass,
        )
        ref_ln = apply_ln(tokens, backbone, meta, device)
        hat_ln = apply_ln(recon, backbone, meta, device)
        postln_patch_mse = float(np.mean(
            (split_groups(hat_ln, n_prefix)["patch"] - split_groups(ref_ln, n_prefix)["patch"]) ** 2
        ))
        raw_mse = float(np.mean((recon - tokens) ** 2))
        postln_mse_list.append(postln_patch_mse)
        raw_mse_list.append(raw_mse)

        x = torch.from_numpy(tokens).unsqueeze(0).float().to(device)
        d_all, d_patch = _delta_l_ref_batch(
            x, codec, tail,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
            prefix_bypass=prefix_bypass,
            patch_only_dist=False,
            hard_eval=False,
        )
        _, d_hard_patch = _delta_l_ref_batch(
            x, codec, tail,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
            prefix_bypass=prefix_bypass,
            patch_only_dist=True,
            hard_eval=True,
        )
        delta_all_list.append(d_all)
        delta_patch_list.append(d_patch)
        delta_hard_patch_list.append(d_hard_patch)

        # prefix post-LN error (often ignored in patch-only distortion)
        pref_mse = float(np.mean(
            (split_groups(hat_ln, n_prefix)["prefix"] - split_groups(ref_ln, n_prefix)["prefix"]) ** 2
        ))
        prefix_postln_list.append(pref_mse)

    # H-A: train distortion metric vs eval post-LN patch MSE
    _dbg("A", "main:metrics", "eval postln vs delta_l_ref", {
        "mean_postln_patch_mse": float(np.mean(postln_mse_list)),
        "mean_raw_mse": float(np.mean(raw_mse_list)),
        "mean_delta_l_all": float(np.mean(delta_all_list)),
        "mean_delta_l_patch": float(np.mean(delta_patch_list)),
        "mean_delta_l_hard_patch": float(np.mean(delta_hard_patch_list)),
        "mean_prefix_postln_mse": float(np.mean(prefix_postln_list)),
        "ratio_postln_to_delta_patch": float(
            np.mean(postln_mse_list) / max(np.mean(delta_patch_list), 1e-12)
        ),
    })

    # H-C: soft vs hard gap on same batch
    _dbg("C", "main:soft_hard", "soft-train vs hard-eval delta_L patch", {
        "mean_delta_soft_patch": float(np.mean(delta_patch_list)),
        "mean_delta_hard_patch": float(np.mean(delta_hard_patch_list)),
        "gap_hard_minus_soft": float(np.mean(delta_hard_patch_list) - np.mean(delta_patch_list)),
    })

    # H-D: norm round-trip identity before LN
    x0 = torch.from_numpy(np.load(feat_dir / "tokens" / f"{stems[0]}.npy")).unsqueeze(0).float().to(device)
    y0, mu0, std0 = batch_normalize_gpu(x0, mode=args.norm_mode, n_prefix=n_prefix)
    x_rt = batch_inv_normalize_gpu(y0, mu0, std0)
    ln_orig = tail.forward_nograd(x0)
    ln_rt = tail.forward_nograd(x_rt)
    _dbg("D", "main:norm_roundtrip", "split_norm roundtrip then LN", {
        "ln_mse_after_roundtrip": float(((ln_orig - ln_rt) ** 2).mean().item()),
        "raw_mse_after_roundtrip": float(((x0 - x_rt) ** 2).mean().item()),
    })

    # H-E: training log D scale vs per-element mean (blk23 ~1376 patch * 1024 dim)
    approx_elems = 1376 * 1024
    train_D_ep99 = 169705.1  # from K256 noR log; user ckpt similar scale
    _dbg("E", "main:scale", "training D vs mean element MSE", {
        "train_D_logged_ep99": train_D_ep99,
        "implied_mean_sq_per_elem_if_div_B_only": train_D_ep99 / approx_elems,
        "eval_postln_patch_mse": float(np.mean(postln_mse_list)),
    })

    print("=== distortion alignment diagnosis ===")
    print(f"ckpt: {Path(args.ckpt).name}")
    print(f"lmbda={lmbda}  train_tokens={train_tokens}  n_prefix={n_prefix}")
    print(f"eval post-LN patch MSE: {np.mean(postln_mse_list):.6f}")
    print(f"eval raw MSE:           {np.mean(raw_mse_list):.6f}")
    print(f"delta_L_ref all:        {np.mean(delta_all_list):.6f}")
    print(f"delta_L_ref patch:      {np.mean(delta_patch_list):.6f}")
    print(f"delta_L_ref hard patch: {np.mean(delta_hard_patch_list):.6f}")
    print(f"prefix post-LN MSE:     {np.mean(prefix_postln_list):.6f}")
    print(f"logs -> {DEBUG_LOG}")


if __name__ == "__main__":
    main()

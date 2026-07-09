#!/usr/bin/env python3
"""Measure post-LN patch MSE for ptpatch: OPQ warm-start (pre-train) vs trained ckpt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_COFAI_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
_ORFC2446_OFFLINE = _COFAI_ROOT / "examples/orfc_2446/offline"
for _p in (_COFAI_ROOT, _EXAMPLE_DIR, _ORFC2446_OFFLINE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.entropy_models.orfc_model import batch_normalize_gpu, learn_orfc_rotation
from cofai.entropy_models.soft_pq import FeatureCodec, OrthogonalTransform, SoftPQ, load_codec
from utils import preload_features, set_seed

from lib.config_utils import load_config, resolve_project_root
from lib.dataset_utils import build_backbone
from lib.dinov3_frozen_tail import build_dinov3_tail
from lib.distortion_metrics import eval_delta_l_ref
from lib.orfc_codec import resolve_n_prefix

PTPATCH_CKPTS = [
    ("K4_e32", "blk23_K4_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt", 4, 32),
    ("K16_e32", "blk23_K16_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt", 16, 32),
    ("K256_e32", "blk23_K256_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt", 256, 32),
    ("K512_e32", "blk23_K512_emb32_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt", 512, 32),
    ("K64_e16", "blk23_K64_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt", 64, 16),
    ("K256_e16", "blk23_K256_emb16_bt1024_ws_split_cls_patch_tau0.5_lr0.0003_ep100_n1000_s42_ptpatch.pt", 256, 16),
]


def _load_train_val_features(feat_dir: Path, max_train: int, n_val: int, seed: int):
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {feat_dir}")

    rng = np.random.RandomState(seed)
    n_total = len(files)
    n_need = min(n_total, max_train + n_val) if max_train > 0 else n_total
    if n_need < n_total:
        pick = rng.choice(n_total, n_need, replace=False)
        files = [files[i] for i in sorted(pick)]

    features, _ = preload_features(files, num_workers=8)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(features))
    n_val = min(n_val, len(features))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if max_train > 0:
        train_idx = train_idx[:max_train]
    train_feats = [features[i] for i in train_idx]
    val_feats = [features[i] for i in val_idx]
    return train_feats, val_feats


def _norm_vectors_from_batch(Y: torch.Tensor, n_prefix: int, D: int) -> np.ndarray:
    if n_prefix > 0:
        return Y[:, n_prefix:, :].reshape(-1, D).cpu().numpy()
    return Y.reshape(-1, D).cpu().numpy()


def _opq_warm_start(
    train_array: np.ndarray,
    *,
    G: int,
    K: int,
    emb: int,
    norm_mode: str,
    n_prefix: int,
    D: int,
    kmeans_max_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, list[np.ndarray]]:
    all_vectors = []
    n_train = train_array.shape[0]
    for start in range(0, n_train, 200):
        end = min(start + 200, n_train)
        X = torch.from_numpy(train_array[start:end]).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        all_vectors.append(_norm_vectors_from_batch(Y, n_prefix, D))
        del X, Y
    full_vecs = np.concatenate(all_vectors, axis=0)
    del all_vectors

    max_flat = kmeans_max_samples // G
    if full_vecs.shape[0] > max_flat:
        rng = np.random.RandomState(seed)
        full_vecs = full_vecs[rng.choice(full_vecs.shape[0], max_flat, replace=False)]

    R_opq, cb_opq, _ = learn_orfc_rotation(
        full_vecs, G, emb, K,
        max_iter_orfc=20, max_iter_kmeans=100, device=device, verbose=False,
    )
    del full_vecs
    torch.cuda.empty_cache()
    return R_opq, cb_opq


def _build_warm_codec(
    R_opq: np.ndarray,
    cb_opq: list[np.ndarray],
    *,
    G: int,
    K: int,
    emb: int,
    D: int,
    device: torch.device,
) -> FeatureCodec:
    R_ws = R_opq.copy()
    C_ws = [c.copy() for c in cb_opq]
    if np.linalg.det(R_ws) < 0:
        R_ws[:, -1] *= -1
        C_ws[-1][:, -1] *= -1

    pq = SoftPQ(G, K, emb, lmbda=0.0, prior_floor=0.0).to(device)
    transform = OrthogonalTransform(D).to(device)
    codec = FeatureCodec(pq, transform).to(device)
    transform.init_from_opq(R_ws)
    pq.init_codebooks(C_ws)
    codec.eval()
    return codec


@torch.inference_mode()
def _eval_postln_patch_mse(
    val_feats: list[np.ndarray],
    codec,
    tail,
    *,
    norm_mode: str,
    n_prefix: int,
    device: torch.device,
    batch_size: int,
) -> dict:
    return eval_delta_l_ref(
        val_feats, codec, tail,
        norm_mode=norm_mode,
        n_prefix=n_prefix,
        prefix_bypass=True,
        device=device,
        batch_size=batch_size,
        patch_only=True,
    )


def measure_one(
    tag: str,
    ckpt_name: str,
    K: int,
    emb: int,
    *,
    weights_dir: Path,
    feat_dir: Path,
    norm_mode: str,
    n_prefix: int,
    max_train: int,
    n_val: int,
    seed: int,
    kmeans_max_samples: int,
    batch_size: int,
    device: torch.device,
    layer_idx: int,
    token_hw: tuple[int, int],
    backbone_cfg: dict,
    skip_opq: bool = False,
) -> dict:
    ckpt_path = weights_dir / ckpt_name
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    train_feats, val_feats = _load_train_val_features(feat_dir, max_train, n_val, seed)
    train_array = np.stack(train_feats)
    D = train_array.shape[-1]
    G = D // emb

    backbone = build_backbone(backbone_cfg, device)
    tail = build_dinov3_tail(backbone, layer_idx, token_hw, device)

    t0 = time.time()
    if skip_opq:
        warm_metrics = {"post_ln_patch_mse": None, "delta_l_ref_mse": None}
        opq_sec = 0.0
    else:
        R_opq, cb_opq = _opq_warm_start(
            train_array,
            G=G, K=K, emb=emb,
            norm_mode=norm_mode, n_prefix=n_prefix, D=D,
            kmeans_max_samples=kmeans_max_samples, seed=seed, device=device,
        )
        opq_sec = time.time() - t0
        warm_codec = _build_warm_codec(
            R_opq, cb_opq, G=G, K=K, emb=emb, D=D, device=device,
        )
        warm_metrics = _eval_postln_patch_mse(
            val_feats, warm_codec, tail,
            norm_mode=norm_mode, n_prefix=n_prefix,
            device=device, batch_size=batch_size,
        )
        del warm_codec, R_opq, cb_opq
        torch.cuda.empty_cache()

    trained_codec = load_codec(str(ckpt_path), device=device)
    trained_metrics = _eval_postln_patch_mse(
        val_feats, trained_codec, tail,
        norm_mode=norm_mode, n_prefix=n_prefix,
        device=device, batch_size=batch_size,
    )

    warm_mse = warm_metrics["post_ln_patch_mse"]
    trained_mse = trained_metrics["post_ln_patch_mse"]
    delta = None if warm_mse is None else trained_mse - warm_mse
    rel_pct = None
    if warm_mse is not None and warm_mse > 0:
        rel_pct = 100.0 * (trained_mse - warm_mse) / warm_mse

    tail.to("cpu")
    del tail, backbone, trained_codec
    torch.cuda.empty_cache()

    return {
        "tag": tag,
        "ckpt": ckpt_name,
        "K": K,
        "embedding_dim": emb,
        "n_val": len(val_feats),
        "n_train_opq": train_array.shape[0],
        "opq_seconds": round(opq_sec, 1),
        "warm_post_ln_patch_mse": warm_mse,
        "trained_post_ln_patch_mse": trained_mse,
        "delta_trained_minus_warm": delta,
        "delta_pct": rel_pct,
        "trained_post_ln_all_mse": trained_metrics["post_ln_all_mse"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=None, help="e.g. K256_e16; default: all configs")
    ap.add_argument("--weights_dir", default="weights/orfc_2446_dinov3")
    ap.add_argument("--norm_mode", default="split_cls_patch")
    ap.add_argument("--max_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip_opq", action="store_true", help="only eval trained ckpt")
    ap.add_argument("--out", default="examples/orfc_2446_dinov3/results/ptpatch_warm_vs_trained.json")
    args = ap.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    cfg = load_config()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(args.seed)

    n_prefix = resolve_n_prefix(args.norm_mode, 0, cfg.get("n_prefix", 5))
    feat_dir = Path(cfg["paths"]["train_feat_dir"])
    weights_dir = Path(args.weights_dir)
    layer_idx = int(cfg.get("layer", "blk23")[-2:])
    token_hw = tuple(cfg.get("coco_token_hw", [64, 85]))

    pairs = PTPATCH_CKPTS
    if args.tag:
        pairs = [p for p in PTPATCH_CKPTS if p[0] == args.tag]
        if not pairs:
            raise SystemExit(f"Unknown tag: {args.tag}")

    results = []
    for tag, ckpt_name, K, emb in pairs:
        print(f"\n=== {tag}  K={K} emb={emb} ===")
        row = measure_one(
            tag, ckpt_name, K, emb,
            weights_dir=weights_dir,
            feat_dir=feat_dir,
            norm_mode=args.norm_mode,
            n_prefix=n_prefix,
            max_train=args.max_train,
            n_val=args.n_val,
            seed=args.seed,
            kmeans_max_samples=args.kmeans_max_samples,
            batch_size=args.batch_size,
            device=device,
            layer_idx=layer_idx,
            token_hw=token_hw,
            backbone_cfg=cfg,
            skip_opq=args.skip_opq,
        )
        results.append(row)
        if row["warm_post_ln_patch_mse"] is not None:
            print(
                f"  warm  post-LN patch MSE: {row['warm_post_ln_patch_mse']:.6f}  "
                f"(OPQ {row['opq_seconds']}s)"
            )
        print(f"  train post-LN patch MSE: {row['trained_post_ln_patch_mse']:.6f}")
        if row["delta_trained_minus_warm"] is not None:
            sign = "+" if row["delta_trained_minus_warm"] >= 0 else ""
            print(
                f"  Δ (trained - warm): {sign}{row['delta_trained_minus_warm']:.6f}  "
                f"({sign}{row['delta_pct']:.2f}%)"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "norm_mode": args.norm_mode,
        "n_prefix": n_prefix,
        "max_train": args.max_train,
        "n_val": args.n_val,
        "seed": args.seed,
        "prefix_bypass": True,
        "metric": "post_ln_patch_mse (delta_L_ref, hard PQ)",
        "configs": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")

    print(f"\n{'tag':>10}  {'warm':>12}  {'trained':>12}  {'Δ':>12}  {'Δ%':>8}")
    print("-" * 60)
    for r in results:
        warm = r["warm_post_ln_patch_mse"]
        trained = r["trained_post_ln_patch_mse"]
        delta = r["delta_trained_minus_warm"]
        pct = r["delta_pct"]
        warm_s = f"{warm:.6f}" if warm is not None else "n/a"
        delta_s = f"{delta:+.6f}" if delta is not None else "n/a"
        pct_s = f"{pct:+.2f}%" if pct is not None else "n/a"
        print(f"{r['tag']:>10}  {warm_s:>12}  {trained:>12.6f}  {delta_s:>12}  {pct_s:>8}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export sidecar .npz (R, codebooks, pmf) from existing DINOv3 SoftPQ .pt checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_COFAI_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_ORFC2446_OFFLINE = _COFAI_ROOT / "examples/orfc_2446/dinov2/offline"
if str(_ORFC2446_OFFLINE) not in sys.path:
    sys.path.insert(0, str(_ORFC2446_OFFLINE))

import compressai  # noqa: F401

from cofai.entropy_models.soft_pq import load_codec, load_codec_meta
from cofai.entropy_models.soft_pq_export import (
    compute_histogram_pmf,
    npz_path_for_codec,
    save_codec_npz,
)
from utils import preload_features

from lib.config_utils import load_config, resolve_project_root
from lib.orfc_codec import NORM_MODE_CHOICES, resolve_norm_settings


def _load_train_features(feat_dir: Path, max_train: int, seed: int):
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {feat_dir}")
    n_need = min(len(files), max_train) if max_train > 0 else len(files)
    if n_need < len(files):
        rng = np.random.RandomState(seed)
        pick = rng.choice(len(files), n_need, replace=False)
        files = [files[i] for i in sorted(pick)]
    features, _ = preload_features(files, num_workers=8)
    return features


def export_one(
    ckpt_path: Path,
    *,
    feat_dir: Path,
    norm_mode: str | None,
    n_prefix: int,
    max_train: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    skip_existing: bool,
) -> bool:
    npz_path = npz_path_for_codec(ckpt_path)
    if skip_existing and npz_path.is_file():
        print(f"  [skip] {npz_path.name} exists")
        return True

    print(f"  Loading features from {feat_dir} (max_train={max_train})...")
    features = _load_train_features(feat_dir, max_train, seed)

    ckpt_meta = load_codec_meta(str(ckpt_path))
    norm_mode, n_prefix, norm_src = resolve_norm_settings(
        norm_mode=norm_mode,
        n_prefix=n_prefix,
        ckpt_path=str(ckpt_path),
        ckpt_meta=ckpt_meta,
        default_n_prefix=5,
    )
    print(f"  norm_mode={norm_mode}  n_prefix={n_prefix}  (source={norm_src})")

    codec = load_codec(str(ckpt_path), device=device)
    torch.cuda.empty_cache()

    pmf = compute_histogram_pmf(
        codec,
        features,
        norm_mode=norm_mode,
        n_prefix=n_prefix,
        device=device,
        batch_size=batch_size,
    )
    save_codec_npz(
        codec, npz_path, pmf, source_pt=ckpt_path,
        norm_mode=norm_mode, n_prefix=n_prefix,
    )
    ent = -(pmf * np.log2(pmf + 1e-30)).sum(axis=1).mean()
    print(f"  [ok] {ckpt_path.name} -> {npz_path.name} (n={len(features)}, H={ent:.4f})")
    del codec
    torch.cuda.empty_cache()
    return True


def main():
    cfg = load_config()
    defaults = cfg.get("train_defaults", {})
    p = argparse.ArgumentParser(description="Export SoftPQ evaluation .npz from optional train .pt")
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--weights_dir", type=str, default="")
    p.add_argument("--feat_dir", type=str, default=None)
    p.add_argument("--max_train", type=int, default=defaults.get("max_train", 1000))
    p.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    p.add_argument(
        "--norm_mode", choices=NORM_MODE_CHOICES, default=None,
        help="Feature norm (default: auto from ckpt meta/filename)",
    )
    p.add_argument("--n_prefix", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 4))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    feat_dir = Path(args.feat_dir or cfg["paths"]["train_feat_dir"])

    jobs: list[Path] = []
    if args.ckpt_path:
        jobs.append(Path(args.ckpt_path))
    else:
        wd = Path(args.weights_dir or cfg["paths"]["weights_dir"])
        jobs = sorted(wd.glob("blk23_K*_ep30_n1000_s42.pt"))

    ok = fail = 0
    for ckpt in jobs:
        if not ckpt.is_file():
            print(f"[error] missing {ckpt}")
            fail += 1
            continue
        print(f"\n{ckpt.name}")
        try:
            if export_one(
                ckpt,
                feat_dir=feat_dir,
                norm_mode=args.norm_mode,
                n_prefix=args.n_prefix,
                max_train=args.max_train,
                seed=args.seed,
                batch_size=args.batch_size,
                device=device,
                skip_existing=not args.force,
            ):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            fail += 1

    print(f"\nDone: ok={ok} failed={fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export ORFC-compatible sidecar .npz (R, codebooks, pmf) from Soft-PQ .pt."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()
OFFLINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_DIR.parents[2]
sys.path.insert(0, str(OFFLINE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
import compressai  # noqa: F401
from cofai.entropy_models.soft_pq import load_codec
from cofai.entropy_models.soft_pq_export import (
    compute_histogram_pmf, save_codec_npz, npz_path_for_codec, parse_layer_from_stem,
)
from utils import preload_features

BACKBONE_FROM_WEIGHTS_SUBDIR = {
    "dinov2_vitl14_ori": "dinov2_vitl14", "dinov2_vitl14": "dinov2_vitl14",
    "dinov2_vitg14_ori": "dinov2_vitg14", "dinov2_vitg14": "dinov2_vitg14",
}

def infer_backbone(weights_dir: Path, explicit: str) -> str:
    if explicit:
        return explicit
    name = weights_dir.name
    if name in BACKBONE_FROM_WEIGHTS_SUBDIR:
        return BACKBONE_FROM_WEIGHTS_SUBDIR[name]
    raise ValueError(f"Cannot infer --backbone from {weights_dir}")

def export_one(ckpt_path, *, backbone, layer, feat_root, max_train_images, seed, norm_mode, device, skip_existing):
    npz_path = npz_path_for_codec(ckpt_path)
    if skip_existing and npz_path.is_file():
        print(f"  [skip] {npz_path.name} exists")
        return True
    train_dir = feat_root / "train" / backbone / layer
    train_files = sorted(train_dir.glob("*.npy"))
    if not train_files:
        print(f"  [error] no train features in {train_dir}")
        return False
    features, _ = preload_features(train_files, num_workers=8)
    if max_train_images > 0 and len(features) > max_train_images:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(features), max_train_images, replace=False)
        features = [features[i] for i in idx]
    codec = load_codec(str(ckpt_path), device=device)
    pmf = compute_histogram_pmf(codec, features, norm_mode=norm_mode, device=device)
    save_codec_npz(codec, npz_path, pmf, source_pt=ckpt_path)
    ent = -(pmf * np.log2(pmf + 1e-30)).sum(axis=1).mean()
    print(f"  [ok] {ckpt_path.name} -> {npz_path.name} (n={len(features)}, H={ent:.4f})")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", default="")
    p.add_argument("--weights_dir", default="")
    p.add_argument("--backbone", default="")
    p.add_argument("--layer", default="")
    p.add_argument("--feat_root", required=True)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--norm_mode", default="per_image")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.force:
        args.skip_existing = False
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    feat_root = Path(args.feat_root)
    jobs = []
    if args.ckpt_path:
        ckpt = Path(args.ckpt_path)
        if not ckpt.is_absolute():
            ckpt = PROJECT_ROOT / ckpt
        layer = args.layer or parse_layer_from_stem(ckpt.stem)
        jobs.append((ckpt, infer_backbone(ckpt.parent, args.backbone), layer))
    else:
        wd = Path(args.weights_dir)
        if not wd.is_absolute():
            wd = PROJECT_ROOT / wd
        bb = infer_backbone(wd, args.backbone)
        for ckpt in sorted(wd.glob("*.pt")):
            layer = parse_layer_from_stem(ckpt.stem)
            if layer:
                jobs.append((ckpt, bb, layer))
    ok = failed = 0
    for ckpt, bb, layer in jobs:
        print(f"\n{ckpt.name} ({bb}/{layer})")
        if export_one(ckpt, backbone=bb, layer=layer, feat_root=feat_root,
                      max_train_images=args.max_train_images, seed=args.seed,
                      norm_mode=args.norm_mode, device=device,
                      skip_existing=args.skip_existing):
            ok += 1
        else:
            failed += 1
    print(f"\nDone: ok={ok} failed={failed}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()

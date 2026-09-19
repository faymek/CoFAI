#!/usr/bin/env python3
"""Extract ADE20K / NYUv2 val tokens+meta for PQFC offline replay."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_SOURCE_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.config_utils import load_config, resolve_project_root, task_feat_dir
from lib.dataset_utils import build_backbone, build_dataset, load_subset, sample_stem


def _fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def _iter_stems(dataset, task: str, cfg: dict, subset: set[str] | None):
    nyu_root = cfg["datasets"]["depth"]["root"] if task == "depth" else None
    for idx in range(len(dataset)):
        sample = dataset[idx]
        stem = sample_stem(sample, task, nyu_root=nyu_root)
        if subset is not None and stem not in subset:
            continue
        yield idx, stem, sample


def cmd_extract(args) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    task = args.task
    subset = load_subset(args.subset)
    dataset = build_dataset(task, cfg)
    backbone = build_backbone(cfg, device)

    feat_dir = task_feat_dir(cfg, task)
    token_dir = feat_dir / "tokens"
    meta_dir = feat_dir / "meta"
    token_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    n_total = n_skip = 0
    t0 = time.time()

    for _idx, stem, sample in tqdm(
        list(_iter_stems(dataset, task, cfg, subset)),
        desc=f"extract-{task}",
    ):
        n_total += 1
        token_path = token_dir / f"{stem}.npy"
        meta_path = meta_dir / f"{stem}.npz"
        if token_path.exists() and meta_path.exists():
            n_skip += 1
            continue

        img = sample["img"]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.asarray(img).transpose(2, 0, 1)).float()
        img = img.unsqueeze(0).to(device)
        with torch.inference_mode():
            h = backbone.encode(img)
        tokens = h.squeeze(0).float().cpu().numpy()
        h_p = img.shape[2] // cfg["patch_size"]
        w_p = img.shape[3] // cfg["patch_size"]

        np.save(token_path, tokens.astype(np.float32))
        meta = {
            "stem": stem,
            "token_hw": np.array([h_p, w_p], dtype=np.int32),
            "img_hw": np.array([img.shape[2], img.shape[3]], dtype=np.int32),
            "img_path": np.array([sample["meta"]["img_path"]], dtype=object),
        }
        if task == "semseg":
            meta["semseg_path"] = np.array(
                [sample["meta"].get("seg_label_path", "")], dtype=object
            )
        np.savez(meta_path, **meta)

    print(
        f"[extract] task={task} total={n_total} skipped={n_skip} "
        f"elapsed={time.time() - t0:.1f}s -> {token_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract CTC val tokens for PQFC replay")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--task", choices=["semseg", "depth"], required=True)
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("PROJECT_ROOT", str(resolve_project_root()))
    cmd_extract(args)


if __name__ == "__main__":
    main()

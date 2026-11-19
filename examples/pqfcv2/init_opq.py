#!/usr/bin/env python3
"""Build and save the PQFCv2 OPQ warm start.

The artifact stores the orthogonal rotation, PQ codebooks, and empirical codeword
counts. Stage 1 training loads it instead of repeating the reservoir sample,
ORFC rotation, and full-set assignment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cofai.ops.orfc import learn_orfc_rotation
from examples.pqfcv2.train_stage1 import (
    finalize_opq,
    load_config,
    load_rms,
    opq_usage_counts,
    reservoir_sample,
    save_opq_init,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = Path(config["paths"]["opq_init"])
    if output.is_file() and not args.force:
        print(f"OPQ init already exists: {output}")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    feature_root = Path(config["paths"]["feature_root"])
    split = json.loads(Path(config["paths"]["split_path"]).read_text(encoding="utf-8"))
    train_files = [feature_root / name for name in split["train"]]
    prefix_np, patch_np, rms_meta = load_rms(
        config["paths"]["rms_path"],
        train_files,
        feature_root,
        int(config["n_prefix"]),
    )
    prefix_rms = torch.from_numpy(prefix_np).to(device)
    patch_rms = torch.from_numpy(patch_np).to(device)
    embed_dim = int(config["embed_dim"])
    embedding_dim = int(config["embedding_dim"])
    groups = embed_dim // embedding_dim
    codebook_size = int(config["K"])
    sample_cap = max(1, int(config["kmeans_max_samples"]) // groups)
    print(
        f"train={len(train_files)} K={codebook_size} e={embedding_dim} "
        f"G={groups} sample_cap={sample_cap}",
        flush=True,
    )

    started = time.time()
    sampled = reservoir_sample(
        train_files,
        prefix_rms,
        patch_rms,
        int(config["n_prefix"]),
        sample_cap,
        int(config["seed"]),
        device,
    )
    print(f"reservoir {sampled.shape} in {time.time() - started:.1f}s", flush=True)
    rotation, codebooks, history = learn_orfc_rotation(
        sampled,
        groups,
        embedding_dim,
        codebook_size,
        max_iter_orfc=20,
        max_iter_kmeans=100,
        device=device,
        verbose=True,
    )
    del sampled
    rotation, codebooks = finalize_opq(rotation, codebooks)
    usage_started = time.time()
    usage = opq_usage_counts(
        train_files,
        rotation,
        codebooks,
        prefix_rms,
        patch_rms,
        int(config["n_prefix"]),
        device,
    )
    print(f"usage pass {time.time() - usage_started:.1f}s", flush=True)
    metadata = {
        "format_version": 1,
        "proposal": "PQFCv2",
        "split_sha256": rms_meta["split_sha256"],
        "train_files": len(train_files),
        "seed": int(config["seed"]),
        "sample_cap": sample_cap,
        "kmeans_max_samples": int(config["kmeans_max_samples"]),
        "K": codebook_size,
        "embedding_dim": embedding_dim,
        "groups": groups,
        "embed_dim": embed_dim,
        "n_prefix": int(config["n_prefix"]),
        "max_iter_orfc": 20,
        "max_iter_kmeans": 100,
        "elapsed_sec": time.time() - started,
    }
    save_opq_init(output, rotation, codebooks, usage, history, metadata)
    print(f"saved {output}  det={float(np.linalg.det(rotation)):.6f}", flush=True)


if __name__ == "__main__":
    main()

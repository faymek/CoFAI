#!/usr/bin/env python3
"""
ORFC Engine Evaluation Script
==============================
Evaluates ORFC (Optimized Rotation for Feature Compression) through
the ``cofai.engine.run_eval`` pipeline. Supports both classification
(ImageNet sel500) and segmentation (VOC2012 sel100) tasks.

This script provides a convenient CLI that maps high-level ORFC
parameters (backbone, layer, K, embedding_dim) to the corresponding
plan YAML and engine overrides, then invokes ``cofai.engine.run_eval``
via Hydra in the same process.

Example usage:

    # Single configuration (cls, real rANS)
    CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
        --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32 \
        --task cls --real --cuda

    # Single configuration (seg)
    CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
        --backbone dinov2_vitg14 --layer blk09 --K 8 --embedding_dim 32 \
        --task seg --real --cuda

    # Multi-run: sweep all K values defined in the plan YAML
    CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
        --backbone dinov2_vitl14 --layer blk10 --task cls --multi-run --real --cuda

    # Use engine directly (alternative — equivalent to the above multi-run)
    python -m cofai.engine.run_eval conf/plan/imagenet-sel500--ORFC-dinov2-vitl14-cls.yaml \
        args.cuda=true args.real=true args.multi_run=true
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VITL14_LAYERS = {"blk05": -18, "blk10": -13, "blk15": -8, "blk20": -3}
VITG14_LAYERS = {"blk09": -30, "blk19": -20, "blk29": -10}

PLAN_MAP = {
    ("dinov2_vitl14", "cls", "blk05"): "imagenet-sel500--ORFC-dinov2-vitl14-cls-blk05",
    ("dinov2_vitl14", "cls", "blk10"): "imagenet-sel500--ORFC-dinov2-vitl14-cls",
    ("dinov2_vitl14", "cls", "blk15"): "imagenet-sel500--ORFC-dinov2-vitl14-cls-blk15",
    ("dinov2_vitl14", "cls", "blk20"): "imagenet-sel500--ORFC-dinov2-vitl14-cls-blk20",
    ("dinov2_vitl14", "seg", "blk05"): "voc2012-sel100--ORFC-dinov2-vitl14-seg-blk05",
    ("dinov2_vitl14", "seg", "blk10"): "voc2012-sel100--ORFC-dinov2-vitl14-seg",
    ("dinov2_vitl14", "seg", "blk15"): "voc2012-sel100--ORFC-dinov2-vitl14-seg-blk15",
    ("dinov2_vitl14", "seg", "blk20"): "voc2012-sel100--ORFC-dinov2-vitl14-seg-blk20",
    ("dinov2_vitg14", "cls", "blk09"): "imagenet-sel500--ORFC-dinov2-vitg14-cls",
    ("dinov2_vitg14", "cls", "blk19"): "imagenet-sel500--ORFC-dinov2-vitg14-cls-blk19",
    ("dinov2_vitg14", "cls", "blk29"): "imagenet-sel500--ORFC-dinov2-vitg14-cls-blk29",
    ("dinov2_vitg14", "seg", "blk09"): "voc2012-sel100--ORFC-dinov2-vitg14-seg",
    ("dinov2_vitg14", "seg", "blk19"): "voc2012-sel100--ORFC-dinov2-vitg14-seg-blk19",
    ("dinov2_vitg14", "seg", "blk29"): "voc2012-sel100--ORFC-dinov2-vitg14-seg-blk29",
}


def resolve_weight_path(backbone: str, layer: str, K: int, embedding_dim: int) -> str:
    return str(PROJECT_ROOT / "weights" / "orfc" / backbone / f"{layer}_K{K}_e{embedding_dim}.npz")


def build_engine_cmd(args) -> list:
    """Build cofai.engine.run_eval command from CLI args."""
    plan_key = (args.backbone, args.task, args.layer)
    plan_name = PLAN_MAP.get(plan_key)
    if plan_name is None:
        avail = [k for k in PLAN_MAP if k[0] == args.backbone and k[1] == args.task]
        raise ValueError(
            f"No plan found for {plan_key}. "
            f"Available layers for {args.backbone}/{args.task}: "
            f"{[k[2] for k in avail]}"
        )

    plan_yaml = f"conf/plan/{plan_name}.yaml"
    cmd = [sys.executable, "-m", "cofai.engine.run_eval", plan_yaml]

    cmd.append(f"args.cuda={str(args.cuda).lower()}")
    cmd.append(f"args.real={str(args.real).lower()}")

    if args.multi_run:
        cmd.append("args.multi_run=true")
    else:
        weight_path = resolve_weight_path(args.backbone, args.layer, args.K, args.embedding_dim)
        cmd.append("args.multi_run=false")
        cmd.append(f"model.K={args.K}")
        cmd.append(f"model.embedding_dim={args.embedding_dim}")
        cmd.append(f"model.orfc_weights_path={weight_path}")
        cmd.append(f"args.quality={args.K}")

    if args.output_dir:
        cmd.append(f"args.output_dir={args.output_dir}")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="ORFC engine evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14"],
                        help="Backbone model")
    parser.add_argument("--layer", type=str, required=True,
                        help="Feature extraction layer (e.g., blk10, blk09)")
    parser.add_argument("--task", type=str, required=True,
                        choices=["cls", "seg"],
                        help="Evaluation task: cls or seg")
    parser.add_argument("--K", type=int, default=64,
                        help="Codebook size (default: 64)")
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="Sub-vector dimension (default: 32)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--real", action="store_true",
                        help="Use real compress/decompress (rANS entropy coding)")
    parser.add_argument("--multi-run", action="store_true",
                        help="Sweep all K values defined in the plan YAML")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the engine command without executing")

    args = parser.parse_args()

    layers = VITL14_LAYERS if "vitl14" in args.backbone else VITG14_LAYERS
    if args.layer not in layers:
        parser.error(f"Invalid layer '{args.layer}' for {args.backbone}. "
                     f"Choose from: {list(layers.keys())}")

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

    cmd = build_engine_cmd(args)

    if args.dry_run:
        print("Command:")
        print("  " + " \\\n    ".join(cmd))
        return

    print(f"[ORFC] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

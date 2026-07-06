#!/usr/bin/env python3
"""
Soft-PQ Engine Evaluation Script
==================================
Evaluates Soft-PQ (Differentiable Product Quantization) through the
``cofai.engine.run_eval`` pipeline using the new DinoFeatureCodecModel +
SoftPQFeatureCodec latent-codec architecture.

Supports classification (ImageNet sel500, Acc@1) and segmentation
(VOC2012 sel100, mIoU) tasks.

Example usage:

    # Single configuration
    CUDA_VISIBLE_DEVICES=0 python examples/soft_pq/run_eval_soft_pq.py \
        --backbone dinov2_vitl14 --layer blk20 --task cls --cuda

    # Multi-run sweep (all K values in plan)
    CUDA_VISIBLE_DEVICES=0 python examples/soft_pq/run_eval_soft_pq.py \
        --backbone dinov2_vitl14 --layer blk20 --task cls --multi-run --cuda

    # Dry-run (print command only)
    python examples/soft_pq/run_eval_soft_pq.py \
        --backbone dinov2_vitg14 --layer blk29 --task seg --multi-run --dry-run
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VITL14_LAYERS = ["blk05", "blk10", "blk15", "blk20"]
VITG14_LAYERS = ["blk09", "blk19", "blk29"]

SLOT_MAP = {
    "blk05": "slot06", "blk10": "slot11", "blk15": "slot16", "blk20": "slot21",
    "blk09": "slot10", "blk19": "slot20", "blk29": "slot30",
}

PLAN_DIR = "conf/plan/dinov2"


def resolve_plan(backbone: str, task: str, layer: str) -> str:
    slot_label = SLOT_MAP[layer]
    bb = backbone.replace("dinov2_", "dinov2-")  # dinov2_vitl14 -> dinov2-vitl14
    if task == "cls":
        name = f"imagenet-sel500__{bb}-{slot_label}__SoftPQ__cls"
    else:
        name = f"voc2012-sel100__{bb}-slide-{slot_label}__SoftPQ__semseg"
    return f"{PLAN_DIR}/{name}.yaml"


def build_engine_cmd(args) -> list:
    plan_yaml = resolve_plan(args.backbone, args.task, args.layer)
    plan_path = PROJECT_ROOT / plan_yaml
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan not found: {plan_path}")

    cmd = [sys.executable, "-m", "cofai.engine.run_eval", plan_yaml]
    cmd.append(f"args.cuda={str(args.cuda).lower()}")
    cmd.append(f"args.real={str(args.real).lower()}")

    if args.multi_run:
        cmd.append("args.multi_run=true")
    else:
        cmd.append("args.multi_run=false")
        if args.quality:
            cmd.append(f"args.quality={args.quality}")

    if args.output_dir:
        cmd.append(f"args.output_dir={args.output_dir}")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Soft-PQ engine evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--layer", type=str, required=True,
                        help="Feature layer (blk05/10/15/20 for vitl14, blk09/19/29 for vitg14)")
    parser.add_argument("--task", type=str, required=True,
                        choices=["cls", "seg"])
    parser.add_argument("--quality", type=str, default="",
                        help="Quality label for single-run (e.g., '16', '256_e16')")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--real", action="store_true",
                        help="Use real compress/decompress path")
    parser.add_argument("--multi-run", action="store_true",
                        help="Sweep all K values defined in the plan YAML")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    valid_layers = VITL14_LAYERS if "vitl14" in args.backbone else VITG14_LAYERS
    if args.layer not in valid_layers:
        parser.error(f"Invalid layer '{args.layer}' for {args.backbone}. "
                     f"Choose from: {valid_layers}")

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

    cmd = build_engine_cmd(args)

    if args.dry_run:
        print("Command:")
        print("  " + " \\\n    ".join(cmd))
        return

    print(f"[SoftPQ] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

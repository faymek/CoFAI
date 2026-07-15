#!/usr/bin/env python3
"""
ORFC-2446 Soft-PQ DINOv3 Engine Evaluation (code template)
=========================================================
Evaluates SoftPQ on DINOv3 ViT-L/16 slot24 via ``cofai.engine.run_eval``.

Requires a SoftPQ ``.npz`` (R + codebooks + pmf [+ norm_mode/n_prefix]).

Example:

    CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov3/run_eval_orfc_2446_dinov3.py \\
        --task semseg \\
        --ckpt_path weights/orfc_2446_dinov3/blk23_K4_emb32_....npz \\
        --cuda --real
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PLAN_MAP = {
    "semseg": "conf/plan/dinov3/ade20k-val__dinov3-vitl16-slot24__SoftPQ__semseg.yaml",
    "depth": "conf/plan/dinov3/nyuv2-val__dinov3-vitl16-slot24__SoftPQ__depth.yaml",
}

BACKBONE_CKPT = "weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.pth"


def main():
    parser = argparse.ArgumentParser(description="DINOv3 SoftPQ online eval (Engine)")
    parser.add_argument("--task", choices=["semseg", "depth"], required=True)
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="SoftPQ .npz weight (R+codebooks+pmf)")
    parser.add_argument("--norm_mode", type=str, default="",
                        help="Override npz meta (per_image|split_cls_patch|split_reg_cls_patch)")
    parser.add_argument("--n_prefix", type=int, default=-1,
                        help="Override npz meta n_prefix (-1 = use npz/plan default)")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--real", action="store_true", default=True,
                        help="Use real compress/decompress (default true when pmf present)")
    parser.add_argument("--no-real", dest="real", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    plan = PLAN_MAP[args.task]
    ckpt = Path(args.ckpt_path)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if ckpt.suffix == ".pt":
        npz = ckpt.with_suffix(".npz")
        if npz.is_file():
            ckpt = npz
    if ckpt.suffix != ".npz":
        parser.error(f"Expected SoftPQ .npz, got: {ckpt}")

    backbone = PROJECT_ROOT / BACKBONE_CKPT
    result_subdir = (
        Path(args.output_dir) / "SoftPQ" / "dinov3-vitl16-slot24" / args.task / ckpt.stem
    )

    cmd = [
        sys.executable, "-m", "cofai.engine.run_eval", plan,
        f"args.cuda={str(args.cuda).lower()}",
        f"args.real={str(args.real).lower()}",
        "args.multi_run=false",
        f"model.dino_backbone.ckpt_path={backbone}",
        f"model.dino_codec.codec_path={ckpt}",
        f"args.result_subdir={result_subdir}",
    ]
    if args.norm_mode:
        cmd.append(f"model.dino_codec.norm_mode={args.norm_mode}")
    if args.n_prefix >= 0:
        cmd.append(f"model.dino_codec.n_prefix={args.n_prefix}")

    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)

    if args.dry_run:
        print("  " + " \\\n    ".join(cmd))
        return

    print(f"[ORFC-2446 DINOv3] Running: {' '.join(cmd)}")
    sys.exit(subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode)


if __name__ == "__main__":
    main()

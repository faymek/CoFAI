#!/usr/bin/env python3
"""
ORFC-2446 (Soft-PQ) Engine Evaluation Script
===========================================
Evaluates Soft-PQ through the ``cofai.engine.run_eval`` pipeline using
``DinoFeatureCodecModel`` + ``SoftPQFeatureCodec``.

Each evaluation run targets one specific ``.pt`` checkpoint (``args.multi_run=false``),
with explicit overrides for codec path and local DINOv2 backbone weights.

This follows the same invocation pattern as ``run_full_softpq_eval.sh``.

Example usage:

    # Single checkpoint (cls)
    CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/run_eval_orfc_2446.py \
        --backbone dinov2_vitl14 --layer blk10 --task cls \
        --ckpt_path weights/orfc_2446/dinov2_vitl14_ori/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt \
        --cuda

    # Single checkpoint (seg)
    CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/run_eval_orfc_2446.py \
        --backbone dinov2_vitg14 --layer blk29 --task seg \
        --ckpt_path weights/orfc_2446/dinov2_vitg14_ori/blk29_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt \
        --cuda

    # Dry-run (print command only)
    python examples/orfc_2446/run_eval_orfc_2446.py \
        --backbone dinov2_vitl14 --layer blk05 --task cls \
        --ckpt_path weights/orfc_2446/dinov2_vitl14_ori/blk05_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt \
        --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLAN_DIR = "conf/plan/dinov2"

SLOT_MAP = {
    "blk05": "slot06",
    "blk10": "slot11",
    "blk15": "slot16",
    "blk20": "slot21",
    "blk09": "slot10",
    "blk19": "slot20",
    "blk29": "slot30",
}

BACKBONE_SHORT = {
    "dinov2_vitl14": "vitl14",
    "dinov2_vitg14": "vitg14",
}

BACKBONE_CKPT = {
    "dinov2_vitl14": "weights/dinov2/backbone/dinov2_vitl14_pretrain.pth",
    "dinov2_vitg14": "weights/dinov2/backbone/dinov2_vitg14_pretrain.pth",
}

VITL14_LAYERS = ["blk05", "blk10", "blk15", "blk20"]
VITG14_LAYERS = ["blk09", "blk19", "blk29"]

# Source-tree compressai (no compiled _CXX) must not shadow pip install.
_BAD_PYTHONPATH_MARKERS = (
    "ORFC/coding/CompressAI",
    "coding/CompressAI",
)


def engine_subprocess_env() -> dict:
    """Build env for cofai.engine.run_eval subprocess."""
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    pypath = env.get("PYTHONPATH", "")
    if pypath:
        parts = [p for p in pypath.split(os.pathsep) if p]
        cleaned = [
            p for p in parts
            if not any(marker in p.replace("\\", "/") for marker in _BAD_PYTHONPATH_MARKERS)
        ]
        if cleaned:
            env["PYTHONPATH"] = os.pathsep.join(cleaned)
        else:
            env.pop("PYTHONPATH", None)

    return env


def resolve_plan(backbone: str, task: str, layer: str) -> str:
    """Map backbone/layer/task to plan YAML under conf/plan/dinov2/."""
    bb_short = BACKBONE_SHORT[backbone]
    slot = SLOT_MAP[layer]
    if task == "cls":
        name = f"imagenet-sel500__dinov2-{bb_short}-{slot}__SoftPQ__cls"
    else:
        name = f"voc2012-sel100__dinov2-{bb_short}-slide-{slot}__SoftPQ__semseg"
    return f"{PLAN_DIR}/{name}.yaml"


_CODEC_TAG_RE = re.compile(r"_K(\d+)_emb(\d+)_")


def codec_config_tag(ckpt_path: Path) -> str:
    """Parse codec tag like ``K64e32`` from checkpoint filename."""
    match = _CODEC_TAG_RE.search(ckpt_path.stem)
    if not match:
        raise ValueError(
            f"Cannot parse codec config (K/emb) from checkpoint name: {ckpt_path.name}"
        )
    return f"K{match.group(1)}e{match.group(2)}"


def resolve_result_subdir(
    *,
    backbone: str,
    task: str,
    layer: str,
    ckpt_path: Path,
    output_base: str = "eval_results",
) -> str:
    """Build hierarchical result path, e.g. ``eval_results/SoftPQ/voc2012-sel100/.../K64e32``."""
    dataset = "imagenet-sel500" if task == "cls" else "voc2012-sel100"
    bb_short = BACKBONE_SHORT[backbone]
    bb_path = f"dinov2-{bb_short}-slide" if task == "seg" else f"dinov2-{bb_short}"
    slot = SLOT_MAP[layer]
    codec_tag = codec_config_tag(ckpt_path)
    base = Path(output_base) if output_base else Path("eval_results")
    return str(base / "SoftPQ" / dataset / bb_path / slot / codec_tag)


def resolve_ckpt_path(ckpt_path: str) -> Path:
    """Resolve checkpoint path relative to PROJECT_ROOT if needed."""
    path = Path(ckpt_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        hint = ""
        if path.parent.exists():
            stem = path.stem
            # Suggest same layer/K prefix matches
            prefix = "_".join(stem.split("_")[:3]) if "_" in stem else stem[:12]
            matches = sorted(path.parent.glob(f"{prefix}*.pt"))
            if matches:
                names = [m.name for m in matches[:3]]
                hint = f"\n  Available nearby: {', '.join(names)}"
        raise FileNotFoundError(f"Checkpoint not found: {path}{hint}")
    return path


def build_engine_cmd(args) -> list:
    plan_yaml = resolve_plan(args.backbone, args.task, args.layer)
    plan_path = PROJECT_ROOT / plan_yaml
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan not found: {plan_path}")

    ckpt_path = resolve_ckpt_path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    backbone_ckpt = PROJECT_ROOT / BACKBONE_CKPT[args.backbone]

    cmd = [sys.executable, "-m", "cofai.engine.run_eval", plan_yaml]
    cmd.append(f"args.cuda={str(args.cuda).lower()}")
    cmd.append("args.multi_run=false")
    cmd.append(f"model.dino_backbone.ckpt_path={backbone_ckpt}")
    cmd.append(f"model.dino_codec.codec_path={ckpt_path}")

    if args.result_subdir:
        result_subdir = args.result_subdir
    else:
        result_subdir = resolve_result_subdir(
            backbone=args.backbone,
            task=args.task,
            layer=args.layer,
            ckpt_path=ckpt_path,
            output_base=args.output_dir or "eval_results",
        )
    cmd.append(f"args.result_subdir={result_subdir}")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="ORFC-2446 (Soft-PQ) engine evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14"],
                        help="Backbone model")
    parser.add_argument("--layer", type=str, required=True,
                        help="Feature extraction layer (e.g. blk10, blk09)")
    parser.add_argument("--task", type=str, required=True,
                        choices=["cls", "seg"],
                        help="Evaluation task: cls or seg")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to Soft-PQ codec checkpoint (.pt)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Base directory for results (default: eval_results)")
    parser.add_argument("--result_subdir", type=str, default="",
                        help="Override result subdir (e.g. eval_results/SoftPQ_by_ckpt/<stem>_<task>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the engine command without executing")

    args = parser.parse_args()

    valid_layers = VITL14_LAYERS if args.backbone == "dinov2_vitl14" else VITG14_LAYERS
    if args.layer not in valid_layers:
        parser.error(
            f"Invalid layer '{args.layer}' for {args.backbone}. "
            f"Choose from: {valid_layers}"
        )

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cmd = build_engine_cmd(args)
    env = engine_subprocess_env()

    if args.dry_run:
        print("Command:")
        print("  " + " \\\n    ".join(cmd))
        if "PYTHONPATH" in env:
            print(f"PYTHONPATH={env['PYTHONPATH']}")
        else:
            print("PYTHONPATH=(unset)")
        return

    print(f"[ORFC-2446] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

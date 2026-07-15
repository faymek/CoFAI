#!/usr/bin/env python3
"""
ORFC-2446 (Soft-PQ) Engine Evaluation Script
===========================================
Evaluates Soft-PQ through ``cofai.engine.run_eval`` using
``DinoFeatureCodecModel`` + ``SoftPQFeatureCodec``.

Canonical weight format is a single ``.npz`` (R + codebooks + pmf), same as ORFC.
When the ``.npz`` contains ``pmf``, real rANS coding is used automatically.

Example usage:

    # Single config via K / embedding_dim (resolves under weights/orfc_2446/)
    CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \\
        --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32 \\
        --task cls --real --cuda

    # Or pass an explicit SoftPQ .npz
    CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \\
        --backbone dinov2_vitl14 --layer blk10 --task cls \\
        --ckpt_path weights/orfc_2446/dinov2_vitl14_ori/blk10_K64_emb32_....npz \\
        --cuda --real

    # Multi-run: sweep all codec_path entries in the SoftPQ plan YAML
    CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \\
        --backbone dinov2_vitl14 --layer blk10 --task cls --multi-run --cuda --real
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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

_BAD_PYTHONPATH_MARKERS = (
    "ORFC/coding/CompressAI",
    "coding/CompressAI",
)


def engine_subprocess_env() -> dict:
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
    bb_short = BACKBONE_SHORT[backbone]
    slot = SLOT_MAP[layer]
    if task == "cls":
        name = f"imagenet-sel500__dinov2-{bb_short}-{slot}__SoftPQ__cls"
    else:
        name = f"voc2012-sel100__dinov2-{bb_short}-slide-{slot}__SoftPQ__semseg"
    return f"{PLAN_DIR}/{name}.yaml"


_CODEC_TAG_RE = re.compile(r"_K(\d+)_emb(\d+)_")


def codec_config_tag(ckpt_path: Path) -> str:
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
    dataset = "imagenet-sel500" if task == "cls" else "voc2012-sel100"
    bb_short = BACKBONE_SHORT[backbone]
    bb_path = f"dinov2-{bb_short}-slide" if task == "seg" else f"dinov2-{bb_short}"
    slot = SLOT_MAP[layer]
    codec_tag = codec_config_tag(ckpt_path)
    # Include lambda/tau tags in path when present to avoid collisions
    stem = ckpt_path.stem
    extra = ""
    for key in ("lmbda", "tau", "lr", "ep"):
        m = re.search(rf"_{key}([0-9.]+)", stem)
        if m:
            extra += f"_{key}{m.group(1)}"
    base = Path(output_base) if output_base else Path("eval_results")
    return str(base / "SoftPQ" / dataset / bb_path / slot / f"{codec_tag}{extra}")


WEIGHTS_SUBDIR = {
    "dinov2_vitl14": "dinov2_vitl14_ori",
    "dinov2_vitg14": "dinov2_vitg14_ori",
}


def softpq_weights_dir(backbone: str) -> Path:
    return PROJECT_ROOT / "weights" / "orfc_2446" / WEIGHTS_SUBDIR[backbone]


def pick_softpq_npz(weights_dir: Path, layer: str, K: int, embedding_dim: int) -> Path:
    """Resolve ``{layer}_K{K}_emb{d}_*.npz``; disambiguate by preferring lmbda0.5."""
    matches = sorted(weights_dir.glob(f"{layer}_K{K}_emb{embedding_dim}_*.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No SoftPQ .npz matching {layer}_K{K}_emb{embedding_dim}_*.npz under {weights_dir}. "
            "Download/unzip weights or pass --ckpt_path."
        )
    if len(matches) == 1:
        return matches[0]
    preferred = [m for m in matches if "lmbda0.5" in m.stem]
    if len(preferred) == 1:
        return preferred[0]
    if preferred:
        return preferred[0]
    non_l02 = [m for m in matches if "lmbda0.2" not in m.stem]
    if len(non_l02) == 1:
        return non_l02[0]
    names = "\n  - ".join(m.name for m in matches)
    raise FileNotFoundError(
        f"Ambiguous SoftPQ weights for {layer} K={K} emb={embedding_dim}. "
        f"Pass --ckpt_path explicitly. Candidates:\n  - {names}"
    )


def resolve_ckpt_path(
    ckpt_path: str = "",
    *,
    backbone: str = "",
    layer: str = "",
    K: int | None = None,
    embedding_dim: int | None = None,
) -> Path:
    if ckpt_path:
        path = Path(ckpt_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.suffix == ".pt":
            npz = path.with_suffix(".npz")
            if npz.is_file():
                path = npz
        if not path.exists():
            hint = ""
            if path.parent.exists():
                prefix = "_".join(path.stem.split("_")[:3]) if "_" in path.stem else path.stem[:12]
                nearby = sorted(path.parent.glob(f"{prefix}*.npz"))
                if nearby:
                    hint = f"\n  Available nearby: {', '.join(m.name for m in nearby[:3])}"
            raise FileNotFoundError(f"Checkpoint not found: {path}{hint}")
        if path.suffix != ".npz":
            raise ValueError(
                f"SoftPQ online eval expects a .npz weight file (got {path}). "
                "Train/export SoftPQ to .npz first."
            )
        return path
    if not backbone or not layer or K is None or embedding_dim is None:
        raise ValueError(
            "Provide --ckpt_path, or (--backbone --layer --K --embedding_dim), or --multi-run."
        )
    return pick_softpq_npz(softpq_weights_dir(backbone), layer, K, embedding_dim)


def build_engine_cmd(args) -> list:
    plan_yaml = resolve_plan(args.backbone, args.task, args.layer)
    plan_path = PROJECT_ROOT / plan_yaml
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan not found: {plan_path}")

    backbone_ckpt = PROJECT_ROOT / BACKBONE_CKPT[args.backbone]
    cmd = [sys.executable, "-m", "cofai.engine.run_eval", plan_yaml]
    cmd.append(f"args.cuda={str(args.cuda).lower()}")
    cmd.append(f"args.real={str(args.real).lower()}")
    cmd.append(f"model.dino_backbone.ckpt_path={backbone_ckpt}")

    if args.multi_run:
        cmd.append("args.multi_run=true")
    else:
        ckpt_path = resolve_ckpt_path(
            args.ckpt_path,
            backbone=args.backbone,
            layer=args.layer,
            K=args.K,
            embedding_dim=args.embedding_dim,
        )
        cmd.append("args.multi_run=false")
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

    if args.output_dir and args.multi_run:
        cmd.append(f"args.output_dir={args.output_dir}")

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="ORFC-2446 (Soft-PQ) engine evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, choices=["cls", "seg"])
    parser.add_argument("--K", type=int, default=64,
                        help="Codebook size (used when --ckpt_path is omitted)")
    parser.add_argument("--embedding_dim", type=int, default=32,
                        help="Sub-vector dim (used when --ckpt_path is omitted)")
    parser.add_argument("--ckpt_path", type=str, default="",
                        help="SoftPQ .npz path (optional if K/embedding_dim resolve uniquely)")
    parser.add_argument("--multi-run", action="store_true",
                        help="Sweep all multi_run codec_path entries in the plan YAML")
    parser.add_argument("--real", action="store_true", default=True,
                        help="Real compress/decompress (uses pmf in .npz when present)")
    parser.add_argument("--no-real", dest="real", action="store_false")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--result_subdir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    valid_layers = VITL14_LAYERS if args.backbone == "dinov2_vitl14" else VITG14_LAYERS
    if args.layer not in valid_layers:
        parser.error(
            f"Invalid layer '{args.layer}' for {args.backbone}. "
            f"Choose from: {valid_layers}"
        )

    os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
    cmd = build_engine_cmd(args)
    env = engine_subprocess_env()

    if args.dry_run:
        print("Command:")
        print("  " + " \\\n    ".join(cmd))
        return

    print(f"[ORFC-2446] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

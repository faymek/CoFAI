#!/usr/bin/env python3
"""
ORFC-2446 Soft-PQ DINOv3 Engine Evaluation
==========================================
Online CTC eval for DINOv3 ViT-L/16 slot24 SoftPQ (ADE20K semseg / NYUv2 depth).

Examples:

    # Single checkpoint
    CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov3/run_eval_orfc_2446_dinov3.py \\
        --task semseg --ckpt_path weights/orfc_2446_dinov3/release/blk23_K256_e32.npz \\
        --cuda --real

    # Sweep all multi_run entries in the plan (multi-GPU parallel)
    python examples/orfc_2446/dinov3/run_eval_orfc_2446_dinov3.py \\
        --task both --multi-run --gpus 0,1,2,3 --cuda --real

    # One-click wrapper
    bash examples/orfc_2446/dinov3/scripts/run_eval_release_ctc.sh
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PLAN_MAP = {
    "semseg": "conf/plan/dinov3/ade20k-val__dinov3-vitl16-slot24__SoftPQ__semseg.yaml",
    "depth": "conf/plan/dinov3/nyuv2-val__dinov3-vitl16-slot24__SoftPQ__depth.yaml",
}

BACKBONE_CKPT = "weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.pth"
RELEASE_DIR = PROJECT_ROOT / "weights" / "orfc_2446_dinov3" / "release"

_WEIGHT_RE = re.compile(
    r"orfc_weights_path:\s*\$\{PROJECT_ROOT\}/(weights/orfc_2446_dinov3/release/blk23_K\d+_e\d+\.npz)"
)
_MULTI_KEY_RE = re.compile(r"^  ([^\s:]+):\s*$")


def engine_env() -> dict:
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def parse_multi_run_weights(plan_path: Path) -> list[tuple[str, str]]:
    """Return ordered ``[(quality_key, relative_npz_path), ...]`` from plan YAML."""
    text = plan_path.read_text(encoding="utf-8")
    if "multi_run:" not in text:
        raise ValueError(f"No multi_run section in {plan_path}")
    section = text.split("multi_run:", 1)[1]
    entries: list[tuple[str, str]] = []
    current_key: str | None = None
    for line in section.splitlines():
        key_m = _MULTI_KEY_RE.match(line)
        if key_m and not line.strip().startswith("#"):
            current_key = key_m.group(1)
            continue
        if current_key is None:
            continue
        w_m = _WEIGHT_RE.search(line)
        if w_m:
            entries.append((current_key, w_m.group(1)))
            current_key = None
    if not entries:
        raise ValueError(f"Failed to parse multi_run weights from {plan_path}")
    return entries


def resolve_ckpt(path: str) -> Path:
    ckpt = Path(path)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    if ckpt.suffix == ".pt":
        npz = ckpt.with_suffix(".npz")
        if npz.is_file():
            ckpt = npz
    if ckpt.suffix != ".npz":
        raise ValueError(f"Expected SoftPQ .npz, got: {ckpt}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def build_single_cmd(
    *,
    task: str,
    ckpt: Path,
    quality: str | None,
    cuda: bool,
    real: bool,
    output_dir: str,
) -> list[str]:
    plan = PLAN_MAP[task]
    backbone = PROJECT_ROOT / BACKBONE_CKPT
    tag = ckpt.stem
    if quality is not None:
        result_subdir = Path(output_dir) / "SoftPQ" / "dinov3-vitl16-slot24" / task / f"q{quality}"
    else:
        result_subdir = Path(output_dir) / "SoftPQ" / "dinov3-vitl16-slot24" / task / tag

    return [
        sys.executable, "-m", "cofai.engine.run_eval", plan,
        f"args.cuda={str(cuda).lower()}",
        f"args.real={str(real).lower()}",
        "args.multi_run=false",
        f"model.dino_backbone.ckpt_path={backbone}",
        f"model.dino_codec.orfc_weights_path={ckpt}",
        f"args.result_subdir={result_subdir}",
    ]


def run_job(
    *,
    task: str,
    ckpt: Path,
    quality: str | None,
    gpu: str,
    cuda: bool,
    real: bool,
    output_dir: str,
    dry_run: bool,
) -> tuple[str, int]:
    cmd = build_single_cmd(
        task=task, ckpt=ckpt, quality=quality,
        cuda=cuda, real=real, output_dir=output_dir,
    )
    tag = f"{task}:q{quality}" if quality is not None else f"{task}:{ckpt.stem}"
    if dry_run:
        print(f"[dry-run][gpu={gpu}] {tag}")
        print("  " + " \\\n    ".join(cmd))
        return tag, 0

    env = engine_env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[launch][gpu={gpu}] {tag} → {ckpt.name}")
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode
    status = "done" if rc == 0 else "FAIL"
    print(f"[{status}][gpu={gpu}] {tag}  (rc={rc})")
    return tag, rc


def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 SoftPQ online CTC eval (Engine)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task", choices=["semseg", "depth", "both"], default="both",
        help="Task(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="",
        help="Single SoftPQ .npz (ignored when --multi-run)",
    )
    parser.add_argument(
        "--multi-run", action="store_true",
        help="Sweep all multi_run entries in the SoftPQ plan YAML(s)",
    )
    parser.add_argument(
        "--gpus", type=str, default="0",
        help="Comma-separated GPU ids for parallel jobs (default: 0)",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--real", action="store_true", default=True)
    parser.add_argument("--no-real", dest="real", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    tasks = ["semseg", "depth"] if args.task == "both" else [args.task]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        parser.error("--gpus is empty")

    jobs: list[dict] = []
    if args.multi_run:
        for task in tasks:
            plan_path = PROJECT_ROOT / PLAN_MAP[task]
            for quality, rel in parse_multi_run_weights(plan_path):
                ckpt = resolve_ckpt(rel)
                jobs.append({"task": task, "ckpt": ckpt, "quality": quality})
    else:
        if not args.ckpt_path:
            # Default single-run: plan default K256e32 if present
            default = RELEASE_DIR / "blk23_K256_e32.npz"
            if not default.is_file():
                parser.error("Provide --ckpt_path or --multi-run")
            ckpt = default
        else:
            ckpt = resolve_ckpt(args.ckpt_path)
        for task in tasks:
            jobs.append({"task": task, "ckpt": ckpt, "quality": None})

    print(
        f"[ORFC-2446 DINOv3] jobs={len(jobs)}  tasks={tasks}  "
        f"gpus={gpus}  multi_run={args.multi_run}"
    )

    fail = 0
    gpu_pool: Queue[str] = Queue()
    for g in gpus:
        gpu_pool.put(g)

    def _run_with_gpu(job: dict) -> tuple[str, int]:
        gpu = gpu_pool.get()
        try:
            return run_job(
                task=job["task"],
                ckpt=job["ckpt"],
                quality=job["quality"],
                gpu=gpu,
                cuda=args.cuda,
                real=args.real,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
        finally:
            gpu_pool.put(gpu)

    # One concurrent job per GPU; free GPUs are reused as jobs finish.
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(_run_with_gpu, job) for job in jobs]
        for fut in as_completed(futs):
            _tag, rc = fut.result()
            if rc != 0:
                fail += 1

    print(f"[ORFC-2446 DINOv3] Complete. fail={fail}/{len(jobs)}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()

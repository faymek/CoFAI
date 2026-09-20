#!/usr/bin/env python3
"""Multi-GPU CTC eval: one cofai-eval job per (plan, quality).

Used by ``scripts/run_eval_ctc.sh``. Resolves Hydra plans under
``examples/pqfc/plan/dinov3`` and writes ``logs/pqfc/dinov3/<plan>/q<quality>/``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

SOURCE_ROOT = Path(os.environ.get("SOURCE_ROOT", Path(__file__).resolve().parents[3]))
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SOURCE_ROOT))
PY = os.environ.get("PYTHON", sys.executable)
PLAN_DIR = SOURCE_ROOT / "examples/pqfc/plan/dinov3"
OUT = Path(os.environ.get("OUTPUT_DIR", str(SOURCE_ROOT / "logs/pqfc/dinov3")))

NPZ = {
    "1": "blk23_K16_e32.npz",
    "2": "blk23_K256_e32.npz",
    "3": "blk23_K1024_e32.npz",
    "4": "blk23_K512_e16.npz",
    "5": "blk23_K1024_e16.npz",
    "6": "blk23_K64_e8.npz",
    "7": "blk23_K256_e8.npz",
    "8": "blk23_K512_e8.npz",
}
NOR_QUALITIES = tuple(str(q) for q in range(1, 8))
TRANSFORM_QUALITIES = tuple(NPZ)


def _qualities(kind: str) -> tuple[str, ...]:
    return NOR_QUALITIES if kind == "noR" else TRANSFORM_QUALITIES


def _npz_name(kind: str, quality: str) -> str:
    name = NPZ[quality]
    if kind == "noR":
        return name.replace(".npz", "_no_transform.npz")
    return name


def _weight_dir(kind: str) -> Path:
    if kind == "noR":
        candidates = (
            SOURCE_ROOT / "weights/pqfc/dinov3_vitl16_noR",
            PROJECT_ROOT / "weights/pqfc/dinov3_vitl16_noR",
        )
        for path in candidates:
            if path.is_dir():
                return path
        return candidates[0]
    return PROJECT_ROOT / "weights/orfc_2446/dinov3_vitl16_ori"


def _parse_gpus(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip() != ""]


def _select_plans(variant: str, task: str) -> list[Path]:
    if task == "semseg":
        glob = "*__semseg.yaml"
    elif task == "depth":
        glob = "*__depth.yaml"
    else:
        glob = "*.yaml"
    plans: list[Path] = []
    for path in sorted(PLAN_DIR.glob(glob)):
        name = path.name
        is_nor = "__PQFC-noR__" in name
        is_tf = "__PQFC__" in name and not is_nor
        if variant == "transform" and is_tf:
            plans.append(path)
        elif variant == "noR" and is_nor:
            plans.append(path)
        elif variant == "both" and (is_tf or is_nor):
            plans.append(path)
    return plans


def run_one(gpu: int, plan: Path, q: str, weight_dir: Path) -> str:
    plan_name = plan.stem
    dest = OUT / plan_name / f"q{q}"
    log = OUT / "runner_logs" / f"{plan_name}_q{q}.log"
    dest.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    kind = "noR" if "__PQFC-noR__" in plan.name else "transform"
    npz = weight_dir / _npz_name(kind, q)
    if not npz.is_file():
        raise FileNotFoundError(f"missing artifact {npz}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PROJECT_ROOT"] = str(PROJECT_ROOT)
    cmd = [
        PY,
        "-m",
        "cofai.engine.run_eval",
        str(plan),
        "args.cuda=true",
        "args.real=true",
        "args.multi_run=false",
        f"args.quality={q}",
        f"model.dino_codec.orfc_weights_path={npz}",
        f"args.output_dir={dest}",
    ]
    t0 = time.time()
    with log.open("w") as f:
        f.write(f"START gpu={gpu} {plan_name} q{q} {time.strftime('%F %T')}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(dest), env=env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\nEND rc={proc.returncode} elapsed={time.time() - t0:.1f}s\n")
    nested = dest / plan_name / "result.json"
    if nested.exists():
        nested.replace(dest / "result.json")
        cfg = dest / plan_name / "config.yaml"
        if cfg.exists():
            cfg.replace(dest / "config.yaml")
        try:
            (dest / plan_name).rmdir()
        except OSError:
            pass
    result_path = dest / "result.json"
    if result_path.exists():
        data = json.loads(result_path.read_text())
        data["quality"] = q
        result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    label = f"{plan_name} q{q}"
    status = "OK" if proc.returncode == 0 else "FAIL"
    print(f"[{time.strftime('%T')}] {status} gpu={gpu} {label} rc={proc.returncode}", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed rc={proc.returncode}; see {log}")
    return label


def aggregate() -> None:
    for plan_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and p.name != "runner_logs"):
        qualities, per = [], []
        desc = ""
        kind = "noR" if "__PQFC-noR__" in plan_dir.name else "transform"
        for q in _qualities(kind):
            p = plan_dir / f"q{q}" / "result.json"
            if not p.exists():
                print(f"MISSING {p}")
                continue
            data = json.loads(p.read_text())
            qualities.append(q)
            per.append(data.get("results") or {})
            desc = data.get("description") or desc
        if not per:
            continue
        keys = sorted({k for r in per for k in r})
        results = {k: [r.get(k) for r in per] for k in keys}
        summary = {
            "name": plan_dir.name,
            "description": desc,
            "qualities": qualities,
            "results": results,
        }
        path = plan_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Wrote {path}")
        print(json.dumps(results, indent=2, ensure_ascii=False))


def main() -> int:
    variant = os.environ.get("VARIANT", "transform")
    task = os.environ.get("TASK", "both")
    gpu_ids = os.environ.get("GPU_IDS", os.environ.get("GPUS", "0,1"))
    gpus = _parse_gpus(gpu_ids)
    plans = _select_plans(variant, task)
    if not plans:
        print(f"ERROR: no plans for VARIANT={variant} TASK={task}", file=sys.stderr)
        return 1
    jobs: list[tuple[Path, str, Path]] = []
    skipped: list[str] = []
    plan_kinds = []
    if variant == "both":
        plan_kinds = [
            (plan, "noR" if "__PQFC-noR__" in plan.name else "transform")
            for plan in plans
        ]
    else:
        plan_kinds = [(plan, variant) for plan in plans]
    for plan, kind in plan_kinds:
        wdir = _weight_dir(kind)
        for q in _qualities(kind):
            npz = wdir / _npz_name(kind, q)
            label = f"{plan.stem} q{q}"
            if not npz.is_file():
                skipped.append(f"{label} ({npz})")
                continue
            jobs.append((plan, q, wdir))
    if skipped:
        print("SKIP missing artifacts:", flush=True)
        for item in skipped:
            print(f"  {item}", flush=True)
    if not jobs:
        print("ERROR: no eval jobs after resolving artifacts", file=sys.stderr)
        return 1

    print(f"Queued {len(jobs)} jobs on GPUs {gpus}", flush=True)
    print(f"SOURCE_ROOT={SOURCE_ROOT}", flush=True)
    print(f"PROJECT_ROOT={PROJECT_ROOT}", flush=True)
    print(f"Output {OUT}", flush=True)

    gpu_slots = list(gpus)
    lock = Lock()
    failed = 0

    def wrapped(job):
        plan, q, wdir = job
        with lock:
            gpu = gpu_slots.pop(0)
        try:
            print(f"[{time.strftime('%T')}] LAUNCH gpu={gpu} {plan.stem} q{q}", flush=True)
            return run_one(gpu, plan, q, wdir)
        finally:
            with lock:
                gpu_slots.append(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futs = [ex.submit(wrapped, job) for job in jobs]
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                failed += 1
                print(f"ERROR {e}", file=sys.stderr, flush=True)
    aggregate()
    print(f"Completed failed={failed}/{len(jobs)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Offline VTM pipeline for DINOv3 ViT-L/16 slot24 (semseg + depth)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure CoFAI root and examples/vtm are importable when run as a script.
_COFAI_ROOT = Path(__file__).resolve().parents[2]
_VTM_DIR = Path(__file__).resolve().parent
for _p in (_COFAI_ROOT, _VTM_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lib.dataset_utils import (  # noqa: E402
    build_backbone,
    build_dataset,
    build_head,
    build_meter,
    load_config,
    load_subset,
    resolve_project_root,
    sample_stem,
    task_feat_dir,
)
from lib.vtm_codec import vtm_worker  # noqa: E402


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


def _build_stem_index(dataset, task: str, cfg: dict) -> dict[str, int]:
    """One-pass stem -> dataset index map for O(1) GT lookup in replay."""
    nyu_root = cfg["datasets"]["depth"]["root"] if task == "depth" else None
    index: dict[str, int] = {}
    for idx in tqdm(range(len(dataset)), desc=f"index-{task}", leave=False):
        sample = dataset[idx]
        stem = sample_stem(sample, task, nyu_root=nyu_root)
        if stem in index:
            _fail(f"Duplicate dataset stem={stem} at idx={idx} and idx={index[stem]}")
        index[stem] = idx
    return index


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

    nyu_root = cfg["datasets"]["depth"]["root"] if task == "depth" else None
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
        h_p, w_p = img.shape[2] // cfg["patch_size"], img.shape[3] // cfg["patch_size"]

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


def cmd_vtm(args) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    task = args.task
    subset = load_subset(args.subset)
    qps = args.qps or cfg["default_qps"]
    workers = args.workers or cfg["default_workers"]

    feat_dir = task_feat_dir(cfg, task)
    token_dir = feat_dir / "tokens"
    tmp_dir = feat_dir / "_vtm_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for tmp_file in glob.glob(str(feat_dir / "decoded" / "**" / "*.tmp.npy"), recursive=True):
        os.remove(tmp_file)

    token_files = sorted(token_dir.glob("*.npy"))
    if subset is not None:
        token_files = [p for p in token_files if p.stem in subset]

    if not token_files:
        _fail(f"No token files under {token_dir}")

    vtm_enc = cfg["vtm"]["encoder_path"]
    vtm_dec = cfg["vtm"]["decoder_path"]
    vtm_cfg = cfg["vtm"]["cfg_path"]
    for p, name in [(vtm_enc, "encoder"), (vtm_dec, "decoder"), (vtm_cfg, "cfg")]:
        if not Path(p).is_file():
            _fail(f"VTM {name} not found: {p}")

    print(f"[vtm] task={task} files={len(token_files)} qps={qps} workers={workers}")

    for qp in qps:
        out_dir = feat_dir / "decoded" / f"qp{qp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for p in token_files:
            out_path = out_dir / f"{p.stem}.npy"
            if out_path.exists():
                continue
            tasks.append(
                {
                    "token_path": str(p),
                    "out_path": str(out_path),
                    "qp": int(qp),
                    "tmp_dir": str(tmp_dir),
                    "vtm_encoder": vtm_enc,
                    "vtm_decoder": vtm_dec,
                    "vtm_cfg": vtm_cfg,
                    "bit_depth": int(cfg["bit_depth"]),
                }
            )

        done = len(token_files) - len(tasks)
        if not tasks:
            print(f"  QP {qp}: all {len(token_files)} done")
            continue

        print(f"  QP {qp}: {done} done, {len(tasks)} remaining")
        stats = []

        if workers <= 1:
            for t in tqdm(tasks, desc=f"QP{qp}"):
                stats.append(vtm_worker(t))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(vtm_worker, t): t for t in tasks}
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"QP{qp}"):
                    stats.append(fut.result())

        errors = [s for s in stats if "error" in s]
        if errors:
            for e in errors[:5]:
                print(f"    FAIL stem={e['stem']} qp={e['qp']}: {e['error']}", file=sys.stderr)
            _fail(f"VTM failed for {len(errors)} file(s) at QP={qp}")

        success = [s for s in stats if "error" not in s]
        avg_bpfp = float(np.mean([s["bpfp"] for s in success])) if success else 0.0
        total_bytes = sum(s["bs_bytes"] for s in success)
        stats_path = out_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "qp": qp,
                    "task": task,
                    "avg_bpfp": avg_bpfp,
                    "n_files": len(success),
                    "total_bytes": total_bytes,
                },
                f,
                indent=2,
            )
        print(f"  QP {qp}: avg BPFP={avg_bpfp:.4f}, total={total_bytes / 1e6:.1f}MB")


@torch.inference_mode()
def _replay_one(
    tokens: np.ndarray,
    meta: dict,
    *,
    task: str,
    backbone,
    head,
    device: torch.device,
    patch_size: int,
    sample: dict | None = None,
):
    h_hat = torch.from_numpy(tokens).unsqueeze(0).to(device=device, dtype=torch.float32)
    h_p, w_p = int(meta["token_hw"][0]), int(meta["token_hw"][1])
    token_res = (h_p, w_p)

    if task == "semseg":
        feat = backbone.decode_seg(h_hat, token_res)
        logits = head.predict(feat, scale=int(patch_size))
        pred = torch.argmax(logits, dim=1)
        gt_t = sample["semseg"].squeeze().long()
        return pred, gt_t

    feat = backbone.decode_depth(h_hat, token_res)
    img_h, img_w = int(meta["img_hw"][0]), int(meta["img_hw"][1])
    depth = head.predict(feat, size=(img_h, img_w))
    return depth, sample["depth"] if sample is not None else None


def cmd_replay(args) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    task = args.task
    mode = args.mode
    subset = load_subset(args.subset)
    dataset = build_dataset(task, cfg)
    backbone = build_backbone(cfg, device)
    head = build_head(task, cfg).to(device).eval()
    feat_dir = task_feat_dir(cfg, task)
    token_dir = feat_dir / "tokens"
    meta_dir = feat_dir / "meta"
    stem_index = _build_stem_index(dataset, task, cfg)

    qps = [None] if mode == "bypass" else (args.qps or cfg["default_qps"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    for qp in qps:
        meter = build_meter(task, cfg)
        if mode == "bypass":
            feat_src = token_dir
            tag = f"{task}_bypass"
        else:
            feat_src = feat_dir / "decoded" / f"qp{qp}"
            tag = f"{task}_vtm_qp{qp}"

        stems = load_subset(args.subset)
        if stems is None:
            stems = {p.stem for p in feat_src.glob("*.npy")}

        for stem in tqdm(sorted(stems), desc=f"replay-{tag}"):
            tok_path = feat_src / f"{stem}.npy"
            meta_path = meta_dir / f"{stem}.npz"
            if not tok_path.is_file() or not meta_path.is_file():
                _fail(f"Missing token/meta for stem={stem}")

            tokens = np.load(tok_path)
            meta = dict(np.load(meta_path, allow_pickle=True))
            idx = stem_index.get(stem)
            if idx is None:
                _fail(f"Could not find dataset sample for stem={stem}")
            sample = dataset[idx]

            pred, gt = _replay_one(
                tokens,
                meta,
                task=task,
                backbone=backbone,
                head=head,
                device=device,
                patch_size=cfg["patch_size"],
                sample=sample,
            )
            if task == "semseg":
                pred_np = pred.squeeze(0).cpu().numpy()
                gt_np = gt.squeeze().cpu().numpy()
                meter.update(pred_np, gt_np)
            else:
                meter.update(pred, gt)

        metrics = meter.compute()
        out = {
            "task": task,
            "mode": mode,
            "qp": qp,
            "n_samples": len(stems),
            "metrics": metrics,
        }
        out_path = results_dir / f"{tag}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[replay] {tag}: {metrics} -> {out_path}")


def cmd_verify(args) -> None:
    cfg = load_config(Path(args.config) if args.config else None)
    task = args.task
    subset = load_subset(args.subset)
    n_samples = len(subset) if subset else args.max_samples

    results_dir = Path(cfg["paths"]["results_dir"])
    offline_path = results_dir / f"{task}_bypass.json"
    if not offline_path.is_file():
        _fail(f"Run replay --mode bypass first: missing {offline_path}")

    with open(offline_path) as f:
        offline = json.load(f)
    metric_key = cfg["datasets"][task]["metric_key"]
    offline_metric = cfg["datasets"][task].get("offline_metric_key")
    if offline_metric is None:
        offline_metric = "mIoU" if task == "semseg" else "rmse"
    offline_val = float(offline["metrics"][offline_metric])

    root = resolve_project_root()
    plan = cfg["datasets"][task]["bypass_plan"]
    out_dir = results_dir / f"cofai_eval_{task}_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "poetry",
        "run",
        "cofai-eval",
        plan,
        f"args.max_samples={n_samples}",
        "args.cuda=true",
        f"args.output_dir={out_dir}",
    ]
    print(f"[verify] running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PROJECT_ROOT"] = str(root)
    proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        _fail(f"cofai-eval failed with code {proc.returncode}")

    result_files = list(out_dir.rglob("result.json"))
    if not result_files:
        _fail(f"No result.json under {out_dir}")
    with open(result_files[0]) as f:
        cofai_res = json.load(f)

    cofai_val = None
    results_block = cofai_res.get("results", {})
    if isinstance(results_block, dict) and metric_key in results_block:
        cofai_val = float(results_block[metric_key])
    if cofai_val is None:
        # Flat search
        def _find(d):
            if isinstance(d, dict):
                if metric_key in d and isinstance(d[metric_key], (int, float)):
                    return float(d[metric_key])
                for v in d.values():
                    r = _find(v)
                    if r is not None:
                        return r
            elif isinstance(d, list):
                for v in d:
                    r = _find(v)
                    if r is not None:
                        return r
            return None

        cofai_val = _find(cofai_res)

    if cofai_val is None:
        _fail(f"Could not find {metric_key} in cofai-eval result: {result_files[0]}")

    tol = float(cfg["verify_tolerance"].get(metric_key, 0.001))
    diff = abs(offline_val - cofai_val)
    print(f"[verify] {metric_key}: offline={offline_val:.6f} cofai-eval={cofai_val:.6f} diff={diff:.6f} tol={tol}")
    if diff > tol:
        _fail(f"Verify failed: |diff|={diff:.6f} > {tol}")
    print("[verify] PASS")


def build_parser():
    ap = argparse.ArgumentParser(description="DINOv3 slot24 offline VTM pipeline")
    ap.add_argument("--config", type=str, default=None, help="Path to dinov3_slot24.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ext = sub.add_parser("extract", help="Extract slot24 tokens")
    p_ext.add_argument("--task", choices=["semseg", "depth"], required=True)
    p_ext.add_argument("--gpu", type=int, default=0)
    p_ext.add_argument("--subset", type=str, default=None)

    p_vtm = sub.add_parser("vtm", help="VTM encode/decode tokens")
    p_vtm.add_argument("--task", choices=["semseg", "depth"], required=True)
    p_vtm.add_argument("--qps", nargs="+", type=int, default=None)
    p_vtm.add_argument("--workers", type=int, default=None)
    p_vtm.add_argument("--subset", type=str, default=None)

    p_rep = sub.add_parser("replay", help="Replay features through decode+head")
    p_rep.add_argument("--task", choices=["semseg", "depth"], required=True)
    p_rep.add_argument("--mode", choices=["bypass", "vtm"], default="bypass")
    p_rep.add_argument("--qps", nargs="+", type=int, default=None)
    p_rep.add_argument("--gpu", type=int, default=0)
    p_rep.add_argument("--subset", type=str, default=None)

    p_ver = sub.add_parser("verify", help="Compare offline bypass vs cofai-eval")
    p_ver.add_argument("--task", choices=["semseg", "depth"], required=True)
    p_ver.add_argument("--subset", type=str, required=True)
    p_ver.add_argument("--max_samples", type=int, default=8)

    return ap


def main():
    args = build_parser().parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "vtm":
        cmd_vtm(args)
    elif args.cmd == "replay":
        cmd_replay(args)
    elif args.cmd == "verify":
        cmd_verify(args)
    else:
        _fail(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

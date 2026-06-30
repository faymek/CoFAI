"""Eval-time inference and default DataUnitCodec test-step helpers.

Entry: `@hydra.main` with ``config_path`` = repo ``conf/`` (standard Hydra CLI,
e.g. ``--config-name=plan/<file>``).

Current limitation: the eval path is only maintained for batch size = 1.
Multi-sample batches are not part of the supported contract.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict

from cofai.engine.builder import build_dataset, build_model, build_tasks
from cofai.engine.config import resolve_plan, validate_plan
from cofai.metrics.utils import DictAverageMeter
from cofai.engine.dataloader import build_dataloader
from cofai.engine.evaluator import MultiTaskEvaluator
from cofai.engine.schema import (
    EvalBatch,
    StepOutput,
    EvalResultPayload,
    _validate_step_output_soft,
)

# Disable Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

_CONF_ROOT = (Path(__file__).resolve().parents[2] / "conf").resolve()


def _bits_from_coded_unit(data: Dict[str, Any]) -> Dict[str, float]:
    if "strings" in data:  # real compression
        return {
            str(name): float(sum(len(s[0]) for s in sub_strings) * 8.0)
            for name, sub_strings in data["strings"].items()
        }
    if "likelihoods" in data:
        return {
            str(name): float((torch.log(likelihoods).sum() / (-math.log(2))).item())
            for name, likelihoods in data["likelihoods"].items()
        }
    if "bits" in data:
        return {str(k): float(v) for k, v in data["bits"].items()}
    raise KeyError("Expected key `strings` or `likelihoods` in out_enc.")


def _bits_from_coded_data(out: Dict[str, Any]) -> Dict[str, float]:
    # Back-compat: adapted CompressAI style.
    if "type" not in out:
        return _bits_from_coded_unit(out)

    out_type = out.get("type")
    if out_type == "unit":
        return _bits_from_coded_unit(out["data"])
    if out_type == "frame":
        flat: Dict[str, float] = {}
        for layer_name, coded_unit in out["data"].items():
            bits_items = _bits_from_coded_unit(coded_unit)
            for k, v in bits_items.items():
                flat[f"{layer_name}.{k}"] = float(v)
        return flat
    if out_type == "frame_wise_video":
        flat = {}
        for frame_name, coded_frame in out["data"].items():
            for layer_name, coded_unit in coded_frame["data"].items():
                bits_items = _bits_from_coded_unit(coded_unit)
                for k, v in bits_items.items():
                    flat[f"{frame_name}.{layer_name}.{k}"] = float(v)
        return flat
    if out_type == "layer_wise_video":
        flat = {}
        for layer_name, coded_frame in out["data"].items():
            for frame_name, coded_unit in coded_frame["data"].items():
                bits_items = _bits_from_coded_unit(coded_unit)
                for k, v in bits_items.items():
                    flat[f"{layer_name}.{frame_name}.{k}"] = float(v)
        return flat
    if out_type == "slide_crops":
        # Sliding-window models emit one coded_unit per crop; aggregate the bits
        # of every crop into a single per-codec total.
        flat = {}
        for coded_unit in out["data"]:
            bits_items = _bits_from_coded_unit(coded_unit)
            for k, v in bits_items.items():
                flat[k] = flat.get(k, 0.0) + float(v)
        return flat
    raise NotImplementedError(f"Unsupported type: {out_type!r}")


@torch.inference_mode()
def inference_model(
    model: Any,
    x: torch.Tensor,
    *,
    qp: Any = 1,
    real: bool = False,
    tasks: List[str] | None = None,
    task_specs: Any | None = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Single inference helper used by unified eval."""

    if tasks is None:
        tasks = []
    codec_kw: Dict[str, Any] = {}
    if task_specs is not None:
        codec_kw["task_specs"] = task_specs

    if real:
        t0 = time.time()
        coded_data = model.compress(x, qp=qp, tasks=tasks)
        enc_time = time.time() - t0

        t1 = time.time()
        task_feats = model.decompress(coded_data, tasks=tasks, **codec_kw)
        dec_time = time.time() - t1

        bits_items = _bits_from_coded_data(coded_data)
        time_items = {"enc_time": float(enc_time), "dec_time": float(dec_time)}
        return time_items, bits_items, task_feats

    t0 = time.time()
    coded_data, task_feats = model.forward_test(x, qp=qp, tasks=tasks, **codec_kw)
    elapsed = time.time() - t0
    bits_items = _bits_from_coded_data(coded_data)
    time_items = {"enc_time": float(elapsed) / 2.0, "dec_time": float(elapsed) / 2.0}
    return time_items, bits_items, task_feats


def _maybe_cpu_squeeze(v: Any) -> Any:
    if not isinstance(v, torch.Tensor):
        return v
    if v.dim() == 3 and v.size(0) == 1:
        v = v.squeeze(0)
    return v.detach().cpu()


def task_decode_pred(raw: Any, task: str) -> Any:
    """Decode per-task raw outputs into pred-ready tensors/arrays."""

    task = str(task)
    output = raw

    if task in {"rec", "rae"}:
        # Align with common GT format (CHW).
        # If model outputs NCHW with N=1, squeeze batch dim to CHW.
        if (
            isinstance(output, torch.Tensor)
            and output.dim() == 4
            and output.size(0) == 1
        ):
            output = output.squeeze(0)
        return output

    if task == "normals":
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0
    elif task in {"semseg", "seg", "human_parts"}:
        if (
            isinstance(output, torch.Tensor)
            and output.dim() == 4
            and output.size(1) > 1
        ):
            output = torch.argmax(output, dim=1)
    elif task == "edge":
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)), dim=3)
    elif task == "sal":
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] * 255
    elif task == "depth":
        # Keep NCHW: the depth meters (e.g. Dinov3DepthEstimationMeter) expect
        # (B, 1, H, W) preds aligned with (B, 1, H, W) GT. Avoid an in-place clamp
        # because task feats may be inference-mode tensors.
        output = output.clamp(min=0.0)
    elif task == "scene":
        _, output = torch.max(output, dim=1)
    elif task == "cls":
        # Top-k class indices per row (see TopKAccuracyMetric).
        if (
            isinstance(output, torch.Tensor)
            and output.dim() == 2
            and output.size(1) > 1
        ):
            k = int(min(5, int(output.size(1))))
            _, top_idx = torch.topk(output, k=k, dim=1)
            output = top_idx.tolist()
    else:
        raise ValueError(f"Unknown task for task_decode_pred: {task!r}")

    return output


def _label_to_kind_from_task_specs(task_specs: Any) -> Dict[str, str]:
    """``task_specs[*].label`` -> ``task_specs[*].kind`` (same field names as YAML)."""
    m: Dict[str, str] = {}
    if not isinstance(task_specs, list):
        return m
    for sp in task_specs:
        if not isinstance(sp, dict):
            continue
        label = sp.get("label")
        kind = sp.get("kind")
        if label and kind:
            m[str(label)] = str(kind)
    return m


def _plan_label_kind_pairs(tasks: List[str], label_to_kind: Dict[str, str]) -> List[Tuple[str, str]]:
    """(plan ``label``, dataset / codec ``kind``) for each task in order."""

    return [(str(t), str(label_to_kind.get(str(t), str(t)))) for t in tasks]


def _build_per_sample_records(
    *,
    samples: List[Dict[str, Any]],
    time_items: Dict[str, float],
    bits_items: Dict[str, float],
    quality: Any,
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        meta = sample["meta"]
        h, w = meta["ori_size"]
        npx = h * w
        bpp_items = {f"bpp_{k}": float(v) / float(npx) for k, v in bits_items.items()}
        bpp_items = {"bpp": float(sum(bpp_items.values())), **bpp_items}
        file = meta.get("img_path") or meta.get("img_name") or f"sample_{i}"
        out.append({"file": file, "quality": quality, **time_items, **bpp_items})
    return out


def eval_step(
    *,
    model: Any,
    batch: EvalBatch,
    step_ctx: Optional[Dict[str, Any]] = {},
) -> StepOutput:
    """Default `test_step` for DataUnitCodec-style models.

    Required runtime context is read from `ctx` (passed by Runner/TestLoop):
    - cfg, tasks, device, task_specs
    """

    if not isinstance(batch, EvalBatch):
        raise TypeError(f"Expected EvalBatch from collate, got {type(batch)!r}")

    cfg = step_ctx["cfg"]
    tasks = list(step_ctx["tasks"])
    device = step_ctx["device"]
    task_specs = step_ctx.get("task_specs")
    label_to_kind = _label_to_kind_from_task_specs(task_specs)

    img = batch.inputs["img"].to(device, non_blocking=True)

    bs = int(img.size(0))
    if bs != 1:
        raise ValueError(
            f"Current eval step only supports batch size = 1, got batch size {bs}."
        )

    time_items, bits_items, task_feats = inference_model(
        model,
        img,
        qp=cfg.args.quality,
        real=cfg.args.real,
        tasks=tasks,
        task_specs=task_specs,
    )

    missing = [t for t in tasks if t not in (task_feats or {})]
    if missing:
        avail = sorted(list((task_feats or {}).keys()))
        raise KeyError(
            "Model did not return outputs for all declared tasks. "
            f"missing={missing!r}, available={avail!r}, task_specs={task_specs!r}"
        )

    per_sample_records = _build_per_sample_records(
        samples=batch.samples,
        time_items=time_items,
        bits_items=bits_items,
        quality=cfg.args.quality,
    )

    if hasattr(model, "get_feature_numel"):
        bpfp = sum(bits_items.values()) / model.get_feature_numel(img)
        for rec in per_sample_records:
            rec["bpfp"] = bpfp

    pairs = _plan_label_kind_pairs(tasks, label_to_kind)
    pred = {
        label: _maybe_cpu_squeeze(task_decode_pred(task_feats[label], kind))
        for label, kind in pairs
    }
    t0 = batch.samples[0]
    gt = {
        label: _maybe_cpu_squeeze(t0[kind]) for label, kind in pairs
    }

    roi = t0["meta"].get("valid_roi")
    if roi:
        top = int(roi["top"])
        left = int(roi["left"])
        rh = int(roi["height"])
        rw = int(roi["width"])

        def _crop_chw(x: Any) -> Any:
            if isinstance(x, torch.Tensor) and x.dim() == 3:
                return x[:, top : top + rh, left : left + rw]
            return x

        for label, kind in pairs:
            if kind == "rec":
                pred[label] = _crop_chw(pred[label])
                gt[label] = _crop_chw(gt[label])

    return StepOutput(
        timing=time_items,
        bits=bits_items,
        pred=pred,
        gt=gt,
        per_sample_records=per_sample_records,
        artifacts={},
    )


def eval_loop(
    *,
    loader: Any,
    model: Any,
    evaluator: Any,
    max_samples: int | None,
    ctx: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_samples = 0
    ctx = dict(ctx or {})
    ds = loader.dataset
    total = int(max_samples) if max_samples and max_samples > 0 else len(ds)
    with tqdm(total=total, desc="Evaluating", unit="sample") as pbar:
        for batch in loader:
            step_out = eval_step(model=model, batch=batch, step_ctx=ctx)
            evaluator.update(step_out, ctx=ctx)
            _validate_step_output_soft(step_out)
            for record in step_out.per_sample_records or []:
                records.append(dict(record))
            bs = int(batch.inputs["img"].size(0))
            seen_samples += bs
            pbar.update(bs)
            if max_samples and max_samples > 0 and seen_samples >= max_samples:
                break
    return records


def write_eval_outputs(
    output_dir: str,
    *,
    name: str,
    description: str,
    results: Dict[str, Any],
    quality: Optional[str] = None,
    records: Optional[List[Dict[str, Any]]] = None,
    cfg: Any | None = None,
) -> str:
    """Write `result.json` and return its path."""

    os.makedirs(output_dir, exist_ok=True)

    payload: EvalResultPayload = {
        "name": name,
        "description": description,
        "results": results,
        "quality": quality,
        "records": records or [],
    }
    canonical_path = os.path.join(output_dir, "result.json")
    with open(canonical_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    if cfg is not None:
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    return canonical_path


def finalize_results(
    *, records: List[Dict[str, Any]], evaluator: MultiTaskEvaluator
) -> Dict[str, Any]:
    """Merge per-sample record means and task metrics into one flat dict for ``result.json``.

    Order: timing / bpp aggregates first, then task-level meter metrics (e.g. ``cls_top-*``).
    """
    # Per-sample dicts from ``eval_loop`` (e.g. timing, bpp); numeric mean over rows.
    record_meter = DictAverageMeter()
    for row in (r for r in (records or []) if isinstance(r, dict)):
        nums = {
            k: v
            for k, v in row.items()
            if k not in ("file", "quality") and not isinstance(v, bool)
        }
        record_meter.update(nums)
    record_summary: Dict[str, Any] = (
        record_meter.average() if record_meter.meter is not None else {}
    )

    # Task-level meter aggregates from the evaluator (e.g. mIoU, accuracy).
    metrics_by_task = evaluator.compute_all_metrics()
    metrics_summary: Dict[str, Any] = {}
    for task_label, task_metrics in metrics_by_task.items():
        if isinstance(task_metrics, dict):
            for metric_name, value in task_metrics.items():
                metrics_summary[f"{task_label}_{metric_name}"] = value
        else:
            metrics_summary[task_label] = task_metrics

    return {**record_summary, **metrics_summary}


def run_eval(
    cfg: Any,
    *,
    write_subdir: Optional[str] = None,
    result_quality: Optional[str] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Unified eval entrypoint (engine-level).

    ``write_subdir``: when set, write ``result.json`` / ``config.yaml`` directly under this
    directory (used by ``multi_run`` per-quality folders). Otherwise use ``logs/<plan.name>/``.
    """

    plan_key, plan_cfg = resolve_plan(cfg)
    validate_plan(plan_cfg, plan_key=plan_key)

    device = torch.device(str(cfg.args.device))

    dataset = build_dataset(plan_cfg=plan_cfg)
    loader = build_dataloader(cfg, dataset)

    tasks, evaluator, task_specs = build_tasks(plan_cfg=plan_cfg)
    model = build_model(
        cfg=cfg,
        plan_cfg=plan_cfg,
        tasks=list(tasks),
        task_specs=task_specs,
        device=device,
    )

    step_ctx = {
        "cfg": cfg,
        "tasks": list(tasks),
        "device": device,
        "task_specs": task_specs,
    }

    ms = cfg.args.max_samples
    max_samples_i = int(ms) if ms is not None else None

    records = eval_loop(
        loader=loader,
        model=model,
        evaluator=evaluator,
        max_samples=max_samples_i,
        ctx=step_ctx,
    )
    summary = finalize_results(records=records, evaluator=evaluator)

    run_name = str(plan_cfg.name)
    description = str(plan_cfg.description)
    output_dir = (
        write_subdir
        if write_subdir is not None
        else os.path.join(str(cfg.args.output_dir or "").strip() or "logs", run_name)
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        canonical_path = os.path.join(output_dir, "result.json")
        payload: EvalResultPayload = {
            "name": run_name,
            "description": description,
            "results": summary,
            "quality": result_quality,
            "records": records,
        }
        with open(canonical_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
        print(f"Results saved to: {canonical_path}")
        print("results:\n" + json.dumps(summary, indent=2, ensure_ascii=False))

    return summary, records


def multi_run(cfg: Any):
    """Each ``cfg.multi_run`` key → one ``run_eval``; aggregate ``summary.json``."""
    from omegaconf import OmegaConf, open_dict

    if not cfg.args.multi_run or not cfg.multi_run:
        return run_eval(cfg)

    base_cfg = cfg.copy()
    with open_dict(base_cfg):
        mr = base_cfg.pop("multi_run", None)
        with open_dict(base_cfg.args):
            base_cfg.args.multi_run = False
    if not mr:
        return run_eval(cfg)

    _, plan = resolve_plan(cfg)
    base_name = str(plan.name)
    base_desc = str(plan.description)
    group_dir = os.path.join(str(cfg.args.output_dir or "").strip() or "logs", base_name)
    os.makedirs(group_dir, exist_ok=True)

    qualities: list[str] = []
    per_quality_results: list[dict[str, Any]] = []

    for quality, patch in mr.items():
        q = str(quality)
        this_cfg = base_cfg.copy()
        if patch is not None:
            this_cfg = OmegaConf.merge(this_cfg, patch)
        with open_dict(this_cfg):
            with open_dict(this_cfg.args):
                # Keep numeric YAML keys (e.g. ``multi_run: 1:``) as int for codec ``qp``; ``q`` stays str for paths/JSON.
                this_cfg.args.quality = quality
        subdir = os.path.join(group_dir, f"q{q}")
        summary, _records = run_eval(
            this_cfg,
            write_subdir=subdir,
            result_quality=q,
        )
        qualities.append(q)
        per_quality_results.append(dict(summary))

    # Build summary: metrics as lists aligned with `qualities`.
    keys = sorted({k for r in per_quality_results for k in r.keys()})
    summary_results: dict[str, list[Any]] = {k: [] for k in keys}
    for r in per_quality_results:
        for k in keys:
            summary_results[k].append(r.get(k))

    summary = {
        "name": base_name,
        "description": base_desc,
        "qualities": qualities,
        "results": summary_results,
    }
    summary_path = os.path.join(group_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")
    print("results:\n" + json.dumps(summary["results"], indent=2, ensure_ascii=False))

    return summary_results, per_quality_results


@hydra.main(
    version_base=None,
    config_path=str(_CONF_ROOT),
    config_name="plan/ade20k-val--Bypass-small-last4",
)
def _hydra_cli(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        cfg.PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
    cfg.args.device = "cuda" if cfg.args.cuda else "cpu"
    OmegaConf.set_struct(cfg, True)
    multi_run(cfg)


def main() -> None:
    """Package entrypoint: `python -m cofai.engine.run_eval`."""

    load_dotenv(override=True, encoding="utf-8")
    argv = sys.argv
    # Sugar: a positional `*.yaml` plan path → Hydra --config-dir/--config-name,
    # so any plan (under conf/ or elsewhere) runs as `cofai-eval path/to/plan.yaml`.
    # Overrides like `args.x=y` contain '=', so they are left untouched.
    for i, arg in enumerate(argv[1:], start=1):
        if arg.endswith(".yaml") and "=" not in arg:
            plan = Path(arg)
            argv[i : i + 1] = [
                f"--config-dir={plan.parent.resolve()}",
                f"--config-name={plan.stem}",
            ]
            break
    argv += ["hydra.run.dir=.", "hydra.output_subdir=null"]
    _hydra_cli()


if __name__ == "__main__":
    main()

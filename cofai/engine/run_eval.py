"""Eval-time inference and default DataUnitCodec test-step helpers.

Entry: `@hydra.main` with ``config_path`` = repo ``conf/`` (standard Hydra CLI,
e.g. ``--config-name=plan/<file>``).

Current limitation: the eval path is only maintained for batch size = 1.
Multi-sample batches are not part of the supported contract.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm.auto import tqdm
from omegaconf import DictConfig, OmegaConf, open_dict

from cofai.engine.builder import build_dataset, build_model, build_tasks
from cofai.engine.bitrate import bits_from_coded_data
from cofai.engine.config import resolve_plan, validate_plan
from cofai.metrics.utils import DictAverageMeter
from cofai.engine.dataloader import build_dataloader
from cofai.engine.evaluator import MultiTaskEvaluator
from cofai.engine.runtime import suppress_unnecessary_runtime_output
from cofai.engine.schema import (
    EvalBatch,
    StepOutput,
    EvalResultPayload,
    _validate_step_output_soft,
)

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

_CONF_ROOT = (Path(__file__).resolve().parents[2] / "conf").resolve()


@torch.inference_mode()
def inference_model(
    model: Any,
    x: Any,
    *,
    qp: Any = 1,
    real: bool = False,
    tasks: List[str] | None = None,
    task_specs: Any | None = None,
    profile: bool = False,
    task_data: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Single inference helper used by unified eval.

    When ``profile`` is True, the codec-only encode/decode times recorded by the
    model (``model._codec_time``) are merged into the returned timing dict.
    """

    if tasks is None:
        tasks = []
    task_data = task_data or {}
    codec_kw: Dict[str, Any] = {}
    if task_specs is not None:
        codec_kw["task_specs"] = task_specs

    if real:
        t0 = time.time()
        coded_data = model.compress(x, qp=qp, tasks=tasks)
        enc_time = time.time() - t0

        t1 = time.time()
        task_feats = model.decompress(
            coded_data,
            tasks=tasks,
            task_data=task_data,
            **codec_kw,
        )
        dec_time = time.time() - t1

        bits_items = bits_from_coded_data(coded_data)
        time_items = {"total_enc_time": float(enc_time), "total_dec_time": float(dec_time)}
        if profile:
            time_items.update(getattr(model, "_codec_time", {}) or {})
        return time_items, bits_items, task_feats

    t0 = time.time()
    coded_data, task_feats = model.forward_test(
        x,
        qp=qp,
        tasks=tasks,
        task_data=task_data,
        **codec_kw,
    )
    elapsed = time.time() - t0
    bits_items = bits_from_coded_data(coded_data)
    time_items = {"total_enc_time": float(elapsed) / 2.0, "total_dec_time": float(elapsed) / 2.0}
    if profile:
        time_items.update(getattr(model, "_codec_time", {}) or {})
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

    if not isinstance(output, torch.Tensor):
        return output

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


def _sample_num_pixels(sample: Dict[str, Any]) -> int:
    meta = sample.get("meta")
    if not isinstance(meta, dict) or "ori_size" not in meta:
        raise KeyError("samples[i]['meta']['ori_size'] is required for eval records")
    h, w = meta["ori_size"]
    return max(1, int(h) * int(w))


def _build_task_pair(
    *,
    sample: Dict[str, Any],
    kind: str,
    raw_feat: Any,
) -> tuple[Any, Any]:
    """Decode one task output and pair it with sample task data for evaluator.update()."""

    if kind not in sample:
        raise KeyError(f"Eval sample is missing task data for kind {kind!r}")

    pred = _maybe_cpu_squeeze(task_decode_pred(raw_feat, kind))
    gt = _maybe_cpu_squeeze(sample[kind])

    if kind != "rec":
        return pred, gt

    roi = sample.get("meta", {}).get("valid_roi")
    if not roi:
        return pred, gt
    top = int(roi["top"])
    left = int(roi["left"])
    rh = int(roi["height"])
    rw = int(roi["width"])

    def _crop_chw(x: Any) -> Any:
        if isinstance(x, torch.Tensor) and x.dim() == 3:
            return x[:, top : top + rh, left : left + rw]
        return x

    return _crop_chw(pred), _crop_chw(gt)


def _build_per_sample_records(
    *,
    samples: List[Dict[str, Any]],
    time_items: Dict[str, float],
    bits_items: Dict[str, float],
    quality: Any,
) -> List[Dict[str, Any]]:

    out: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        npx = _sample_num_pixels(sample)
        bpp_items = {f"bpp_{k}": float(v) / float(npx) for k, v in bits_items.items()}
        bpp_items = {"bpp": float(sum(bpp_items.values())), **bpp_items}
        meta = sample["meta"]
        file = meta.get("img_path") or meta.get("img_name") or f"sample_{i}"
        out.append({"file": file, "quality": quality, **time_items, **bpp_items})
    return out


def eval_step(
    *,
    model: Any,
    batch: EvalBatch,
    step_ctx: Optional[Dict[str, Any]] = None,
) -> StepOutput:
    """Default `test_step` for DataUnitCodec-style models.

    Required runtime context is read from `ctx` (passed by Runner/TestLoop):
    - cfg, tasks, device, task_specs
    """

    if not isinstance(batch, EvalBatch):
        raise TypeError(f"Expected EvalBatch from collate, got {type(batch)!r}")

    step_ctx = step_ctx or {}
    cfg = step_ctx["cfg"]
    tasks = list(step_ctx["tasks"])
    device = step_ctx["device"]
    task_specs = step_ctx.get("task_specs")
    label_to_kind = _label_to_kind_from_task_specs(task_specs)
    pairs = [(str(task), str(label_to_kind.get(str(task), str(task)))) for task in tasks]

    bs = len(batch.samples)
    if bs != 1:
        raise ValueError(
            f"Current eval step only supports batch size = 1, got batch size {bs}."
        )

    if "img" not in batch.inputs:
        raise KeyError("EvalBatch.inputs must contain the shared `img` field.")
    img = batch.inputs["img"].to(device, non_blocking=True)
    sample0 = batch.samples[0]
    kinds = dict.fromkeys(kind for _label, kind in pairs)
    missing_task_data = [kind for kind in kinds if kind not in sample0]
    if missing_task_data:
        raise KeyError(f"Eval sample is missing task data: {missing_task_data!r}")
    task_data = {kind: sample0[kind] for kind in kinds}
    time_items, bits_items, task_feats = inference_model(
        model,
        img,
        qp=cfg.args.quality,
        real=cfg.args.real,
        tasks=tasks,
        task_specs=task_specs,
        profile=bool(getattr(cfg.args, "profile", False)),
        task_data=task_data,
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

    record0 = per_sample_records[0]
    pred: Dict[str, Any] = {}
    gt: Dict[str, Any] = {}
    for label, kind in pairs:
        pred[label], gt[label] = _build_task_pair(
            sample=sample0,
            kind=kind,
            raw_feat=task_feats[label],
        )

    if "vqa" in task_data:
        record0.update(task_data["vqa"])
        record0["prediction"] = pred["vqa"]

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
            bs = len(batch.samples)
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


def measure_codec_complexity(
    *, loader: Any, model: Any, device: Any, qp: Any, max_samples: int | None
) -> Dict[str, Any]:
    """Average latent-codec params/FLOPs over the dataset (profile pass).

    Runs before the eval loop so the codec is measured on clean (non
    inference-mode) state. ``codec_params`` is constant; the ``*_flops`` fields
    are averaged over samples (they vary with input resolution).
    """
    params: Optional[int] = None
    flops: Dict[str, List[int]] = {}
    seen = 0
    total = int(max_samples) if max_samples and max_samples > 0 else len(loader.dataset)
    with tqdm(total=total, desc="Profiling", unit="sample") as pbar:
        for batch in loader:
            img = batch.inputs["img"].to(device, non_blocking=True)
            comp = model.codec_complexity(img, qp=qp)
            params = comp.get("codec_params", params)
            for k, v in comp.items():
                if k.endswith("_flops"):
                    flops.setdefault(k, []).append(int(v))
            seen += 1
            pbar.update(1)
            if max_samples and max_samples > 0 and seen >= max_samples:
                break

    out: Dict[str, Any] = {}
    if params is not None:
        out["codec_params"] = params
    for k, vals in flops.items():
        out[k] = sum(vals) / len(vals)
    return out


def _group_codec_keys(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Reorder a summary so all ``codec_*`` fields are grouped at the end."""
    codec_order = [
        "codec_params",
        "codec_enc_flops",
        "codec_dec_flops",
        "codec_enc_time",
        "codec_dec_time",
    ]
    non_codec = {k: v for k, v in summary.items() if not k.startswith("codec_")}
    codec = {k: v for k, v in summary.items() if k.startswith("codec_")}
    ordered: Dict[str, Any] = dict(non_codec)
    for k in codec_order:
        if k in codec:
            ordered[k] = codec.pop(k)
    ordered.update(codec)
    return ordered


def _print_complexity_summary(summary: Dict[str, Any]) -> None:
    if "codec_params" in summary:
        if summary["codec_params"] == 0:
            print("[complexity] codec params: 0 (bypass)")
        else:
            print(f"[complexity] codec params: {summary['codec_params'] / 1e6:.2f} M")

    bypass = summary.get("codec_params") == 0
    for key, label in (("codec_enc_flops", "enc"), ("codec_dec_flops", "dec")):
        if key not in summary:
            continue
        if summary[key] == 0 and bypass:
            print(f"[complexity] codec {label} FLOPs: 0 (bypass)")
        elif summary[key] == 0:
            print(f"[complexity] codec {label} FLOPs: 0")
        else:
            print(f"[complexity] codec {label} FLOPs: {summary[key] / 1e9:.2f} G")


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

    complexity: Dict[str, Any] = {}
    profile = bool(OmegaConf.select(cfg, "args.profile", default=False))
    if profile and hasattr(model, "codec_complexity"):
        complexity = measure_codec_complexity(
            loader=loader,
            model=model,
            device=device,
            qp=cfg.args.quality,
            max_samples=max_samples_i,
        )

    records = eval_loop(
        loader=loader,
        model=model,
        evaluator=evaluator,
        max_samples=max_samples_i,
        ctx=step_ctx,
    )

    summary = finalize_results(records=records, evaluator=evaluator)
    summary.update(complexity)
    summary = _group_codec_keys(summary)

    _print_complexity_summary(summary)

    run_name = str(plan_cfg.name)
    description = str(plan_cfg.description)
    output_dir = (
        write_subdir
        if write_subdir is not None
        else os.path.join(str(cfg.args.output_dir or "").strip() or "logs", run_name)
    )

    if output_dir:
        canonical_path = write_eval_outputs(
            output_dir,
            name=run_name,
            description=description,
            results=summary,
            quality=result_quality,
            records=records,
            cfg=cfg,
        )
        print(f"Results saved to: {canonical_path}")
        print("results:\n" + json.dumps(summary, indent=2, ensure_ascii=False))

    return summary, records


def multi_run(cfg: Any):
    """Each ``cfg.multi_run`` key → one ``run_eval``; aggregate ``summary.json``."""
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
    suppress_unnecessary_runtime_output(verbose=bool(cfg.args.verbose))
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

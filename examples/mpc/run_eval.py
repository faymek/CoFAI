"""
Example MPC evaluation script (self-contained; does not import ``cofai.engine``).

Per-sample inference and bitrate accounting match ``inference_model``, ``task_decode_pred``,
and ``_bits_from_coded_data`` in ``cofai/engine/run_eval.py``; task and data-key conventions
are documented in ``docs/engine.md`` (semantic segmentation uses **semseg** throughout).
"""

import os
import sys
import json
import time
import argparse
import importlib
import shutil
import tqdm
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import pandas as pd
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor

from cofai.datasets import *
from cofai.backbone import *
from cofai.heads import *
from cofai.models import *
from cofai.metrics import *
from cofai.metrics.iqa_metrics import create_img_metrics, create_dist_metrics
from cofai.utils.tensor_ops import tensor2image, center_pad, center_crop
from cofai.utils.utils import rename_key_by_rules
from cofai.utils.transforms import rgb2ycbcr, ycbcr2rgb

from dotenv import load_dotenv
load_dotenv()

# from cofai.utils.debug import extract_shapes
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    _FVCORE_AVAILABLE = True
except Exception:
    _FVCORE_AVAILABLE = False

# Disable Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


class ResolutionTransform:
    """
    Resolution transform for common ViT + CV pipelines.

    This utility wraps a pair of operations (``adapt`` / ``revert``)
    that act on 4D image tensors in BCHW format and only modify the spatial
    resolution. Two modes are supported:

    - ``mode="resize"``:
      - ``adapt(x)``: resize input tensor ``x`` to a fixed square size
        ``(size, size)`` using bilinear interpolation, and record the original
        spatial resolution.
      - ``revert(x, size=None)``: resize tensor ``x`` to the given
        spatial size ``(H, W)``. If ``size`` is ``None``, the original spatial
        resolution recorded in ``adapt`` is used.

    - ``mode="center_pad"``:
      - ``adapt(x)``: center-pad input tensor ``x`` so that height and
        width become multiples of ``size`` (treated as a padding multiple),
        and record the padding tuple.
      - ``revert(x, size=None)``: remove padding by calling
        ``center_crop``. If ``size`` is ``None``, the padding tuple recorded
        in ``preprocess`` is used; otherwise, ``size`` is treated as the
        padding tuple.

    This class is intended to provide a simple, symmetric interface for
    resolution-only pre/post processing in evaluation scripts, so that image
    resizing or padding logic can be configured and reused via a single
    object.
    """

    def __init__(self, mode: str = "resize", size: int = 518):
        assert mode in ["resize", "center_pad"], f"Unsupported mode: {mode}"
        self.mode = mode
        self.size = size
        self._orig_size = None
        self._padding = None

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "resize":
            # Remember original spatial size for revert().
            self._orig_size = x.shape[-2:]
            return F.interpolate(
                x,
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )
        elif self.mode == "center_pad":
            # Remember padding for revert().
            x_padded, padding = center_pad(x, self.size)
            self._padding = padding
            return x_padded
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def revert(
        self,
        x: torch.Tensor,
        size=None,
    ) -> torch.Tensor:
        if self.mode == "resize":
            if size is None:
                if self._orig_size is None:
                    raise ValueError("orig_size is not recorded; please provide `size`.")
                size = self._orig_size
            return F.interpolate(
                x,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        elif self.mode == "center_pad":
            if size is None:
                if self._padding is None:
                    raise ValueError("padding is not recorded; please provide `size`.")
                # size = self._padding
            return center_crop(x, self._padding)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


def get_obj_from_str(string, reload=False):
    if "." in string:
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    else:
        return getattr(sys.modules[__name__], string)


def instantiate_class(config, **kwargs):
    # MPC_I2 checks isinstance(x, dict); OmegaConf DictConfig is not a dict—materialize first.
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    else:
        config = dict(config)
    if "type" not in config:
        print(config)
        raise KeyError("Expected key `type` to instantiate.")
    cls = config.pop("type")
    obj = get_obj_from_str(cls)
    return obj(**config, **kwargs)


def _bits_from_coded_unit(data: Dict[str, Any]) -> Dict[str, float]:
    if "strings" in data:
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
    raise KeyError("Expected key `strings` or `likelihoods` in coded unit.")


def _bits_from_coded_data(out: Dict[str, Any]) -> Dict[str, float]:
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
    raise NotImplementedError(f"Unsupported coded_data type: {out_type!r}")


@torch.inference_mode()
def inference_model(
    model: Any,
    x: torch.Tensor,
    *,
    qp: Any = 1,
    real: bool = False,
    tasks: Optional[List[str]] = None,
    task_specs: Any = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Uses only ``compress``/``decompress`` or ``forward_test``; ``task_feats`` are filled by the model."""
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


def task_decode_pred(raw: Any, task: str) -> Any:
    """Same idea as the namesake in cofai/engine/run_eval.py: map one task_feats entry to a meter-ready pred."""
    task = str(task)
    output = raw

    if task in {"rec", "rae"}:
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
        output.clamp_(min=0.0)
        output = output.permute(0, 2, 3, 1)
    elif task == "scene":
        _, output = torch.max(output, dim=1)
    elif task == "cls":
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


def profile_function(func, x, **kwargs):
    """Optionally run one-shot FLOPs / parameter stats for ``func`` and print them.

    Args:
        func: Callable to analyze.
        x: Single-sample input tensor passed as inputs to FlopCountAnalysis.
        **kwargs: Forwarded into ``func`` when profiling (e.g. qp, tasks).

    Returns:
        True if profiling ran and printed; False otherwise.
    """
    if not _FVCORE_AVAILABLE:
        print("[profile] fvcore not found; install with: pip install fvcore")
        return False

    try:
        # Wrap as nn.Module so FlopCountAnalysis can call with kwargs.
        import torch.nn as nn

        class _FuncModule(nn.Module):
            def __init__(self, wrapped_func, call_kwargs):
                super().__init__()
                self.wrapped_func = wrapped_func
                self.call_kwargs = call_kwargs

            def forward(self, input_tensor):
                return self.wrapped_func(input_tensor, **self.call_kwargs)

        wrapper = _FuncModule(func, kwargs)

        fca = FlopCountAnalysis(wrapper, (x,))
        total_flops = fca.total()
        by_module = fca.by_module()
        by_operator = fca.by_operator()

        print("\n====== Complexity (single sample) ======")
        print(f"Total FLOPs: {total_flops / 1e9:.3f} GFLOPs")
        try:
            print("\nParameter count (by module):")
            target_model = getattr(func, "__self__", None)
            if target_model is not None:
                print(parameter_count_table(target_model))
        except Exception:
            pass

        if isinstance(by_module, dict) and len(by_module) > 0:
            print("\nFLOPs by module (top 100):")
            items = sorted(by_module.items(), key=lambda kv: kv[1], reverse=True)[:100]
            for name, flops in items:
                print(f"{name}: {flops / 1e6:.3f} MFLOPs")

        if isinstance(by_operator, dict) and len(by_operator) > 0:
            print("\nFLOPs by operator (top 20):")
            items = sorted(by_operator.items(), key=lambda kv: kv[1], reverse=True)[:20]
            for name, flops in items:
                print(f"{name}: {flops / 1e6:.3f} MFLOPs")

        print("===============================================\n")
        return True
    except Exception as e:
        print(f"[profile] profiling failed: {e}")
        return False


@torch.inference_mode()
def eval_model(cfg):
    # Resolve preset from config.
    preset_name = cfg.args.preset
    assert preset_name in cfg.eval_presets, f"eval preset {preset_name!r} not found"
    preset_config = cfg.eval_presets[preset_name]

    print(f"preset: {preset_name}")
    print(f"description: {preset_config.description}")
    print(f"dataset: {preset_config.dataset}")
    print(f"metric: {preset_config.metric}")
    print(f"model: {cfg.model.type}")
    print(f"head: {cfg.args.head}")

    # Dataset and metric configs
    dataset_config = cfg.datasets[preset_config.dataset]
    metric_config = cfg.metrics[preset_config.metric]
    head_config = cfg.heads[cfg.args.head] if cfg.args.head else None

    device = torch.device(cfg.args.device)
    model = instantiate_class(cfg.model).to(device)

    if "load" in cfg and cfg.load:
        checkpoint = torch.load(cfg.load.path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        if cfg.load.rules:
            sd_new = {}
            for org_key in sorted(state_dict.keys()):
                new_key = rename_key_by_rules(org_key, cfg.load.rules)
                if new_key != "":  # empty string drops the key
                    sd_new[new_key] = state_dict[org_key]
            state_dict = sd_new
        model.load_state_dict(state_dict, strict=cfg.load.strict)
    model.eval()
    # Inference prep
    if hasattr(model, "update"):
        model.update()

    heads_mod = getattr(model, "heads", None)
    has_cls = heads_mod is not None and "cls" in heads_mod
    has_semseg = heads_mod is not None and "semseg" in heads_mod
    has_rae = heads_mod is not None and ("rec" in heads_mod or "rae" in heads_mod)

    task_specs = OmegaConf.select(cfg, "task_specs", default=None)
    if task_specs is not None:
        task_specs = OmegaConf.to_container(task_specs, resolve=True)

    tasks: List[str] = []
    if cfg.args.head and "cls" in cfg.args.head:
        if not has_cls:
            raise KeyError(
                "Classification needs ``task_feats['cls']`` (same contract as engine eval_step): "
                "set ``model.heads.cls`` in the merged config "
                "(see ``examples/mpc/config/eval_MPC2-v3-base-vbr.yaml``)."
            )
        tasks.append("cls")
    if cfg.args.head and "seg" in cfg.args.head:
        if not has_semseg:
            raise KeyError(
                "Semantic segmentation uses key ``semseg`` only (docs/engine.md §5.4): "
                "configure ``model.heads.semseg``; this script does not build seg features from the bitstream."
            )
        tasks.append("semseg")
    if cfg.args.recon != 0:
        if cfg.args.head and "rae" in cfg.args.head:
            # RAE path: decode_rae() → GeneralDecoder → reconstructed image
            if not has_rae:
                raise KeyError(
                    "RAE reconstruction requires ``model.heads.rec`` (or ``model.heads.rae``). "
                    "Set ``--head rae_<name>`` so the head is injected into model.heads.rec."
                )
            tasks.append("rae")
        else:
            tasks.append("rec" + str(cfg.args.recon))

    metric_meter = DictAverageMeter()
    records = []

    # Build dataset
    print("Building dataset...")
    dataset = instantiate_class(dataset_config)
    print(f"dataset size: {len(dataset)}")

    cls_metric = None
    seg_metric = None

    if head_config and "cls" in tasks:
        cls_metric = instantiate_class(metric_config)
        print("Classification: ``task_feats['cls']`` from ``model.heads.cls`` (same as engine).")
    elif head_config and "semseg" in tasks:
        seg_metric = instantiate_class(metric_config)
        print("Segmentation: ``task_feats['semseg']`` from ``model.heads.semseg`` (same as engine).")

    # Image / distribution metrics
    print("Creating image-quality metrics...")
    img_metrics_dict = {}
    dist_metrics_dict = {}
    if cfg.args.recon != 0:
        # img_metrics_dict = create_img_metrics()
        # dist_metrics_dict = create_dist_metrics()
        
        # RAE/MPC reconstruction passes tensors (x_hat, x_orig) here, so only
        # enable full-reference tensor metrics. Directory-level metrics such as
        # detection mAP/FID require saved image/label folders and are not valid
        # for this per-sample call site.
        img_metrics_dict = create_img_metrics(["PSNR", "MS-SSIM", "LPIPS-Alex"])


    # Output dirs
    if cfg.args.output_dir:
        out_sub_dir = f"{cfg.args.output_dir}/{cfg.args.quality}"
        os.makedirs(out_sub_dir, exist_ok=True)
        temp_input_dir = f"{cfg.args.output_dir}/temp_input_dir"
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        os.makedirs(temp_input_dir, exist_ok=True)

    # Evaluation loop
    did_profile = False if getattr(cfg.args, "profile", False) else True
    for raw in tqdm.tqdm(dataset):
        if isinstance(raw, dict):
            batch = raw
            x = batch["img"]
            img_meta = dict(batch["meta"])
            if "semseg" in batch:
                sl = np.asarray(batch["semseg"], dtype=np.int64).squeeze(-1)
                img_meta["seg_label"] = sl
        else:
            x, img_meta = raw
            img_meta = dict(img_meta)
        x = ToTensor()(x).to(device)
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x_orig = x.clone()

        reso_transform: ResolutionTransform = instantiate_class(cfg.resolution_transform)
        if "rae" in tasks and reso_transform.mode == "center_pad":
            rae_pad_multiple = int(getattr(model.dino, "patch_size", 16)) * 8
            reso_transform.size = max(int(reso_transform.size), rae_pad_multiple)
        x_adapt = reso_transform.adapt(x_orig)
        if hasattr(model, "use_yuv") and model.use_yuv:
            x_adapt = rgb2ycbcr(x_adapt)


        # Optional: one-shot forward_test FLOPs / parameter count
        if not did_profile:
            profile_kw: Dict[str, Any] = {"qp": cfg.args.quality, "tasks": tasks}
            if task_specs is not None:
                profile_kw["task_specs"] = task_specs
            did_profile = profile_function(model.forward_test, x_adapt, **profile_kw)

        time_items, bits_items, out_net = inference_model(
            model,
            x_adapt,
            qp=cfg.args.quality,
            real=cfg.args.real,
            tasks=tasks,
            task_specs=task_specs,
        )
        missing = [t for t in tasks if t not in (out_net or {})]
        if missing:
            avail = sorted(list((out_net or {}).keys()))
            raise KeyError(
                "Model did not return outputs for all declared tasks (same check as cofai/engine/run_eval.eval_step). "
                f"missing={missing!r}, available={avail!r}"
            )
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp_items = {f"bpp_{k}": v / num_pixels for k, v in bits_items.items()}
        bpp = sum(bpp_items.values())

        out_result = {
            **time_items,
            "bpp": bpp,
            **bpp_items,
        }

        if hasattr(model, "get_feature_numel"):
            numel = model.get_feature_numel(x_adapt)
            out_result["bpfp"] = sum(bits_items.values()) / numel

        # Image quality metrics
        iqa_result = {}
        if cfg.args.recon != 0:
            if "rae" in out_net:
                x_hat = out_net["rae"].clamp(0, 1)
            elif "x_hat" in out_net:
                x_hat = out_net["x_hat"]
            else:
                x_hat = out_net["rec2" if cfg.args.recon == 2 else "rec1"]
                x_hat = x_hat.clamp(0, 1)

            if hasattr(model, "use_yuv") and model.use_yuv:
                x_hat = ycbcr2rgb(x_hat)
            x_hat = reso_transform.revert(x_hat)

            # PSNR etc.
            iqa_result = {
                key: func(x_hat, x_orig).item() for key, func in img_metrics_dict.items()
            }

        # Classification metrics (engine: task_decode_pred(task_feats[label], kind))
        if cls_metric is not None:
            pred_cls = task_decode_pred(out_net["cls"], "cls")
            cls_metric.update(pred_cls, [img_meta["cls_label"]])

        # Segmentation: spatial align then semseg decode
        if seg_metric is not None:
            raw_sem = reso_transform.revert(out_net["semseg"])
            pred_sem = task_decode_pred(raw_sem, "semseg")
            if isinstance(pred_sem, torch.Tensor):
                pred_sem = pred_sem.squeeze(0).detach().cpu().numpy()
            seg_metric.update(pred_sem, img_meta["seg_label"])

        # Per-sample record
        file = img_meta["img_path"]
        record = {"file": file, "quality": cfg.args.quality}
        record.update(out_result)
        record.update(iqa_result)

        metric_meter.update(record)

        if cfg.args.verbose:
            # _rv = {key: round(value, 4) for key, value in per_file_record.items()}
            print(file, record)

        # per_file_record = {key: round(value, 8) for key, value in per_file_record.items()}
        records.append(record)

    # Average meters
    avg_metrics = metric_meter.average()

    if cfg.args.recon != 0:
        dist_result = {
            key: func(temp_input_dir, out_sub_dir)
            for key, func in dist_metrics_dict.items()
        }
        avg_metrics.update(dist_result)

    # Merge task metrics
    if cls_metric:
        cls_results = cls_metric.compute()
        avg_metrics.update(cls_results)

    if seg_metric:
        seg_results = seg_metric.compute()
        avg_metrics.update(seg_results)

    avg_metrics = {key: round(value, 6) for key, value in avg_metrics.items()}
    return avg_metrics, records


def setup_args():
    """Define CLI arguments."""
    parser = argparse.ArgumentParser(description="MPC model evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        nargs="+",  # one or more YAML paths (merged in order)
        help="Config file path(s); multiple files are merged left-to-right",
    )
    # parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--preset", type=str, required=True, help="Eval preset name (eval_presets key)")
    parser.add_argument(
        "--head",
        type=str,
        help="Registered heads[<name>]; if name contains cls/seg, overrides model.heads.cls / semseg",
    )
    parser.add_argument("--quality", type=str, default="1.0", help="Quality / QP index")
    parser.add_argument("--real", action="store_true", help="Use actual compress/decompress")
    parser.add_argument(
        "--recon", type=int, default=2, choices=[0, 1, 2, 3], help="Reconstruction stage index"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose per-sample prints")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--output_dir", type=str, default="", help="Directory for JSON outputs")
    parser.add_argument(
        "--profile", action="store_true", help="Print one-shot FLOPs / params by module and operator"
    )
    parser.add_argument(
        "--multi-run",
        action="store_true",
        help="If merged config has multi_run, run once per entry; else single --quality run",
    )
    return parser


def merge_args_to_config(config, args):
    """Attach parsed CLI args under config.args."""
    args_dict = dict(vars(args))
    args_dict["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    # Latent codecs index qp with tensor slices; must be int (e.g. "1.0" -> 1).
    args_dict["quality"] = int(float(args_dict["quality"]))
    config.args = OmegaConf.create(args_dict)
    return config


def apply_cli_head_overrides(config) -> None:
    k = config.args.head
    if not k:
        return
    if k not in config.heads:
        available = sorted(list(config.heads.keys()))
        raise KeyError(
            f"Head {k!r} not found in config.heads. Available heads: {available}"
        )
    if "cls" in k:
        config.model.heads.cls = OmegaConf.create(
            OmegaConf.to_container(config.heads[k], resolve=True)
        )
    if "seg" in k:
        config.model.heads.semseg = OmegaConf.create(
            OmegaConf.to_container(config.heads[k], resolve=True)
        )
    if "rae" in k:
        # Inject as model.heads.rec so MPC_I2.forward_test picks it up via
        # head_key = "rec" if "rec" in self.heads else "rae"
        config.model.heads.rec = OmegaConf.create(
            OmegaConf.to_container(config.heads[k], resolve=True)
        )


def main(config, args):
    # All instantiation happens inside eval_model.
    print(f"\n[{args.preset}] evaluation start")
    print("=" * 50)
    avg_metrics, records = eval_model(config)

    # Results
    print(f"\n[{args.preset}] evaluation results")
    print("=" * 50)
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")

    result = {
        "task": args.preset,
        "quality": config.args.quality,
        "description": config.eval_presets[args.preset].description,
        "results": avg_metrics,
        "records": records,
    }
    # Save JSON
    if config.args.output_dir:
        result_file = os.path.join(
            config.args.output_dir, f"{args.preset}_results_{config.args.quality}.json"
        )
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to: {result_file}")
    return result


if __name__ == "__main__":
    parser = setup_args()
    args = parser.parse_args()

    # Load and merge YAML configs
    config = OmegaConf.load(args.config[0])
    for config_path in args.config[1:]:
        overlay_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, overlay_config)

    # Merge CLI into config
    multi_run_results = []
    config = merge_args_to_config(config, args)
    apply_cli_head_overrides(config)
    use_multi = getattr(args, "multi_run", False) and "multi_run" in config
    if use_multi:
        this_cfg = config.copy()
        for quality, patchy_cfg in config.multi_run.items():
            args.quality = quality
            this_cfg.args.quality = quality
            if patchy_cfg is not None:
                this_cfg = OmegaConf.merge(this_cfg, patchy_cfg)
            result = main(this_cfg, args)
            multi_run_results.append(result)
    else:
        result = main(config, args)
        multi_run_results.append(result)

    if len(multi_run_results) > 1:
        rows = []
        for result in multi_run_results:
            row = result["results"]
            row.update({"quality": result["quality"]})
            rows.append(row)
        combined_df = pd.DataFrame(rows)

        print(f"\n[{args.preset}] multi-run summary")
        print("=" * 50)
        print(combined_df)
        summary_results = combined_df.to_dict(orient="list")
        final_result = {
            "task": args.preset,
            "description": config.eval_presets[args.preset].description,
            "results": summary_results,
        }
        if config.args.output_dir:
            json_path = os.path.join(config.args.output_dir, f"{args.preset}_results.json")
            with open(json_path, "w") as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            print(f"Saved summary results to {json_path}")

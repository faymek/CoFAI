"""S
This script keeps only the depth-estimation inference path split from
``examples/mpc/run_eval.py``. It expects the configured model to return
``task_feats["depth"]`` as a list of patch2d features.
"""

import argparse
import importlib
import json
import math
import os
import shutil
import sys
import time
import warnings
from typing import Any, Dict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor

from cofai.backbone import *
from cofai.datasets import *
from cofai.heads import *
from cofai.metrics import *
from cofai.models import *
from cofai.utils.tensor_ops import center_crop, center_pad
from cofai.utils.transforms import rgb2ycbcr
from cofai.utils.utils import rename_key_by_rules
from cofai.backbone.dinov3.eval.depth.datasets.datasets_utils import (
    _EvalCropType,
    make_valid_mask,
)
from cofai.backbone.dinov3.eval.depth.metrics import (
    DEPTH_METRICS,
    calculate_depth_metrics,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


class ResolutionTransform:
    def __init__(self, mode: str = "resize", size: int = 518):
        assert mode in ["resize", "center_pad"], f"Unsupported mode: {mode}"
        self.mode = mode
        self.size = size
        self._orig_size = None
        self._padding = None

    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "resize":
            self._orig_size = x.shape[-2:]
            return F.interpolate(
                x,
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )
        x_padded, padding = center_pad(x, self.size)
        self._padding = padding
        return x_padded

    def revert(self, x: torch.Tensor, size=None) -> torch.Tensor:
        if self.mode == "resize":
            if size is None:
                if self._orig_size is None:
                    raise ValueError("orig_size is not recorded; please provide `size`.")
                size = self._orig_size
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if self._padding is None:
            raise ValueError("padding is not recorded; call adapt() before revert().")
        return center_crop(x, self._padding)


def get_obj_from_str(string, reload=False):
    if "." in string:
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    return getattr(sys.modules[__name__], string)


def instantiate_class(config, **kwargs):
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    else:
        config = dict(config)
    if "type" not in config:
        raise KeyError("Expected key `type` to instantiate.")
    cls = config.pop("type")
    return get_obj_from_str(cls)(**config, **kwargs)


def calc_bits_items_for_unit_data(data: Dict[str, Any]) -> Dict[str, float]:
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
    raise KeyError("Expected key `strings`, `likelihoods`, or `bits` in coded unit.")


def calc_bits_items(out: Dict[str, Any]) -> Dict[str, float]:
    if "type" not in out:
        return calc_bits_items_for_unit_data(out)
    if out["type"] == "unit":
        return calc_bits_items_for_unit_data(out["data"])
    if out["type"] == "frame":
        flat = {}
        for layer_name, coded_unit in out["data"].items():
            for k, v in calc_bits_items_for_unit_data(coded_unit).items():
                flat[f"{layer_name}.{k}"] = v
        return flat
    if out["type"] == "frame_wise_video":
        flat = {}
        for frame_name, coded_frame in out["data"].items():
            for layer_name, coded_unit in coded_frame["data"].items():
                for k, v in calc_bits_items_for_unit_data(coded_unit).items():
                    flat[f"{frame_name}.{layer_name}.{k}"] = v
        return flat
    if out["type"] == "layer_wise_video":
        flat = {}
        for layer_name, coded_frame in out["data"].items():
            for frame_name, coded_unit in coded_frame["data"].items():
                for k, v in calc_bits_items_for_unit_data(coded_unit).items():
                    flat[f"{layer_name}.{frame_name}.{k}"] = v
        return flat
    raise NotImplementedError(f"Unsupported type: {out.get('type', None)}")


@torch.inference_mode()
def inference_depth(model, x, qp=1, real=False):
    tasks = ["depth"]
    if real:
        start = time.time()
        coded_data = model.compress(x, qp=qp)
        enc_time = time.time() - start
        start = time.time()
        task_feats = model.decompress(coded_data, tasks=tasks)
        dec_time = time.time() - start
    else:
        start = time.time()
        coded_data, task_feats = model.forward_test(x, qp=qp, tasks=tasks)
        elapsed_time = time.time() - start
        enc_time = elapsed_time / 2.0
        dec_time = elapsed_time / 2.0

    if "depth" not in task_feats:
        raise RuntimeError(
            "Missing `task_feats['depth']`; the model must support the depth task."
        )
    return {"enc_time": enc_time, "dec_time": dec_time}, calc_bits_items(coded_data), task_feats


def _sample_to_image_and_meta(sample):
    if isinstance(sample, dict):
        meta = dict(sample.get("meta", {}))
        if "depth" in sample:
            meta["depth_label"] = np.asarray(sample["depth"])
        return sample["img"], meta
    x, meta = sample
    return x, dict(meta)


def _to_tensor_image(x, device):
    if torch.is_tensor(x):
        x = x.float()
        if x.dim() == 3 and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
        if x.max() > 1.0:
            x = x / 255.0
    else:
        x = ToTensor()(x)
    x = x.to(device)
    return x.unsqueeze(0) if x.dim() == 3 else x


@torch.inference_mode()
def eval_depth(cfg):
    preset_name = cfg.args.preset
    assert preset_name in cfg.eval_presets, f"eval preset {preset_name!r} not found"
    preset_config = cfg.eval_presets[preset_name]

    dataset_config = cfg.datasets[preset_config.dataset]
    metric_config = cfg.metrics[preset_config.metric]
    head_config = cfg.heads[cfg.args.head]

    device = torch.device(cfg.args.device)
    print(f"preset: {preset_name}")
    print(f"description: {preset_config.description}")
    print(f"dataset: {preset_config.dataset}")
    print(f"metric: {preset_config.metric}")
    print(f"model: {cfg.model.type}")
    print(f"head: {cfg.args.head}")

    model = instantiate_class(cfg.model).to(device)
    if "load" in cfg and cfg.load and getattr(cfg.load, "path", None):
        checkpoint = torch.load(cfg.load.path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        if getattr(cfg.load, "rules", None):
            state_dict = {
                new_key: state_dict[org_key]
                for org_key in sorted(state_dict.keys())
                for new_key in [rename_key_by_rules(org_key, cfg.load.rules)]
                if new_key != ""
            }
        model.load_state_dict(state_dict, strict=cfg.load.strict)
    model.eval()
    if hasattr(model, "update"):
        model.update()

    dep_head = instantiate_class(head_config).to(device).eval()
    dep_metric_names = list(metric_config.names)
    metrics_spec = [m for m in DEPTH_METRICS if m.name in dep_metric_names]

    print("Building dataset...")
    dataset = instantiate_class(dataset_config)
    print(f"dataset size: {len(dataset)}")

    if cfg.args.output_dir:
        os.makedirs(cfg.args.output_dir, exist_ok=True)
        out_sub_dir = os.path.join(cfg.args.output_dir, str(cfg.args.quality))
        os.makedirs(out_sub_dir, exist_ok=True)
        temp_input_dir = os.path.join(cfg.args.output_dir, "temp_input_dir")
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        os.makedirs(temp_input_dir, exist_ok=True)

    totals = {name: 0.0 for name in dep_metric_names}
    count = 0
    records = []

    for sample in tqdm.tqdm(dataset):
        x_raw, img_meta = _sample_to_image_and_meta(sample)
        if "depth_label" not in img_meta:
            raise KeyError("Depth evaluation requires `depth_label` in dataset metadata or sample dict.")

        x_orig = _to_tensor_image(x_raw, device)
        reso_transform = instantiate_class(cfg.resolution_transform)
        x_adapt = reso_transform.adapt(x_orig)
        if hasattr(model, "use_yuv") and model.use_yuv:
            x_adapt = rgb2ycbcr(x_adapt)

        time_items, bits_items, out_net = inference_depth(
            model,
            x_adapt,
            qp=cfg.args.quality,
            real=cfg.args.real,
        )
        num_pixels = x_orig.size(0) * x_orig.size(2) * x_orig.size(3)
        bpp_items = {f"bpp_{k}": v / num_pixels for k, v in bits_items.items()}

        depth_label = torch.as_tensor(img_meta["depth_label"], device=device).float()
        if depth_label.dim() == 2:
            depth_label = depth_label.unsqueeze(0).unsqueeze(0)
        elif depth_label.dim() == 3:
            depth_label = depth_label.unsqueeze(0)
        depth_label = depth_label / float(getattr(metric_config, "normalization_constant", 1000.0))

        target_size = tuple(int(v) for v in depth_label.shape[-2:])
        preds = dep_head.predict(out_net["depth"], size=target_size)

        if bool(getattr(metric_config, "use_tta", False)):
            x_flip = torch.flip(x_orig, dims=[3])
            x_flip = reso_transform.adapt(x_flip)
            if hasattr(model, "use_yuv") and model.use_yuv:
                x_flip = rgb2ycbcr(x_flip)
            _, _, out_net_flip = inference_depth(
                model,
                x_flip,
                qp=cfg.args.quality,
                real=cfg.args.real,
            )
            preds_flip = dep_head.predict(out_net_flip["depth"], size=target_size)
            preds = 0.5 * (preds + torch.flip(preds_flip, dims=[3]))

        preds = preds.clamp(
            min=float(metric_config.min_depth),
            max=float(metric_config.max_depth),
        )
        gt_for_eval = torch.where(
            torch.logical_or(
                depth_label >= float(metric_config.max_depth),
                depth_label <= float(metric_config.min_depth),
            ),
            torch.tensor(float(metric_config.ignored_value), device=device, dtype=depth_label.dtype),
            depth_label,
        )
        valid_mask = make_valid_mask(
            gt_for_eval,
            eval_crop=_EvalCropType(metric_config.eval_mask),
            ignored_value=float(metric_config.ignored_value),
        )
        dep_metrics = calculate_depth_metrics(
            gt_for_eval,
            preds,
            valid_mask,
            list_metrics=metrics_spec,
        )
        dep_result = {name: float(getattr(dep_metrics, name)) for name in dep_metric_names}
        for name, value in dep_result.items():
            totals[name] += value
        count += 1

        record = {
            "file": img_meta.get("img_path", img_meta.get("img_name", str(count - 1))),
            "quality": cfg.args.quality,
            **time_items,
            "bpp": sum(bpp_items.values()),
            **bpp_items,
            **dep_result,
        }
        if hasattr(model, "get_feature_numel"):
            record["bpfp"] = sum(bits_items.values()) / model.get_feature_numel(x_adapt)
        records.append(record)
        if cfg.args.verbose:
            print(record["file"], record)

    avg_metrics = {name: (totals[name] / count if count else 0.0) for name in dep_metric_names}
    bitrate_keys = sorted({k for r in records for k in r if k.startswith("bpp") or k in {"bpfp", "enc_time", "dec_time"}})
    for key in bitrate_keys:
        values = [float(r[key]) for r in records if key in r]
        if values:
            avg_metrics[key] = float(sum(values) / len(values))
    return {k: round(v, 6) for k, v in avg_metrics.items()}, records


def setup_args():
    parser = argparse.ArgumentParser(description="MPC3 depth evaluation")
    parser.add_argument("--config", type=str, required=True, nargs="+")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional model checkpoint override")
    parser.add_argument("--head_checkpoint", type=str, default="", help="Optional depth head checkpoint override")
    parser.add_argument("--preset", type=str, default="")
    parser.add_argument("--task", type=str, default="", help="Alias for --preset")
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--quality", type=str, default="1.0")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    return parser


def merge_args_to_config(config, args):
    args_dict = dict(vars(args))
    if not args_dict.get("preset") and args_dict.get("task"):
        args_dict["preset"] = args_dict["task"]
    args.preset = args_dict["preset"]
    if not args.preset:
        raise ValueError("Depth eval requires --preset or --task.")
    args_dict["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    args_dict["quality"] = int(float(args_dict["quality"]))
    config.args = OmegaConf.create(args_dict)
    if args_dict.get("checkpoint"):
        if "load" not in config or config.load is None:
            config.load = OmegaConf.create({})
        config.load.path = args_dict["checkpoint"]
    if args_dict.get("head_checkpoint"):
        config.heads[args.head].checkpoint = args_dict["head_checkpoint"]
    return config


def main(config, args):
    avg_metrics, records = eval_depth(config)
    print(f"\n[{args.preset}] depth evaluation results")
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
    if config.args.output_dir:
        result_file = os.path.join(
            config.args.output_dir,
            f"{args.preset}_depth_results_{config.args.quality}.json",
        )
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {result_file}")
    return result


if __name__ == "__main__":
    parser = setup_args()
    args = parser.parse_args()

    config = OmegaConf.load(args.config[0])
    for config_path in args.config[1:]:
        config = OmegaConf.merge(config, OmegaConf.load(config_path))
    config = merge_args_to_config(config, args)

    if "multi_run" in config:
        results = []
        base_cfg = config.copy()
        for quality, patch_cfg in config.multi_run.items():
            this_cfg = base_cfg.copy()
            this_cfg.args.quality = int(float(quality))
            args.quality = str(quality)
            if patch_cfg is not None:
                this_cfg = OmegaConf.merge(this_cfg, patch_cfg)
            results.append(main(this_cfg, args))
        rows = []
        for result in results:
            row = dict(result["results"])
            row["quality"] = result["quality"]
            rows.append(row)
        summary = pd.DataFrame(rows)
        print(f"\n[{args.preset}] multi-run summary")
        print("=" * 50)
        print(summary)
    else:
        main(config, args)

"""
统一的评估脚本，支持从配置文件中指定任务
"""

import os
import sys
import json
import time
import argparse
import importlib
import shutil
import hashlib
import tqdm
import math
import warnings
import numpy as np
from PIL import Image
import pandas as pd
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torchvision import transforms
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
from cofai.utils.debug import tensor_hash
from cofai.backbone.dinov3.eval.depth.metrics import DEPTH_METRICS, calculate_depth_metrics
from cofai.backbone.dinov3.eval.depth.datasets.datasets_utils import _EvalCropType, make_valid_mask
import matplotlib.pyplot as plt

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


def hash_any(x):
    """Compute a stable SHA256 hash for nested tensor-like outputs."""
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return tensor_hash(x)
    if isinstance(x, (list, tuple)):
        sub_hash = [hash_any(v) for v in x]
        return hashlib.sha256(json.dumps(sub_hash, sort_keys=True).encode("utf-8")).hexdigest()
    if isinstance(x, dict):
        sub_hash = {k: hash_any(v) for k, v in sorted(x.items(), key=lambda kv: kv[0])}
        return hashlib.sha256(json.dumps(sub_hash, sort_keys=True).encode("utf-8")).hexdigest()
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


def shape_any(x):
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return [shape_any(v) for v in x]
    if isinstance(x, dict):
        return {k: shape_any(v) for k, v in x.items()}
    return type(x).__name__


def append_hash_trace(trace, stage, x):
    trace.append(
        {
            "stage": stage,
            "hash": hash_any(x),
            "shape": shape_any(x),
            "dtype": str(getattr(x, "dtype", type(x))),
        }
    )


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
            # 记录原始尺寸，以便在 revert 中恢复
            self._orig_size = x.shape[-2:]
            return F.interpolate(
                x,
                size=(self.size, self.size),
                mode="bilinear",
                align_corners=False,
            )
        elif self.mode == "center_pad":
            # 记录 padding 信息，以便在 revert 中恢复
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
    config = config.copy()
    if "type" in config:
        cls = config.pop("type")
        obj = get_obj_from_str(cls)
        return obj(**config, **kwargs)
    else:
        print(config)
        raise KeyError("Expected key `type` to instantiate.")


def calc_bits_items_for_unit_data(data):
    if "strings" in data:  # real compression
        bits_items = {
            f"{name}": sum(len(s[0]) for s in sub_strings) * 8.0
            for name, sub_strings in data["strings"].items()
        }
        return bits_items
    elif "likelihoods" in data:
        bits_items = {
            f"{name}": (torch.log(likelihoods).sum() / (-math.log(2))).item()
            for name, likelihoods in data["likelihoods"].items()
        }
        return bits_items
    elif "bits" in data:
        return data["bits"]
    else:
        raise KeyError("Expected key `strings` or `likelihoods` in out_enc.")


def calc_bits_items(out):
    flatten_bits_items = {}
    if "type" not in out:
        # Returns values in an adapted (partially compatible) CompressAI format.
        # is called coded_unit in this reference software
        # see the API docs
        return calc_bits_items_for_unit_data(out)

    # wrapper designed in this reference software
    # to organize the coded_units
    # see the API docs
    if out["type"] == "unit":
        return calc_bits_items_for_unit_data(out["data"])
    elif out["type"] == "frame":
        for layer_name, coded_unit in out["data"].items():
            bits_items = calc_bits_items_for_unit_data(coded_unit)
            for k, v in bits_items.items():
                flatten_bits_items[f"{layer_name}.{k}"] = v
        return flatten_bits_items
    elif out["type"] == "frame_wise_video":
        for frame_name, coded_frame in out["data"].items():
            for layer_name, coded_unit in coded_frame["data"].items():
                bits_items = calc_bits_items_for_unit_data(coded_unit)
                for k, v in bits_items.items():
                    flatten_bits_items[f"{frame_name}.{layer_name}.{k}"] = v
        return flatten_bits_items
    elif out.get("type", None) == "layer_wise_video":
        for layer_name, coded_frame in out["data"].items():
            for frame_name, coded_unit in coded_frame["data"].items():
                bits_items = calc_bits_items_for_unit_data(coded_unit)
                for k, v in bits_items.items():
                    flatten_bits_items[f"{layer_name}.{frame_name}.{k}"] = v
        return flatten_bits_items
    else:
        raise NotImplementedError(f"Unsupported type: {out.get('type', None)}")


@torch.inference_mode()
def inference_x(
    model, x, qp=1, real=False, tasks=[]
):
    """推理单个文件"""
    if real:  # 实际压缩
        start = time.time()
        coded_data = model.compress(x, qp=qp)
        enc_time = time.time() - start
        start = time.time()
        task_feats = model.decompress(
            coded_data,
            tasks=tasks,
        )
        dec_time = time.time() - start
        # print("coded_data", extract_shapes(coded_data))
        bits_items = calc_bits_items(coded_data)
        time_items = {
            "enc_time": enc_time,
            "dec_time": dec_time,
        }
    else:  # 估计
        start = time.time()
        coded_data, task_feats = model.forward_test(
            x,
            qp=qp,
            tasks=tasks,
        )
        elapsed_time = time.time() - start
        bits_items = calc_bits_items(coded_data)
        time_items = {
            "enc_time": elapsed_time / 2.0,  # 粗略估计
            "dec_time": elapsed_time / 2.0,
        }

    return time_items, bits_items, task_feats


def profile_function(func, x, **kwargs):
    """可选地对 func 进行一次 FLOPs/参数统计并打印。

    参数:
        func: 被分析的函数
        x: 单样本输入张量，将作为 inputs 传给 FlopCountAnalysis
        **kwargs: 可选，支持传入 cfg（用于读取 profile 开关）

    返回值:
        bool: 若已成功执行统计并打印，返回 True，否则返回 False。
    """
    if not _FVCORE_AVAILABLE:
        print("[profile] 未检测到 fvcore，请先安装：pip install fvcore")
        return False

    try:
        # 将函数包装为 nn.Module，以便 FlopCountAnalysis 能够调用并传递 kwargs
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

        print("\n====== 计算复杂度（单样本）======")
        print(f"Total FLOPs: {total_flops / 1e9:.3f} GFLOPs")
        try:
            print("\n参数统计（按模块汇总）:")
            target_model = getattr(func, "__self__", None)
            if target_model is not None:
                print(parameter_count_table(target_model))
        except Exception:
            pass

        if isinstance(by_module, dict) and len(by_module) > 0:
            print("\n按模块 FLOPs Top-100：")
            items = sorted(by_module.items(), key=lambda kv: kv[1], reverse=True)[:100]
            for name, flops in items:
                print(f"{name}: {flops / 1e6:.3f} MFLOPs")

        if isinstance(by_operator, dict) and len(by_operator) > 0:
            print("\n按算子 FLOPs Top-20：")
            items = sorted(by_operator.items(), key=lambda kv: kv[1], reverse=True)[:20]
            for name, flops in items:
                print(f"{name}: {flops / 1e6:.3f} MFLOPs")

        print("===============================================\n")
        return True
    except Exception as e:
        print(f"[profile] 统计失败：{e}")
        return False


@torch.inference_mode()
def eval_model(cfg):
    # 从配置中获取任务信息
    preset_name = cfg.args.preset
    assert preset_name in cfg.eval_presets, f"任务预设 {preset_name} 不存在"
    preset_config = cfg.eval_presets[preset_name]

    tasks = []
    if cfg.args.head and "cls" in cfg.args.head:
        tasks.append("cls")
    if cfg.args.head and "seg" in cfg.args.head:
        tasks.append("seg")
    if cfg.args.head and "dep" in cfg.args.head:
        tasks.append("seg")
    if cfg.args.recon != 0:
        tasks.append("rec"+str(cfg.args.recon))

    print(f"预设: {preset_name}")
    print(f"描述: {preset_config.description}")
    print(f"数据: {preset_config.dataset}")
    print(f"指标: {preset_config.metric}")
    print(f"模型: {cfg.model.type}")
    print(f"头部: {cfg.args.head}")

    # 获取数据集和指标配置
    dataset_config = cfg.datasets[preset_config.dataset]
    metric_config = cfg.metrics[preset_config.metric]
    head_config = cfg.heads[cfg.args.head] if cfg.args.head else None

    device = torch.device(cfg.args.device)
    model = instantiate_class(cfg.model).to(device)

    if "load" in cfg and cfg.load and getattr(cfg.load, "path", None):
        checkpoint = torch.load(cfg.load.path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        if getattr(cfg.load, "rules", None):
            sd_new = {}
            for org_key in sorted(state_dict.keys()):
                new_key = rename_key_by_rules(org_key, cfg.load.rules)
                if new_key != "":  # 为空表示删除该key
                    sd_new[new_key] = state_dict[org_key]
            state_dict = sd_new
        model.load_state_dict(state_dict, strict=cfg.load.strict)
    model.eval()
    # 模型推理
    if hasattr(model, "update"):
        model.update()

    metric_meter = DictAverageMeter()
    records = []

    # 构建数据集
    print("构建数据集...")
    dataset = instantiate_class(dataset_config)
    print(f"数据集大小: {len(dataset)}")

    # 构建头部模型和指标
    cls_head = None
    cls_metric = None
    seg_head = None
    seg_metric = None
    dep_head = None
    dep_metric_names = None

    # 根据任务类型构建相应的头部和指标
    if head_config and "cls" in preset_name:
        print("构建分类头部...")
        cls_head = instantiate_class(head_config).to(device).eval()
        cls_metric = instantiate_class(metric_config)
    elif head_config and "seg" in preset_name:
        print("构建分割头部...")
        seg_head = instantiate_class(head_config).to(device).eval()
        seg_metric = instantiate_class(metric_config)
    elif head_config and "dep" in preset_name:
        print("构建深度头部...")
        dep_head = instantiate_class(head_config).to(device).eval()
        dep_metric_names = list(metric_config.names)

    # 创建图像和分布指标
    print("创建图像质量指标...")
    img_metrics_dict = {}
    dist_metrics_dict = {}
    if cfg.args.recon != 0:
        img_metrics_dict = create_img_metrics()
        dist_metrics_dict = create_dist_metrics()

    # 创建输出目录
    if cfg.args.output_dir:
        out_sub_dir = f"{cfg.args.output_dir}/{cfg.args.quality}"
        os.makedirs(out_sub_dir, exist_ok=True)
        temp_input_dir = f"{cfg.args.output_dir}/temp_input_dir"
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        os.makedirs(temp_input_dir, exist_ok=True)

    # 评估循环
    did_profile = False if getattr(cfg.args, "profile", False) else True
    vis_count = 0
    if getattr(cfg.args, "vis_n", 0) > 0:
        os.makedirs(cfg.args.vis_dir, exist_ok=True)

    for sample_idx, (x, img_meta) in enumerate(tqdm.tqdm(dataset)):
        stop_after_this_sample = False
        x = ToTensor()(x).to(device)
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x_orig = x.clone()

        hash_trace = []
        do_hash_trace = getattr(cfg.args, "hash_debug", False) and sample_idx == cfg.args.hash_idx
        if do_hash_trace:
            append_hash_trace(hash_trace, "input.x_orig", x_orig)
            
        reso_transform: ResolutionTransform = instantiate_class(cfg.resolution_transform)
        x_adapt = reso_transform.adapt(x_orig)

        if do_hash_trace:
            append_hash_trace(hash_trace, "input.x_adapt", x_adapt)

        if hasattr(model, "use_yuv") and model.use_yuv:
            x_adapt = rgb2ycbcr(x_adapt)

        # 可选：统计一次模型 forward_test 的计算复杂度（FLOPs）与参数量
        if not did_profile:
            did_profile = profile_function(
                model.forward_test, 
                x_adapt,
                qp=cfg.args.quality,
                tasks=tasks,
            )

        time_items, bits_items, out_net = inference_x(
            model,
            x_adapt,
            qp=cfg.args.quality,
            real=cfg.args.real,
            tasks=tasks,
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

        # 计算图像质量指标
        iqa_result = {}
        if cfg.args.recon != 0:
            if "x_hat" in out_net:
                x_hat = out_net["x_hat"]
                # x_hat = x_hat.clamp(0, 1)
            else:
                x_hat = out_net["rec2" if cfg.args.recon == 2 else "rec1"]
                x_hat = x_hat.clamp(0, 1)

            if hasattr(model, "use_yuv") and model.use_yuv:
                x_hat = ycbcr2rgb(x_hat)
            x_hat = reso_transform.revert(x_hat)

            # 计算PSNR
            iqa_result = {
                key: func(x_hat, x_orig).item() for key, func in img_metrics_dict.items()
            }

        # 更新分类指标
        if "cls" in tasks:
            logits = cls_head.forward(out_net["cls"])
            cls_preds = F.softmax(logits, dim=1)
            values, top_indices = torch.topk(cls_preds, k=5, dim=1)
            cls_metric.update(top_indices, [img_meta["cls_label"]])

        # 更新分割指标
        if seg_head is not None:
            seg_out = out_net.get("seg", None)
            if seg_out is None:
                raise RuntimeError("Missing 'seg' output from model.decompress().")
            if do_hash_trace:
                append_hash_trace(hash_trace, "out_net.seg", seg_out)

            if torch.is_tensor(seg_out):
                if seg_out.dim() == 4:
                    logits = seg_out
                else:
                    logits = seg_head.predict(seg_out, scale=model.patch_size)
            elif isinstance(seg_out, (list, tuple)):
                logits = seg_head.predict(seg_out, scale=model.patch_size)
            else:
                raise TypeError(f"Unsupported seg output type: {type(seg_out)}")
            if do_hash_trace:
                append_hash_trace(hash_trace, "seg.logits_pre_align", logits)

            seg_label = img_meta["seg_label"]  # numpy [H, W]
            tgt_h, tgt_w = int(seg_label.shape[0]), int(seg_label.shape[1])

            # 当输入使用 center_pad 时，优先使用 center_crop 还原，避免插值引入偏差
            if logits.shape[-2:] != (tgt_h, tgt_w):
                if getattr(reso_transform, "mode", None) == "center_pad":
                    logits = reso_transform.revert(logits)
            if do_hash_trace:
                append_hash_trace(hash_trace, "seg.logits_cropped", logits)
                

            # 尺寸仍不一致时再执行插值对齐
            if logits.shape[-2:] != (tgt_h, tgt_w):
                up_mode = "nearest" if cfg.args.seg_nearest else "bilinear"
                if up_mode == "nearest":
                    logits = F.interpolate(logits, size=(tgt_h, tgt_w), mode=up_mode)
                else:
                    logits = F.interpolate(
                        logits, size=(tgt_h, tgt_w), mode=up_mode, align_corners=False
                    )
            # if do_hash_trace:
            #     append_hash_trace(hash_trace, "seg.logits_aligned", logits)

            seg_preds = logits.argmax(dim=1).squeeze(0)
            seg_preds_np = seg_preds.cpu().numpy().astype(np.int64)
            if do_hash_trace:
                append_hash_trace(hash_trace, "seg.preds", seg_preds_np)
                print("\n[HASH_TRACE][run_eval]", img_meta["img_path"])
                for item in hash_trace:
                    print(f"  - {item['stage']}: {item['hash']} shape={item['shape']}")

                hash_out = getattr(cfg.args, "hash_out", "")
                if hash_out:
                    with open(hash_out, "a", encoding="utf-8") as f:
                        payload = {
                            "script": "run_eval",
                            "sample_idx": sample_idx,
                            "file": img_meta["img_path"],
                            "trace": hash_trace,
                        }
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

                if getattr(cfg.args, "hash_only_one", False):
                    stop_after_this_sample = True
            seg_metric.update(seg_preds_np, seg_label)

        # 记录结果
        file = img_meta["img_path"]
        record = {"file": file, "quality": cfg.args.quality}
        record.update(out_result)
        record.update(iqa_result)

        if dep_head is not None:
            dep_out = out_net.get("seg", None)
            if dep_out is None:
                raise RuntimeError("Missing 'seg' output from model.decompress() for depth task.")

            depth_label = torch.from_numpy(img_meta["depth_label"]).float().to(device)
            if depth_label.dim() == 2:
                depth_label = depth_label.unsqueeze(0).unsqueeze(0)
            elif depth_label.dim() == 3:
                depth_label = depth_label.unsqueeze(0)
            depth_label = depth_label / float(getattr(metric_config, "normalization_constant", 1000.0))

            tgt_size = tuple(int(v) for v in depth_label.shape[-2:])
            preds = dep_head.predict(dep_out, size=tgt_size)
            if bool(getattr(metric_config, "use_tta", False)):
                x_flip = torch.flip(x_orig, dims=[3])
                x_flip = reso_transform.adapt(x_flip)
                if hasattr(model, "use_yuv") and model.use_yuv:
                    x_flip = rgb2ycbcr(x_flip)
                _, _, out_net_flip = inference_x(
                    model,
                    x_flip,
                    qp=cfg.args.quality,
                    real=cfg.args.real,
                    tasks=tasks,
                )
                preds_flip = dep_head.predict(out_net_flip["seg"], size=tgt_size)
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
            metrics_spec = [m for m in DEPTH_METRICS if m.name in dep_metric_names]
            dep_metrics = calculate_depth_metrics(
                gt_for_eval,
                preds,
                valid_mask,
                list_metrics=metrics_spec,
            )
            dep_result = {
                name: float(getattr(dep_metrics, name))
                for name in dep_metric_names
            }
            record.update(dep_result)

        metric_meter.update(record)

        if cfg.args.verbose:
            # _rv = {key: round(value, 4) for key, value in per_file_record.items()}
            print(file, record)

        # per_file_record = {key: round(value, 8) for key, value in per_file_record.items()}
        records.append(record)

        if stop_after_this_sample:
            print("[HASH_TRACE][run_eval] hash_only_one=True，当前样本记录完成后提前结束评估。")
            break

    # 计算平均值
    if len(records) == 0:
        avg_metrics = {}
    else:
        avg_metrics = metric_meter.average()

    if cfg.args.recon != 0:
        dist_result = {
            key: func(temp_input_dir, out_sub_dir)
            for key, func in dist_metrics_dict.items()
        }
        avg_metrics.update(dist_result)

    # 添加分类和分割指标
    if cls_metric:
        cls_results = cls_metric.compute()
        avg_metrics.update(cls_results)

    if seg_metric:
        seg_results = seg_metric.compute()
        avg_metrics.update(seg_results)

    avg_metrics = {key: round(value, 6) for key, value in avg_metrics.items()}
    return avg_metrics, records


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="MPC模型评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        nargs="+",  # 允许接收一个或多个参数
        help="配置文件路径，可以指定多个文件",
    )
    parser.add_argument("--checkpoint", type=str, default="", help="模型检查点路径，可选，兼容旧命令")
    parser.add_argument("--head_checkpoint", type=str, default="", help="任务头检查点路径，可选，用于覆盖配置中的checkpoint")
    parser.add_argument("--preset", type=str, default="", help="预定义的评估任务名称")
    parser.add_argument("--task", type=str, default="", help="预定义的评估任务名称，兼容旧命令")
    parser.add_argument(
        "--head", type=str, help="头部模型名称，需要是预定义的头部模型"
    )
    parser.add_argument("--quality", type=str, default="1.0", help="质量参数")
    parser.add_argument("--real", action="store_true", help="使用实际压缩")
    parser.add_argument(
        "--recon", type=int, default=2, choices=[0, 1, 2, 3], help="重建层"
    )
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--cuda", action="store_true", help="使用CUDA")
    parser.add_argument("--output_dir", type=str, default="", help="输出目录")
    parser.add_argument(
        "--profile", action="store_true", help="统计一次FLOPs/参数并按模块与算子打印"
    )
    parser.add_argument("--debug_seg", action="store_true", help="打印分割调试信息")
    parser.add_argument("--seg_nearest", action="store_true", help="分割logits上采样用nearest")
    parser.add_argument("--hash_debug", action="store_true", help="打印关键中间张量hash用于逐步比对")
    parser.add_argument("--hash_idx", type=int, default=0, help="进行hash对比的样本索引")
    parser.add_argument("--hash_out", type=str, default="", help="可选：将hash trace追加写入jsonl文件")
    parser.add_argument("--hash_only_one", action="store_true", help="仅对hash_idx样本执行并提前结束")
    return parser


def merge_args_to_config(config, args):
    """将命令行参数合并到配置中"""
    args_dict = dict(vars(args))
    if not args_dict.get("preset") and args_dict.get("task"):
        args_dict["preset"] = args_dict["task"]
    args.preset = args_dict["preset"]
    args_dict["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.args = OmegaConf.create(args_dict)
    if args_dict.get("checkpoint"):
        if "load" not in config or config.load is None:
            config.load = OmegaConf.create({})
        config.load.path = args_dict["checkpoint"]
    if args_dict.get("head_checkpoint") and args_dict.get("head"):
        if "heads" not in config or config.heads is None or args_dict["head"] not in config.heads:
            raise KeyError(f"Cannot override checkpoint for undefined head: {args_dict['head']}")
        config.heads[args_dict["head"]].checkpoint = args_dict["head_checkpoint"]
    return config


def main(config, args):
    # 运行评估（所有实例化都在eval_model内部完成）
    print(f"\n【{args.preset}】评估开始:")
    print("=" * 50)
    avg_metrics, records = eval_model(config)

    # 输出结果
    print(f"\n【{args.preset}】评估结果:")
    print("=" * 50)
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")

    # 保存结果
    if config.args.output_dir:
        result_file = os.path.join(
            config.args.output_dir, f"{args.preset}_results_{config.args.quality}.json"
        )
        result = {
            "task": args.preset,
            "quality": config.args.quality,
            "description": config.eval_presets[args.preset].description,
            "results": avg_metrics,
            "records": records,
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {result_file}")
    return result


if __name__ == "__main__":
    """主函数"""
    parser = setup_args()
    args = parser.parse_args()

    # 加载配置
    config = OmegaConf.load(args.config[0])
    for config_path in args.config[1:]:
        overlay_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, overlay_config)

    # 将命令行参数合并到配置中
    multi_run_results = []
    config = merge_args_to_config(config, args)
    if "multi_run" in config:
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


    rows = []
    for result in multi_run_results:
        row = result["results"]
        row.update({"quality": result["quality"]})
        rows.append(row)
    combined_df = pd.DataFrame(rows)

    print(f"\n【{args.preset}】multi-run summary:")
    print("=" * 50)
    print(combined_df)
    summary_results = combined_df.to_dict(orient="list")
    final_result = {
        "task": args.preset,
        "description": config.eval_presets[args.preset].description,
        "results": summary_results,
    }
    json_path = os.path.join(config.args.output_dir, f"{args.preset}_results.json")
    with open(json_path, "w") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    print(f"Saved summary results to {json_path}")



"""
# example usage for MPC2:
python examples/mpc/run_eval_mpc.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_mpc2.yaml \
    --checkpoint "" \
    --task imagenet_sel100_cls \
    --head imagenet_cls_small_last4 \
    --quality 1.0 \
    --cuda --recon 0 --real \
    --output_dir eval_imagenet_sel100_mpc2_real

# example usage for MPC12:
python examples/mpc/run_eval_mpc.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_mpc12.yaml \
    --checkpoint "" \
    --task voc2012_sel20_seg \
    --head voc2012_seg_small_last4 \
    --quality 1.0 \
    --cuda --recon 2 --real \
    --output_dir eval_voc2012_sel20_mpc12_real

# example usage for VTM feature coding:
python examples/mpc/run_eval_mpc.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/vtm/dino_timm_patch_small_last1_vtm.yaml \
    --checkpoint "" \
    --task voc2012_sel20_seg \
    --head voc2012_seg_small_last1 \
    --quality 1.0 \
    --cuda --recon 0 --real --verbose \
    --output_dir eval_voc2012_sel20_small_last1_vtm

python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_Bypass-large-last1.yaml \
    --checkpoint "" \
    --task nyuv2_val_dep \
    --head nyuv2_dep_large_last1 \
    --cuda --recon 0  \
    --output_dir eval_nyuv2_val_dep_Bypass-large-last1

# examples usage for MPC3
python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC3-v3-large.yaml \
    --checkpoint "" \
    --task ade20k_val_seg \
    --head ade20k_seg_large_last1 \
    --cuda --recon 0  \
    --output_dir eval_ade20k_val_seg_MPC3-v3-large

python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC3-v3-large.yaml \
    --checkpoint "" \
    --task nyuv2_val_dep \
    --head nyuv2_dep_large_last1 \
    --cuda --recon 0  \
    --output_dir eval_nyuv2_val_dep_MPC3-v3-large

python examples/mpc/run_eval.py \
    --config examples/mpc/config/eval_base.yaml examples/mpc/config/eval_MPC3-v3-large-vbr.yaml \
    --checkpoint "" \
    --task ade20k_val_seg \
    --head ade20k_seg_large_last1 \
    --cuda --recon 0  \
    --output_dir eval_ade20k_val_seg_MPC3-v3-large-vbr


"""



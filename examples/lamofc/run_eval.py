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
import tqdm
import math
import warnings
import numpy as np
from PIL import Image
import pandas as pd
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor


from CoFAI.cofai.datasets import *
from CoFAI.cofai.backbone import *
from CoFAI.cofai.heads import *
from CoFAI.cofai.models import *
from CoFAI.cofai.metrics import *
from CoFAI.cofai.utils.tensor_ops import tensor2image, center_pad, center_crop
from CoFAI.cofai.utils.utils import rename_key_by_rules

from dotenv import load_dotenv

load_dotenv()

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
                    raise ValueError(
                        "orig_size is not recorded; please provide `size`."
                    )
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
def inference_x(model, x, qp=1, real=False, tasks=[]):
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

    if "load" in cfg and cfg.load:
        checkpoint = torch.load(cfg.load.path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        if cfg.load.rules:
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

    # 根据任务类型构建相应的头部和指标
    if head_config and "cls" in preset_name:
        print("构建分类头部...")
        cls_head = instantiate_class(head_config).to(device).eval()
        cls_metric = instantiate_class(metric_config)
    elif head_config and "seg" in preset_name:
        print("构建分割头部...")
        seg_head = instantiate_class(head_config).to(device).eval()
        seg_metric = instantiate_class(metric_config)

    # 创建输出目录
    if cfg.args.output_dir:
        out_sub_dir = f"{cfg.args.output_dir}/{cfg.args.quality}"
        os.makedirs(out_sub_dir, exist_ok=True)
        temp_input_dir = f"{cfg.args.output_dir}/temp_input_dir"
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        os.makedirs(temp_input_dir, exist_ok=True)

    # 评估循环
    for batch in tqdm.tqdm(dataset):
        x = batch["img"]
        img_meta = dict(batch["meta"])
        if "semseg" in batch:
            sl = np.asarray(batch["semseg"], dtype=np.int64).squeeze(-1)
            img_meta["seg_label"] = sl
        x = ToTensor()(x).to(device)
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x_orig = x.clone()

        reso_transform: ResolutionTransform = instantiate_class(
            cfg.resolution_transform
        )
        x_adapt = reso_transform.adapt(x_orig)

        time_items, bits_items, task_feats = inference_x(
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

        # 更新分类指标
        if "cls" in tasks:
            logits = cls_head.forward(task_feats["cls"])
            cls_preds = F.softmax(logits, dim=1)
            values, top_indices = torch.topk(cls_preds, k=5, dim=1)
            cls_metric.update(top_indices.cpu().tolist(), [img_meta["cls_label"]])

        # 更新分割指标
        if "seg" in tasks:
            logits = seg_head.predict(task_feats["seg"], scale=model.patch_size)
            logits = reso_transform.revert(logits)
            seg_preds = logits.argmax(dim=1).squeeze(0)
            seg_preds = seg_preds.cpu().numpy()  # [H, W]
            seg_label = img_meta["seg_label"]  # [H, W]
            seg_metric.update(seg_preds, seg_label)

        # 记录结果
        file = img_meta["img_path"]
        record = {"file": file, "quality": cfg.args.quality}
        record.update(out_result)

        metric_meter.update(record)

        if cfg.args.verbose:
            # _rv = {key: round(value, 4) for key, value in per_file_record.items()}
            print(file, record)

        # per_file_record = {key: round(value, 8) for key, value in per_file_record.items()}
        records.append(record)

    # 计算平均值
    avg_metrics = metric_meter.average()

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
    # parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument(
        "--preset", type=str, required=True, help="预定义的评估任务名称"
    )
    parser.add_argument("--head", type=str, help="头部模型名称，需要是预定义的头部模型")
    parser.add_argument("--quality", type=str, default="1.0", help="质量参数")
    parser.add_argument("--real", action="store_true", help="使用实际压缩")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--cuda", action="store_true", help="使用CUDA")
    parser.add_argument("--output_dir", type=str, default="", help="输出目录")
    return parser


def merge_args_to_config(config, args):
    """将命令行参数合并到配置中"""
    args_dict = dict(vars(args))
    args_dict["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.args = OmegaConf.create(args_dict)
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
# example usage for VTM feature coding:
CUDA_VISIBLE_DEVICES=0 python examples/lamofc/run_eval.py \
    --config examples/lamofc/config/eval_base.yaml examples/lamofc/config/dino_timm_patch_small_last1_vtm.yaml \
    --preset voc2012_sel20_seg \
    --head "voc2012_seg_small_last1" \
    --quality 1.0 \
    --cuda --output_dir eval_test --real
"""

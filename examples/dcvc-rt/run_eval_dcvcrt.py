"""
统一的评估脚本，支持从配置文件中指定任务
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import importlib
import shutil
import tqdm
import math
import warnings

from cofai.datasets import *
from cofai.backbone import *
from cofai.heads import *
from cofai.models import *
from cofai.metrics import *
from cofai.metrics.iqa_metrics import create_img_metrics, create_dist_metrics
from cofai.utils.tensor_ops import tensor2image, center_pad, center_crop
from cofai.utils.utils import rename_key_by_rules
from cofai.utils.transforms import rgb2ycbcr, ycbcr2rgb
from cofai.models.dcvcrt.video import DCVC_RT_Video
from cofai.datasets.video import VideoFolder


from cofai.metrics.utils import DictAverageMeter, DataFrameRecords
from cofai.utils.debug import extract_shapes


# Disable Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image

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


def calc_bits_items(out):
    if "strings" in out:  # real compression
        bits_items = {
            f"{name}": sum(len(s[0]) for s in sub_strings) * 8.0
            for name, sub_strings in out["strings"].items()
        }
        return bits_items
    elif "likelihoods" in out:
        bits_items = {
            f"{name}": (torch.log(likelihoods).sum() / (-math.log(2))).item()
            for name, likelihoods in out["likelihoods"].items()
        }
        return bits_items
    elif "bit_stream" in out:
        return {
            "bit_stream": len(out["bit_stream"]) * 8.0,
        }
    elif "bits" in out:
        return out["bits"]
    else:
        raise KeyError("Expected key `strings` or `likelihoods` in out_enc.")


def mpc_calc_bits_items(out):
    mpc_bits_items = {}
    if "ibranch1" in out or "ibranch2" in out or "ibranch3" in out:
        if "ibranch1" in out:
            for name, value in calc_bits_items(out["ibranch1"]).items():
                mpc_bits_items[f"i1_{name}"] = value
        if "ibranch2" in out:
            for name, value in calc_bits_items(out["ibranch2"]).items():
                mpc_bits_items[f"i2_{name}"] = value
        if "ibranch3" in out:
            for name, value in calc_bits_items(out["ibranch3"]).items():
                mpc_bits_items[f"i3_{name}"] = value
    else:
        mpc_bits_items = {"all": calc_bits_items(out)}
    return mpc_bits_items


@torch.inference_mode()
def inference_video(model, reader, codec_args):
    start = time.time()
    out_enc = model.compress_video(reader, codec_args=codec_args)
    enc_time = time.time() - start
    start = time.time()
    out_dec = model.decompress_video(**out_enc, codec_args=codec_args)
    dec_time = time.time() - start
    # print("out_enc", extract_shapes(out_enc))
    # {'bits': 849368}
    # print("out_dec", extract_shapes(out_dec))
    # [{'x_hat': (1, 3, 1080, 1920)}, {'x_hat': (1, 3, 1080, 1920)}, ...]
    bits_items = mpc_calc_bits_items(out_enc)
    time_items = {
        "enc_time": enc_time,
        "dec_time": dec_time,
    }

    return time_items, bits_items, out_dec


@torch.inference_mode()
def eval_model(cfg):
    # 从配置中获取任务信息
    task_name = cfg.args.task
    assert task_name in cfg.eval_tasks, f"任务 {task_name} 不存在"
    task_config = cfg.eval_tasks[task_name]

    print(f"任务: {task_name}")
    print(f"描述: {task_config.description}")
    print(f"数据: {task_config.dataset}")
    print(f"指标: {task_config.metric}")
    print(f"模型: {cfg.model.type}")
    print(f"头部: {cfg.args.head}")

    # 获取数据集和指标配置
    dataset_config = cfg.datasets[task_config.dataset]
    metric_config = cfg.metrics[task_config.metric]
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
    elif cfg.args.checkpoint:
        checkpoint = torch.load(
            cfg.args.checkpoint, map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    # 模型推理
    if hasattr(model, "update"):
        model.update()

    dataset_meter = DictAverageMeter()
    video_records = DataFrameRecords()

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
    if head_config and "cls" in task_name:
        print("构建分类头部...")
        cls_head = instantiate_class(head_config).to(device).eval()
        cls_metric = instantiate_class(metric_config)
    elif head_config and "seg" in task_name:
        print("构建分割头部...")
        seg_head = instantiate_class(head_config).to(device).eval()
        seg_metric = instantiate_class(metric_config)

    # 创建图像和分布指标
    print("创建图像质量指标...")
    img_metrics_dict = {}
    if cfg.args.recon != 0:
        img_metrics_dict = create_img_metrics()

    # 创建输出目录
    if cfg.args.output_dir:
        out_sub_dir = f"{cfg.args.output_dir}/{cfg.args.quality}"
        os.makedirs(out_sub_dir, exist_ok=True)
        temp_input_dir = f"{cfg.args.output_dir}/temp_input_dir"
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir)
        os.makedirs(temp_input_dir, exist_ok=True)

    # 评估循环
    records = []
    for vid_reader, vid_meta in tqdm.tqdm(dataset):
        codec_args = cfg.codec_args.copy()
        codec_args.update(vid_meta)
        time_items, bits_items, codec_out = inference_video(
            model,
            vid_reader,
            codec_args,
        )
        num_pixels = vid_meta["src_width"] * vid_meta["src_height"] * vid_meta["frame_num"]
        bpp_items = {f"bpp_{k}": v / num_pixels for k, v in bits_items.items()}
        bpp = sum(bpp_items.values())

        video_codec_result = {
            **time_items,
            "bpp": bpp,
            **bpp_items,
        }

        # # out_net 结构
        # [
        #     {"x_hat": x_hat, "cls": cls, "seg": seg, ...}, # frame 0
        # ]
        # # vid_lables 结构
        # [
        #     {"cls_label": cls_label, "seg_label": seg_label, ...}, # frame 0
        # ]

        # 逐帧计算图像质量指标，更新到 video_records
        if cfg.args.recon != 0:
            vid_reader.restart()
            for i in range(vid_meta["frame_num"]):
                x = vid_reader.read_one_frame().yuv444
                x_org = np_image_to_tensor(x, model.device)
                x_hat = codec_out[i]["x_hat"]
                frame_iqa_result = {
                    key: func(x_hat, x_org).item() for key, func in img_metrics_dict.items()
                }
                frame_iqa_result["_id"] = i
                video_records.update(frame_iqa_result)
        # print(video_records.df)
        video_iqa_result = video_records.average()


        # 逐帧计算图像质量指标
        if head_config and "cls" in task_name:
            vid_reader.restart()
            for i in range(vid_meta["frame_num"]):
                logits = cls_head.forward(codec_out[i]["cls"])
                cls_preds = F.softmax(logits, dim=1)
                values, top_indices = torch.topk(cls_preds, k=5, dim=1)
                cls_metric.update(top_indices, [vid_meta["cls_label"]])

        # 记录结果
        file = vid_meta["seq_name"]
        record = {"file": file, "quality": cfg.args.quality}
        record.update(video_codec_result)
        record.update(video_iqa_result)

        dataset_meter.update(record)

        if cfg.args.verbose:
            # _rv = {key: round(value, 4) for key, value in per_file_record.items()}
            print(file, record)

        # per_file_record = {key: round(value, 8) for key, value in per_file_record.items()}
        records.append(record)

    # 计算平均值
    avg_metrics = dataset_meter.average()

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
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--task", type=str, required=True, help="预定义的评估任务名称")
    parser.add_argument(
        "--head", type=str, required=True, help="头部模型名称，需要是预定义的头部模型"
    )
    parser.add_argument("--quality", type=str, default="12.0", help="质量参数")
    parser.add_argument("--real", action="store_true", help="使用实际压缩")
    parser.add_argument(
        "--recon", type=int, default=2, choices=[0, 1, 2, 3], help="重建层"
    )
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
    print(f"\n【{args.task}】评估开始:")
    print("=" * 50)
    avg_metrics, records = eval_model(config)

    # 输出结果
    print(f"\n【{args.task}】评估结果:")
    print("=" * 50)
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")

    # 保存结果
    if config.args.output_dir:
        result_file = os.path.join(
            config.args.output_dir, f"{args.task}_results_{config.args.quality}.json"
        )
        result = {
            "task": args.task,
            "quality": config.args.quality,
            "description": config.eval_tasks[args.task].description,
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


    import pandas as pd
    rows = []
    for result in multi_run_results:
        row = result["results"]
        row.update({"quality": result["quality"]})
        rows.append(row)
    combined_df = pd.DataFrame(rows)

    print(f"\n【{args.task}】multi-run summary:")
    print("=" * 50)
    print(combined_df)
    summary_results = combined_df.to_dict(orient="list")
    final_result = {
        "task": args.task,
        "description": config.eval_tasks[args.task].description,
        "results": summary_results,
    }
    json_path = os.path.join(config.args.output_dir, f"{args.task}_results.json")
    with open(json_path, "w") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)
    print(f"Saved summary results to {json_path}")


"""
# example usage for DCVC-RT:
python examples/dcvc-rt/run_eval_dcvcrt.py \
    --config examples/dcvc-rt/config/eval_base.yaml examples/dcvc-rt/config/eval_dcvcrt.yaml \
    --checkpoint "" \
    --task uvg_val_rec \
    --head "" \
    --quality 1.0 \
    --cuda --recon 2 --real \
    --output_dir eval_uvg_val_dcvcrt
"""

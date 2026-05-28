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
from torchvision.transforms import ToPILImage, ToTensor, Compose
import importlib
import shutil
import tqdm
import math
import warnings
import cv2
import torchvision

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cofai.datasets import *
from cofai.backbone import *
from cofai.heads import *
from cofai.models import *
from cofai.models.cavcodec import *
from cofai.utils.video_processor import *
from cofai.metrics import *
from cofai.backbone import backbone_tools as feature_fns

from cofai.metrics.iqa_metrics import (
    create_img_metrics,
    create_dist_metrics,
    split_img_metrics,
)
from cofai.utils.utils import rename_key_by_rules

from cofai.metrics.utils import DictAverageMeter, DataFrameRecords
from cofai.utils.debug import extract_shapes

from cofai.backbone.vgg import (
    setup_vgg_feature_extractor,
    extract_vgg_features,
)
REGISTRY = {
    "extract_vgg_features": extract_vgg_features,
    "setup_vgg_feature_extractor": setup_vgg_feature_extractor,
}

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

def np_image_to_tensor(img, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = Compose([
            ToTensor()
        ])
    image = transform(img)
    image = image.unsqueeze(0)
    return image

def load_png_to_tensor(img_path, device):
    img = Image.open(img_path).convert("RGB")
    tensor = ToTensor()(img).unsqueeze(0).to(device)
    return tensor

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
    bit_items = {}
    # 先计算特征的大小
    feature_total_bits = 0
    features = out["data"]["layer2"]
    # feature_bit = out["data"]["layer2"][0]["strings"]
    for i in range(len(features)):
        feature_total_bits += len(features[i]["strings"]) * 8
    # print(len(feature_bit))
    bit_items["f"] = feature_total_bits

    # 计算视频流的大小
    compress_vid_path = out["data"]["layer3"]["video"]["strings"]
    video_total_bits = os.path.getsize(compress_vid_path) * 8
    bit_items["v"] = video_total_bits
    return bit_items

def mpc_calc_bits_items(out):
    return calc_bits_items(out)

@torch.inference_mode()
def inference_video(model, reader, meta, codec_args):
    start = time.time()
    out_enc = model.compress_video(reader, meta, codec_args=codec_args)
    enc_time = time.time() - start
    start = time.time()
    out_dec = model.decompress_video(out_enc, codec_args=codec_args)
    dec_time = time.time() - start

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

    # 构建数据集
    print("构建数据集...")
    dataset = instantiate_class(dataset_config)
    print(f"数据集大小: {len(dataset)}")

    # 构建头部模型和指标
    cls_head = None
    cls_metric = None
    seg_head = None
    seg_metric = None
    det_head = None
    det_metric = None

    # 根据任务类型构建相应的头部和指标
    if head_config and "cls" in task_name:
        print("构建分类网络...")
        cls_head = instantiate_class(head_config).to(device).eval()
        cls_metric = instantiate_class(metric_config)
    elif head_config and "det" in task_name:
        print("构建检测网络...")
        seg_head = instantiate_class(head_config).to(device).eval()
        seg_metric = instantiate_class(metric_config)

    # 创建图像和分布指标
    print("创建图像质量指标...")
    img_metrics_dict = {}
    dir_metrics_dict = {}
    if cfg.args.recon != 0:
        img_metrics_dict = create_img_metrics()
        img_metrics_dict, dir_metrics_dict = split_img_metrics(img_metrics_dict)

    # 创建输出目录
    recon_dir = os.path.join(cfg.args.output_dir, "recon_frames")
    temp_input_dir = os.path.join(cfg.args.output_dir, "temp_input_dir")
    if cfg.args.output_dir:
        out_sub_dir = f"{cfg.args.output_dir}/{cfg.args.quality}"
        os.makedirs(out_sub_dir, exist_ok=True)
    if os.path.exists(temp_input_dir):
        shutil.rmtree(temp_input_dir)
    os.makedirs(temp_input_dir, exist_ok=True)
    if os.path.exists(recon_dir):
        shutil.rmtree(recon_dir)
    os.makedirs(recon_dir, exist_ok=True)

    # 评估循环
    records = []
    OFFLINE_COMPUTE = True
    frame_gt_dir = cfg.args.get("frame_gt_dir", "")
    use_external_frame_gt = bool(frame_gt_dir)

    for vid_reader, vid_meta in tqdm(dataset, desc="Processing", total=len(dataset)):
        codec_args = cfg.codec_args.copy()
        codec_args.update(vid_meta)
        time_items, bits_items, decoded_vid = inference_video(
            model,
            vid_reader,
            vid_meta,
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

        video_iqa_result = {}
        if cfg.args.recon != 0:
            video_records = DataFrameRecords()
            vid_reader = VideoReader(vid_meta["path"])
            vid_name = vid_meta["seq_name"].split(".")[0]
            missing_external_gt = 0
        
            for i, frame_bgr in enumerate(vid_reader):
                x_org = None
                if use_external_frame_gt:
                    gt_name = f"{vid_name}_{(i + 1):08d}.png"
                    gt_path = os.path.join(frame_gt_dir, gt_name)
                    if os.path.isfile(gt_path):
                        x_org = load_png_to_tensor(gt_path, model.device)
                    else:
                        missing_external_gt += 1

                if x_org is None:
                    # Fallback to source video frame when external GT is not provided
                    # or frame file is missing.
                    x_org = np_image_to_tensor(frame_bgr, model.device)
                x_hat = decoded_vid[i]["x_hat"]
                
                if getattr(cfg.args, "debug", False):
                    import IPython
                    IPython.embed()
                
                frame_iqa_result = {
                    key: func(x_hat, x_org).item()
                    for key, func in img_metrics_dict.items()
                }
                frame_iqa_result["_id"] = i
                video_records.update(frame_iqa_result)

            if use_external_frame_gt and missing_external_gt > 0:
                print(
                    f"[warn] {vid_name}: {missing_external_gt} frames missing in "
                    "--frame_gt_dir, online frame metrics fell back to source video"
                )

            video_iqa_result = video_records.average()

        # 逐帧计算detection任务指标
        if head_config and "det" in task_name:
            pass

        # 记录结果
        file = vid_meta["seq_name"]
        record = {"file": file, "quality": cfg.args.quality}
        record.update(video_codec_result)
        record.update(video_iqa_result)

        dataset_meter.update(record)

        if cfg.args.verbose:
            print(file, record)

        records.append(record)

        if True: # OFFLINE_COMPUTE
            vid_name = vid_meta['seq_name'].split('.')[0]
            save_dir = recon_dir
            os.makedirs(save_dir, exist_ok=True)
            for i in tqdm(range(len(decoded_vid))):
                save_path = os.path.join(save_dir, f"{vid_name}_{(i+1):08d}.png")
                torchvision.utils.save_image(decoded_vid[i]["x_hat"], save_path)

            gt_dir = temp_input_dir
            vid_reader = VideoReader(vid_meta["path"])
            for i, frame_bgr in enumerate(vid_reader):
                gt_path = os.path.join(gt_dir, f"{vid_name}_{(i+1):08d}.png")
                gt_tensor = np_image_to_tensor(frame_bgr, model.device)
                torchvision.utils.save_image(gt_tensor, gt_path)
            vid_reader.release()

    avg_metrics = dataset_meter.average()

    if cfg.args.recon != 0 and dir_metrics_dict:
        dir_results = {}
        for name, func in dir_metrics_dict.items():
            if name.startswith("Det-"):
                gt_dir = cfg.args.get("det_gt_dir", "")
                if not gt_dir:
                    print(f"[warn] skip {name}: --det_gt_dir is not set")
                    continue
            else:
                gt_dir = frame_gt_dir if use_external_frame_gt else temp_input_dir

            if not os.path.isdir(recon_dir) or not os.path.isdir(gt_dir):
                print(f"[warn] skip {name}: missing dirs {recon_dir} or {gt_dir}")
                continue

            result = func(recon_dir, gt_dir)
            if isinstance(result, dict):
                for k, v in result.items():
                    dir_results[f"{name}/{k}"] = v
            elif torch.is_tensor(result):
                dir_results[name] = result.item()
            else:
                dir_results[name] = float(result)
        avg_metrics.update(dir_results)

    avg_metrics = {key: round(value, 6) for key, value in avg_metrics.items()}
    return avg_metrics, records

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="MPC模型评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        nargs="+",
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
    parser.add_argument("--output_dir", type=str, default="out_cavc", help="输出目录")
    parser.add_argument(
        "--det_gt_dir",
        type=str,
        default="",
        help="Detection dir-metrics 的 GT 标签目录（.pt），用于 Det-* 指标计算",
    )
    parser.add_argument(
        "--frame_gt_dir",
        type=str,
        default="",
        help="重建帧指标（如 PSNR-Frames/LPIPS-Frames）的 GT 帧目录（.png）",
    )
    parser.add_argument("--debug", action="store_true", help="进入交互式调试（会调用 IPython.embed）")
    return parser

def merge_args_to_config(config, args):
    """将命令行参数合并到配置中"""
    args_dict = dict(vars(args))
    args_dict["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.args = OmegaConf.create(args_dict)
    return config

def main(config, args):
    print(f"\n【{args.task}】评估开始:")
    print("=" * 50)
    print(config['model']['cond_tcm'])
    
    cfg = config["model"]["cond_tcm"]

    setup_name = cfg.get("setup_vgg_feature_extractor_fn")
    extract_name = cfg.get("extract_vgg_features_fn")
    
    if isinstance(setup_name, str):
        cfg["setup_feature_extractor_fn"] = feature_fns[setup_name]
    if isinstance(extract_name, str):
        cfg["extract_feature_fn"] = feature_fns[extract_name]
  
    avg_metrics, records = eval_model(config)

    print(f"\n【{args.task}】评估结果:")
    print("=" * 50)
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")

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
    parser = setup_args()
    args = parser.parse_args()

    config = OmegaConf.load(args.config[0])
    for config_path in args.config[1:]:
        overlay_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, overlay_config)

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
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"Saved summary results to {json_path}")

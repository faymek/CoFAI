#!/usr/bin/env python3
"""
RFC/MLoRE 多任务特征压缩评估脚本

该脚本用于评估 MLoRE 模型在 PASCAL Context 和 NYUD 数据集上的
多任务压缩性能，包括：
- 任务性能指标（mIoU, F-measure, RMSE等）
- 压缩效率指标（BPP, 编解码时间）

使用方法:
    python examples/rfc/run_eval_rfc.py \
        --config examples/rfc/config/eval_base.yaml examples/rfc/config/eval_rfc_pascal.yaml \
        --checkpoint /path/to/checkpoint.pth \
        --task pascal_multitask \
        --cuda

框架规范: https://faymek.github.io/CoFAI/framework/
"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from cofai.models import MLoREFrameCodec, MLoREVideoCodec
from cofai.datasets import PASCALContextDataset, NYUDDataset, collate_mlore
from cofai.transforms import get_mlore_transforms
from cofai.losses.mlore_loss import MLoRECodingLoss
from cofai.utils.tensor_ops import center_pad
from cofai.utils.rfc_utils import center_crop
# Disable warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# torch.backends.cudnn.deterministic = True
# torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


class DictAverageMeter:
    """用于计算字典值平均值的工具类"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum_dict = {}
        self.count = 0
    
    def update(self, val_dict: Dict[str, float]):
        self.count += 1
        for key, val in val_dict.items():
            if isinstance(val, (int, float)):
                if key not in self.sum_dict:
                    self.sum_dict[key] = 0.0
                self.sum_dict[key] += val
    
    def average(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        return {key: val / self.count for key, val in self.sum_dict.items()}


def calc_bits_from_strings(strings: Dict[str, List[List[bytes]]]) -> Dict[str, float]:
    """从压缩码流计算比特数"""
    bits_items = {}
    
    # 处理RFC格式（列表）
    if isinstance(strings, list):
        if len(strings) == 2:
            # RFC format: [y_strings, z_strings]
            y_strings, z_strings = strings
            bits_items["y"] = sum(len(s) if isinstance(s, bytes) else len(s[0]) for s in (y_strings if isinstance(y_strings, list) else [y_strings])) * 8.0
            bits_items["z"] = sum(len(s) if isinstance(s, bytes) else len(s[0]) for s in (z_strings if isinstance(z_strings, list) else [z_strings])) * 8.0
        else:
            raise ValueError(f"Unexpected strings list length: {len(strings)}")
    # 处理CoFAI格式（字典）
    elif isinstance(strings, dict):
        for name, sub_strings in strings.items():
            bits = sum(len(s[0]) if isinstance(s, list) else len(s) for s in sub_strings) * 8.0
            bits_items[name] = bits
    else:
        raise ValueError(f"Unexpected strings type: {type(strings)}")
    
    return bits_items


def calc_bits_from_likelihoods(likelihoods: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """从似然值估计比特数"""
    bits_items = {}
    for name, lh in likelihoods.items():
        bits = (torch.log(lh).sum() / (-math.log(2))).item()
        bits_items[name] = bits
    return bits_items



def get_output(output, task, p=None, label=None, semseg_save_train_class=True):
    
    if task == 'normals':
        output = output.permute(0, 2, 3, 1)
        output = (F.normalize(output, p = 2, dim = 3) + 1.0) * 255 / 2.0
    
    elif task in {'semseg'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)

    elif task in {'human_parts'}:
        output = output.permute(0, 2, 3, 1)
        _, output = torch.max(output, dim=3)
    
    elif task in {'edge'}:
        output = output.permute(0, 2, 3, 1)
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)), dim=3)

    elif task in {'sal'}:
        output = output.permute(0, 2, 3, 1)
        output = F.softmax(output, dim=3)[:, :, :, 1] *255 # torch.squeeze(255 * 1 / (1 + torch.exp(-output)))
    
    elif task in {'depth'}:
        output.clamp_(min=0.)
        output = output.permute(0, 2, 3, 1)
    
    elif task in {'scene'}:
        _, output = torch.max(output, dim=1)
    
    else:
        raise ValueError('Select one of the valid tasks')

    return output


def instantiate_dataset(config: Dict) -> torch.utils.data.Dataset:
    """根据配置实例化数据集"""
    config = dict(config)
    dataset_type = config.pop('type')
    
    if dataset_type == 'PASCALContextDataset':
        return PASCALContextDataset(**config)
    elif dataset_type == 'NYUDDataset':
        return NYUDDataset(**config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def instantiate_model(config: Dict, eval_tasks=None, device: str = 'cuda') -> MLoREFrameCodec:
    """根据配置实例化模型"""
    config = dict(config)
    model_type = config.pop('type')
    
    # 构建配置对象
    from cofai.backbone.mlore import create_mlore_config
    
    tasks_config = config.get('tasks', {})
    # 如果指定了评估任务，使用评估任务列表；否则使用配置中的任务列表
    if eval_tasks is not None:
        task_names = eval_tasks
    else:
        task_names = tasks_config.get('NAMES', ['semseg', 'edge', 'normals', 'sal', 'human_parts'])
    
    p = create_mlore_config(
        tasks=task_names,  # 使用实际需要的任务列表
        stage=config.get('stage', 'stage1'),
        img_size=tuple(config.get('img_size', [512, 512])),
        final_embed_dim=config.get('final_embed_dim', 640),
        rank_list=config.get('rank_list'),
    )
    
    # 更新任务输出维度
    if 'NUM_OUTPUT' in tasks_config:
        p.TASKS.NUM_OUTPUT.update(tasks_config['NUM_OUTPUT'])
    
        

    
    if model_type == 'MLoREFrameCodec':
        model = MLoREFrameCodec(
            p=p,
            stage=config.get('stage', 'stage2'),
            pretrained=config.get('pretrained', True),
            img_size=tuple(config.get('img_size', [512, 512])),
            drop_path_rate=config.get('drop_path_rate', 0.15),
            device=device,
        )
    elif model_type == 'MLoREVideoCodec':
        model = MLoREVideoCodec(
            p=p,
            stage=config.get('stage', 'stage2'),
            pretrained=config.get('pretrained', True),
            img_size=tuple(config.get('img_size', [512, 512])),
            device=device,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def get_performance_meter(task: str, p: Dict) -> Any:
    """RFC task meter backed by ``cofai.metrics`` (migrated from examples/rfc/evaluation)."""
    database = p.get('train_db_name', 'PASCALContext')
    ignore_index = p.get('ignore_index', 255)

    if task == 'semseg':
        from cofai.metrics import SemanticSegmentationMeter

        return SemanticSegmentationMeter(database, ignore_idx=ignore_index)
    if task == 'human_parts':
        from cofai.metrics import HumanPartSegmentationMeter

        return HumanPartSegmentationMeter(database, ignore_idx=ignore_index)
    if task == 'edge':
        from cofai.metrics import EdgeDetectionMeter

        edge_w = p.get('edge_w', 0.95)
        return EdgeDetectionMeter(pos_weight=edge_w, ignore_index=ignore_index)
    if task == 'normals':
        from cofai.metrics import SurfaceNormalsEstimationMeter

        return SurfaceNormalsEstimationMeter(ignore_index=ignore_index)
    if task == 'sal':
        from cofai.metrics import SaliencyDetectionMeter

        return SaliencyDetectionMeter(
            ignore_index=ignore_index, threshold_step=0.05, beta_squared=0.3
        )
    if task == 'depth':
        from cofai.metrics import DepthEstimationMeter

        tasks_cfg = p.get('TASKS') or {}
        max_depth = tasks_cfg.get('depth_max', 10.0)
        min_depth = tasks_cfg.get('depth_min', 0.001)
        return DepthEstimationMeter(max_depth=max_depth, min_depth=min_depth)
    if task == 'scene':
        from cofai.metrics import SceneClassificationMeter

        return SceneClassificationMeter(database)
    return None


@torch.no_grad()
def inference_compress_decompress(
    model: MLoREFrameCodec,
    x: torch.Tensor,
    tasks: List[str],
    real: bool = False,
) -> Dict[str, Any]:
    """
    执行压缩-解压缩推理
    
    Args:
        model: MLoRE模型
        x: 输入图像张量
        tasks: 任务列表
        real: 是否执行实际压缩（写入码流）
        
    Returns:
        结果字典，包含任务输出、时间和比特信息
    """
    result = {}
    
    if real:
        # 实际压缩
        start_time = time.time()
        coded_unit = model.compress(x, tasks=tasks)
        enc_time = time.time() - start_time
        
        start_time = time.time()
        task_outputs = model.decompress(coded_unit, tasks=tasks)
        dec_time = time.time() - start_time
        
        bits_items = calc_bits_from_strings(coded_unit["strings"])
        result["enc_time"] = enc_time
        result["dec_time"] = dec_time
    else:
        # 码率估计模式 - 使用forward方法，与原始RFC一致
        # 原始RFC: model.module(images, batch=batch) - 不传递episode_tasks
        start_time = time.time()
        # 使用forward而不是forward_test，传递tasks但不传递episode_tasks（使用默认值None）
        # 注意：原始RFC的forward不接收tasks参数，但CoFAI版本需要
        out = model.forward(x, tasks=tasks, episode_tasks=None)
        elapsed_time = time.time() - start_time
        
        # 过滤任务输出
        task_outputs = {k: v for k, v in out.items() if k in tasks}
        # 保留raw输出（包含bpp_loss/mse_loss等），用于对齐RFC的打印/统计
        result["raw_out"] = out
        
        # 计算bits（原始RFC使用bpp_loss）
        bits_items = {}
        if "bpp_loss" in out:
            num_pixels = x.size(0) * x.size(2) * x.size(3)
            bits_items["estimated"] = out["bpp_loss"].item() * num_pixels
        
        result["enc_time"] = elapsed_time / 2.0
        result["dec_time"] = elapsed_time / 2.0
    
    result["task_outputs"] = task_outputs
    result["bits_items"] = bits_items
    
    return result


@torch.no_grad()
def eval_model(cfg: OmegaConf) -> tuple:
    """
    评估模型
    
    Args:
        cfg: 配置对象
        
    Returns:
        avg_metrics: 平均指标
        records: 每张图像的记录
    """
    args = cfg.args
    task_name = args.task
    
    # 获取任务配置
    if task_name not in cfg.eval_tasks:
        raise ValueError(f"Task {task_name} not found in eval_tasks")
    task_config = cfg.eval_tasks[task_name]
    tasks = list(task_config.tasks)

    tasks.sort()
    
    print(f"任务: {task_name}")
    print(f"描述: {task_config.description}")
    print(f"数据集: {task_config.dataset}")
    print(f"评估任务: {task_config.tasks}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 实例化模型
    print("加载模型...")
    model = instantiate_model(cfg.model,eval_tasks=tasks, device=str(device))
    
    # 加载检查点
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint, strict=False)
    elif "load" in cfg and cfg.load.path:
        model.load_checkpoint(cfg.load.path, strict=cfg.load.get("strict", False))
    
    model.eval()
    if hasattr(model, 'update'):
        model.update()
    
    # 获取数据集配置
    dataset_config = dict(cfg.datasets[task_config.dataset])
    # tasks已在前面排序，这里不要重新获取
    dataset_config['tasks'] = tasks

    # ===== 保持与原始RFC一致：使用RFC同款 transforms 对 image + labels 一起处理 =====
    # 原始RFC的评估数据是通过 transforms.Normalize + PadImage + AddIgnoreRegions + ToTensor 得到的。
    if 'transform' not in dataset_config or dataset_config.get('transform') is None:
        from cofai.transforms import get_mlore_transforms
        # 优先使用模型自身的配置（create_mlore_config生成），保证TEST.SCALE一致
        p_for_tf = getattr(model, 'p', None)
        dataset_config['transform'] = get_mlore_transforms(p_for_tf, split='val')
    
    # 实例化数据集
    print("加载数据集...")
    dataset = instantiate_dataset(dataset_config)
    print(f"数据集大小: {len(dataset)}")
    
    # 创建DataLoader以支持batch处理（与原始RFC一致）
    batch_size = getattr(args, "batch_size", 6)
    num_workers = getattr(args, "num_workers", 4)
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_mlore
    )
    print(f"使用batch_size={batch_size}, num_workers={num_workers}")
    
    # 创建性能评估器
    task_meters = {}
    p_config = OmegaConf.to_container(cfg, resolve=True)
    for task in tasks:
        meter = get_performance_meter(task, p_config)
        if meter is not None:
            task_meters[task] = meter
    
    # 评估循环
    metric_meter = DictAverageMeter()
    records = []
    
    # 注意：当dataset启用RFC transforms 后，image/gt 已完成Normalize/Pad/ToTensor，这里不再做额外padding

    # Optional: limit number of samples for quick debug/parity check
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None and max_samples <= 0:
        max_samples = None

    # # Optional: RFC-style loss report (edge/bpp/mse) based on raw logits
    report_rfc_loss = bool(getattr(args, "report_rfc_loss", False))
    if report_rfc_loss:
        from cofai.losses.loss_functions import BalancedBinaryCrossEntropyLoss
        ignore_index = int(p_config.get("ignore_index", 255))
        edge_w = float(p_config.get("edge_w", 0.95))
        edge_crit = BalancedBinaryCrossEntropyLoss(pos_weight=edge_w, ignore_index=ignore_index).to(device)
        sum_edge_loss = 0.0
        sum_bpp_loss = 0.0
        sum_mse_loss = 0.0
        loss_count = 0
    
    # 使用batch处理（与原始RFC完全一致）
    # 原始RFC: for i, batch in enumerate(tqdm(test_loader)):
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if max_samples is not None and batch_idx * batch_size >= max_samples:
            break
        
        # Forward pass（与原始RFC一致）
        # Original RFC used ``batch['image']``; ``collate_mlore`` stacks under ``img``.
        images = batch["img"].to(device, non_blocking=True)
        targets = {task: batch[task].to(device, non_blocking=True) for task in tasks if task in batch}
        metas = batch.get('meta', [])
        
        # 推理（batch处理，与原始RFC一致）
        # 原始RFC: output = model.module(images, batch=batch)
        result = inference_compress_decompress(
            model, images, tasks, real=args.real
        )
        
        # 计算BPP（batch处理）
        batch_size_actual = images.size(0)
        num_pixels = images.size(2) * images.size(3)  # H * W
        total_bits = sum(result["bits_items"].values())
        # 对于batch，bits_items是总的，需要除以batch_size
        bpp_per_sample = (total_bits / batch_size_actual) / num_pixels
        
        # 更新任务指标（与原始RFC完全一致：对整个batch处理）
        # 原始RFC: performance_meter.update({t: get_output(output[t], t) for t in two_d_tasks}, 
        #                                    {t: targets[t] for t in two_d_tasks})
        task_outputs = result["task_outputs"]
        pred_dict = {}
        gt_dict = {}
        for task in tasks:
            if task in task_outputs and task in targets:
                # 对整个batch调用get_output（与原始RFC一致）
                pred_dict[task] = get_output(task_outputs[task], task)
                gt_dict[task] = targets[task]
        
        # 更新评估器（与原始RFC一致：整个batch一起处理，保持GPU计算）
        for task in tasks:
            if task in pred_dict and task in task_meters:
                task_meters[task].update(pred_dict[task], gt_dict[task])
        
        # RFC-style loss report: compute on raw logits (before get_output)
        # if report_rfc_loss and not args.real:
        #     raw_out = result.get("raw_out", {})
        #     if isinstance(raw_out, dict):
        #         # batch处理：loss是batch的平均值，需要乘以batch_size来累加
        #         if "bpp_loss" in raw_out:
        #             sum_bpp_loss += float(raw_out["bpp_loss"].detach().cpu().item()) * batch_size_actual
        #         if "mse_loss" in raw_out:
        #             sum_mse_loss += float(raw_out["mse_loss"].detach().cpu().item()) * batch_size_actual
        #         if "edge" in tasks and "edge" in raw_out and "edge" in targets:
        #             gt_edge = targets["edge"]
        #             pred_edge = raw_out["edge"]
        #             # 统一到(B, C, H, W)格式（batch处理）
        #             if gt_edge.dim() == 3:
        #                 gt_edge = gt_edge.unsqueeze(1)  # (B, 1, H, W)
        #             gt_edge = gt_edge.to(device).float()

        #             # RFC的PadImage会对label做中心padding到TEST.SCALE，这里做同样的对齐
        #             pred_h, pred_w = int(pred_edge.shape[-2]), int(pred_edge.shape[-1])
        #             gt_h, gt_w = int(gt_edge.shape[-2]), int(gt_edge.shape[-1])
        #             if (gt_h, gt_w) != (pred_h, pred_w):
        #                 # center pad (or crop if gt is larger)
        #                 if gt_h < pred_h or gt_w < pred_w:
        #                     pad_h = max(pred_h - gt_h, 0)
        #                     pad_w = max(pred_w - gt_w, 0)
        #                     pad_top = pad_h // 2
        #                     pad_bottom = pad_h - pad_top
        #                     pad_left = pad_w // 2
        #                     pad_right = pad_w - pad_left
        #                     gt_edge = F.pad(
        #                         gt_edge,
        #                         (pad_left, pad_right, pad_top, pad_bottom),
        #                         mode="constant",
        #                         value=255.0,  # edge ignore_index
        #                     )
        #                 # if gt is larger, center crop to pred size
        #                 if gt_edge.shape[-2] > pred_h or gt_edge.shape[-1] > pred_w:
        #                     dh = gt_edge.shape[-2] - pred_h
        #                     dw = gt_edge.shape[-1] - pred_w
        #                     top = dh // 2
        #                     left = dw // 2
        #                     gt_edge = gt_edge[..., top:top + pred_h, left:left + pred_w]

        #             # batch处理：计算整个batch的loss
        #             batch_edge_loss = edge_crit(pred_edge, gt_edge).detach().cpu().item()
        #             sum_edge_loss += batch_edge_loss * batch_size_actual
        #             loss_count += batch_size_actual
        
        # 记录每个样本的结果（用于详细输出）
        for i in range(batch_size_actual):
            out_result = {
                "enc_time": result["enc_time"] / batch_size_actual,
                "dec_time": result["dec_time"] / batch_size_actual,
                "bpp": bpp_per_sample,
            }
            
            # 计算每个码流的BPP
            for name, bits in result["bits_items"].items():
                out_result[f"bpp_{name}"] = (bits / batch_size_actual) / num_pixels
            
            # 记录结果
            img_name = metas[i].get('img_name', f'batch_{batch_idx}_sample_{i}') if isinstance(metas, list) and i < len(metas) else f'batch_{batch_idx}_sample_{i}'
            record = {"file": img_name, **out_result}
            metric_meter.update(out_result)
            
            if args.verbose:
                print(f"{img_name}: {record}")
            
            records.append(record)
    
    # 计算平均指标
    avg_metrics = metric_meter.average()
    
    # 添加任务指标
    for task, meter in task_meters.items():
        task_results = meter.compute()
        if isinstance(task_results, dict):
            for key, val in task_results.items():
                avg_metrics[f"{task}_{key}"] = val
        else:
            avg_metrics[task] = task_results
    
    # 格式化结果
    avg_metrics = {key: round(val, 6) if isinstance(val, float) else val 
                   for key, val in avg_metrics.items()}

    # if report_rfc_loss:
    #     denom = max(loss_count, 1)
    #     avg_metrics["rfc_edge_loss"] = round(sum_edge_loss / denom, 6)
    #     n_seen = max(len(records), 1)
    #     avg_metrics["rfc_bpp_loss"] = round(sum_bpp_loss / n_seen, 6)
    #     avg_metrics["rfc_mse_loss"] = round(sum_mse_loss / n_seen, 6)
    
    return avg_metrics, records


def setup_args() -> argparse.ArgumentParser:
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description="RFC/MLoRE 多任务特征压缩评估脚本"
    )
    parser.add_argument(
        "--config", type=str, required=True, nargs="+",
        help="配置文件路径，可以指定多个文件叠加"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="",
        help="模型检查点路径"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="评估任务名称（需与配置文件中定义的一致）"
    )
    parser.add_argument(
        "--real", action="store_true",
        help="使用实际熵编码（写入码流）"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="使用CUDA"
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="输出目录"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="最多评估多少张样本（用于快速对齐/调试）"
    )
    parser.add_argument(
        "--report_rfc_loss", action="store_true",
        help="额外统计并输出RFC风格的bpp_loss/mse_loss与edge loss（基于raw logits；仅在--real为False时生效）,需手动修改eval，取消注释相关代码（仅适配了Edge）"
    )
    parser.add_argument(
        "--batch_size", type=int, default=6,
        help="评估时的batch size（默认6，与原始RFC的valBatch一致，可调整以加速评估）"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="数据加载的worker数量（默认4，与原始RFC一致）"
    )
    return parser


def merge_args_to_config(config: OmegaConf, args: argparse.Namespace) -> OmegaConf:
    """将命令行参数合并到配置"""
    args_dict = dict(vars(args))
    args_dict["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.args = OmegaConf.create(args_dict)
    return config


def main():
    """主函数"""
    parser = setup_args()
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config[0])
    for config_path in args.config[1:]:
        overlay_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, overlay_config)
    
    # 合并命令行参数
    config = merge_args_to_config(config, args)
    
    # 运行评估
    print(f"\n【{args.task}】评估开始:")
    print("=" * 60)
    
    avg_metrics, records = eval_model(config)
    
    # 输出结果
    print(f"\n【{args.task}】评估结果:")
    print("=" * 60)
    for key, value in sorted(avg_metrics.items()):
        print(f"  {key}: {value}")
    
    # 保存结果
    output_dir = args.output_dir or config.args.get("output_dir", "")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, f"{args.task}_results.json")
        result = {
            "task": args.task,
            "description": config.eval_tasks[args.task].description,
            "results": avg_metrics,
            "records": records,
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main()


"""
使用示例:

# PASCAL Context 多任务评估
python examples/rfc/run_eval_rfc.py \
    --config examples/rfc/config/eval_base.yaml examples/rfc/config/eval_rfc_pascal.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --task pascal_multitask \
    --cuda --real \
    --output_dir ./eval_results/pascal

# PASCAL Context 语义分割评估
python examples/rfc/run_eval_rfc.py \
    --config examples/rfc/config/eval_base.yaml examples/rfc/config/eval_rfc_pascal.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --task pascal_semseg \
    --cuda --real --verbose \
    --output_dir ./eval_results/pascal_semseg

# NYUD 多任务评估
python examples/rfc/run_eval_rfc.py \
    --config examples/rfc/config/eval_base.yaml examples/rfc/config/eval_rfc_nyud.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --task nyud_multitask \
    --cuda --real \
    --output_dir ./eval_results/nyud
"""





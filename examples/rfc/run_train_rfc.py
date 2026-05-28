#!/usr/bin/env python3
"""
RFC/MLoRE 多任务特征压缩训练脚本

该脚本用于训练 MLoRE 模型，支持三阶段训练流程：
- Stage 0: 无压缩多任务预训练
- Stage 1: 压缩模块训练
- Stage 2: Mona Adapter 微调

支持分布式训练（DDP）和多种数据集。

使用方法:
    # 单卡训练
    python examples/rfc/run_train_rfc.py \
        --config examples/rfc/config/train_stage1.yaml \
        --run_mode train

    # 多卡分布式训练
    torchrun --nproc_per_node=4 examples/rfc/run_train_rfc.py \
        --config examples/rfc/config/train_stage1.yaml \
        --run_mode train

框架规范: https://faymek.github.io/CoFAI/framework/
"""

import argparse
import datetime
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add RFC directory to path
RFC_ROOT = PROJECT_ROOT / "RFC"
sys.path.insert(0, str(RFC_ROOT))

from utils.utils import mkdir_if_missing
from utils.config import create_config
from utils.common_config import (
    get_train_dataset, get_transformations,
    get_test_dataset, get_train_dataloader, get_test_dataloader,
    get_optimizer, get_model, get_criterion
)
from utils.logger import Logger
from utils.test_utils import test_phase
from CoFAI.examples.rfc.evaluation.evaluate_utils import PerformanceMeter


# 设置环境
torch.backends.cudnn.benchmark = True
cv2.setNumThreads(0)


def set_seed(seed: int = 0):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=datetime.timedelta(hours=2)
        )
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_pretrained_encoder(model: nn.Module, checkpoint_path: str, p: Dict):
    """加载预训练编码器"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("未找到预训练编码器，使用ImageNet预训练权重")
        return
    
    print(f"加载预训练编码器: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint.get('model', checkpoint)
    
    # 过滤掉不需要加载的键
    keys_to_remove = []
    for k in list(state_dict.keys()):
        if 'MLoRE' in k or '.heads.' in k or 'task_mask' in k or 'fea_fuse' in k:
            keys_to_remove.append(k)
    
    for k in keys_to_remove:
        state_dict.pop(k)
    
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"加载完成: {msg}")


def load_checkpoint(model: nn.Module, optimizer, checkpoint_path: str) -> tuple:
    """加载检查点"""
    start_epoch = 0
    iter_count = 0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"模型加载: {msg}")
        
        if 'optimizer' in checkpoint and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"优化器加载失败: {e}")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'iter_count' in checkpoint:
            iter_count = checkpoint['iter_count']
    
    return start_epoch, iter_count


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    iter_count: int,
    save_path: str,
    is_best: bool = False
):
    """保存检查点"""
    state = {
        'model': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'epoch': epoch,
        'iter_count': iter_count,
    }
    
    torch.save(state, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth.tar', '_best.pth.tar')
        shutil.copyfile(save_path, best_path)


def train_epoch(
    p: Dict,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    tb_writer_train: Optional[SummaryWriter],
    tb_writer_test: Optional[SummaryWriter],
    iter_count: int,
) -> tuple:
    """
    训练一个epoch
    
    Returns:
        end_signal: 是否结束训练
        iter_count: 更新后的迭代计数
    """
    model.train()
    local_rank = args.local_rank
    
    # 根据stage选择训练函数
    if 'coding' in p.model and 'baseline' not in p.model:
        from utils.train_coding_utils import train_phase
    else:
        from utils.train_utils import train_phase
    
    end_signal, iter_count = train_phase(
        p, args, train_loader, val_loader,
        model, criterion, optimizer, scheduler,
        epoch, tb_writer_train, tb_writer_test, iter_count
    )
    
    return end_signal, iter_count


def evaluate(
    p: Dict,
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    epoch: int,
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    
    eval_results = test_phase(p, val_loader, model, criterion, epoch)
    
    return eval_results


def main():
    """主函数"""
    # 解析参数
    parser = argparse.ArgumentParser(description='RFC/MLoRE Training')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--local_rank', default=0, type=int, help='分布式训练本地rank')
    parser.add_argument('--run_mode', default='train', choices=['train', 'infer'], help='运行模式')
    parser.add_argument('--trained_model', default=None, help='推理时使用的模型路径')
    args = parser.parse_args()
    
    # 设置分布式环境
    rank, world_size, local_rank = setup_distributed()
    args.local_rank = local_rank
    
    # 设置随机种子
    set_seed(0)
    
    # 创建配置
    params = {'run_mode': args.run_mode}
    p = create_config(args.config, params)
    
    # 创建输出目录
    if rank == 0:
        mkdir_if_missing(p['output_dir'])
        sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
        print(f"配置: {p}")
        shutil.copyfile(args.config, os.path.join(p['output_dir'], 'config.yml'))
    
    # 设置TensorBoard
    tb_log_dir = os.path.join(p.root_dir, 'tb_dir')
    p.tb_log_dir = tb_log_dir
    
    if rank == 0:
        train_tb_log_dir = os.path.join(tb_log_dir, 'train')
        test_tb_log_dir = os.path.join(tb_log_dir, 'test')
        if args.run_mode != 'infer':
            mkdir_if_missing(tb_log_dir)
            mkdir_if_missing(train_tb_log_dir)
            mkdir_if_missing(test_tb_log_dir)
        tb_writer_train = SummaryWriter(train_tb_log_dir)
        tb_writer_test = SummaryWriter(test_tb_log_dir)
        print(f"TensorBoard 目录: {tb_log_dir}")
    else:
        tb_writer_train = None
        tb_writer_test = None
    
    # 创建模型
    print(f"[Rank {rank}] 创建模型...")
    model = get_model(p)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    
    # 创建损失函数
    criterion = get_criterion(p).cuda()
    
    # 初始化性能评估器
    PerformanceMeter(p, [t for t in p.TASKS.NAMES])
    
    # 获取数据变换
    train_transforms, val_transforms = get_transformations(p)
    
    # 创建数据加载器
    if args.run_mode != 'infer':
        train_dataset = get_train_dataset(p, train_transforms)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, drop_last=True
        ) if world_size > 1 else None
        train_loader = get_train_dataloader(p, train_dataset, train_sampler)
    else:
        train_loader = None
        train_sampler = None
    
    test_dataset = get_test_dataset(p, val_transforms)
    test_loader = get_test_dataloader(p, test_dataset)
    
    # 加载预训练编码器
    if p.get('load_multitask_encoder'):
        load_pretrained_encoder(model, p.load_multitask_encoder, p)
    
    # 训练模式
    if args.run_mode == 'train':
        # 创建优化器和调度器
        scheduler, optimizer = get_optimizer(p, model)
        
        # 加载检查点
        start_epoch, iter_count = load_checkpoint(
            model, optimizer,
            p.get('checkpoint') if args.trained_model is None else args.trained_model
        )
        
        if rank == 0:
            print(f"从 epoch {start_epoch}, iter {iter_count} 开始训练")
        
        # 根据stage选择训练函数
        if 'coding' in p.model and 'baseline' not in p.model:
            from utils.train_coding_utils import train_phase
        else:
            from utils.train_utils import train_phase
        
        # 训练循环
        start_time = time.time()
        
        for epoch in range(start_epoch, p['epochs']):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{p['epochs']}")
                print("-" * 40)
            
            end_signal, iter_count = train_phase(
                p, args, train_loader, test_loader,
                model, criterion, optimizer, scheduler,
                epoch, tb_writer_train, tb_writer_test, iter_count
            )
            
            if end_signal:
                break
        
        if rank == 0:
            elapsed_time = (time.time() - start_time) / 3600
            print(f"\n训练完成，总耗时: {elapsed_time:.2f} 小时")
    
    # 推理模式
    elif args.run_mode == 'infer':
        if rank == 0:
            if args.trained_model is None:
                raise ValueError("推理模式需要指定 --trained_model 参数")
            
            # 加载模型
            checkpoint = torch.load(args.trained_model, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=True)
            
            # 如果有重参数化方法
            if hasattr(model, 'module'):
                if hasattr(model.module.backbone, 'reparameter'):
                    model.module.backbone.reparameter()
            else:
                if hasattr(model.backbone, 'reparameter'):
                    model.backbone.reparameter()
            
            # 评估
            eval_results = test_phase(p, test_loader, model, criterion, 0)
            
            print("\n推理结果:")
            print("=" * 40)
            print(eval_results)
    
    # 清理
    cleanup_distributed()


if __name__ == "__main__":
    main()


"""
使用示例:

# Stage 0: 无压缩预训练（单卡）
python examples/rfc/run_train_rfc.py \
    --config examples/rfc/config/train_stage0.yaml \
    --run_mode train

# Stage 1: 压缩训练（4卡分布式）
torchrun --nproc_per_node=4 examples/rfc/run_train_rfc.py \
    --config examples/rfc/config/train_stage1.yaml \
    --run_mode train

# Stage 2: Mona微调（4卡分布式）
torchrun --nproc_per_node=4 examples/rfc/run_train_rfc.py \
    --config examples/rfc/config/train_stage2.yaml \
    --run_mode train

# 推理评估
python examples/rfc/run_train_rfc.py \
    --config examples/rfc/config/train_stage1.yaml \
    --run_mode infer \
    --trained_model /path/to/checkpoint.pth.tar
"""





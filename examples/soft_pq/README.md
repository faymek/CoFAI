# Soft-PQ (可微分乘积量化) 集成

## 在线评测（Engine 模式）

通过 `cofai.engine.run_eval` 管道进行端到端评测，无需手动提取特征。

### 配置

1. 权重路径：`weights/orfc_2446/{backbone}/` 下的 `.pt` 文件
2. Plan 配置：`conf/plan/dinov2/` 下的 SoftPQ 系列 YAML
3. 环境变量：确保 `PROJECT_ROOT` 指向 CoFAI 根目录

### 单次评测

```bash
# 分类 (ImageNet sel500)
CUDA_VISIBLE_DEVICES=0 python examples/soft_pq/run_eval_soft_pq.py \
    --backbone dinov2_vitl14 --layer blk20 --task cls --multi-run --cuda

# 分割 (VOC2012 sel100)
CUDA_VISIBLE_DEVICES=0 python examples/soft_pq/run_eval_soft_pq.py \
    --backbone dinov2_vitg14 --layer blk29 --task seg --multi-run --cuda
```

### 批量评测（多卡并行）

```bash
# 指定 GPU，同时评测所有 backbone/layer/task
GPU_IDS=4,5,6,7 NUM_GPUS=4 bash examples/soft_pq/scripts/run_online_eval_all.sh
```

结果保存在 `eval_results/soft_pq_online/` 下。

---

## 离线模式（Offline）

离线模式需预先提取特征到 `features/orfc/` 目录。

### 训练

```bash
# 单次训练
CUDA_VISIBLE_DEVICES=0 python examples/soft_pq/offline/train_soft_pq.py \
    --backbone dinov2_vitl14 --layer blk20 \
    --K 16 --embedding_dim 32 --lmbda 0.5 --epochs 100

# 批量训练（多卡）
GPU_IDS=4,5,6,7 NUM_GPUS=4 bash examples/soft_pq/scripts/train_all.sh
```

### 离线测试

```bash
# 分类
python examples/soft_pq/offline/test_cls.py \
    --backbone dinov2_vitl14 --layer blk20 \
    --ckpt weights/orfc_2446/dinov2_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt

# 分割
python examples/soft_pq/offline/test_seg.py \
    --backbone dinov2_vitl14 --layer blk20 \
    --ckpt weights/orfc_2446/dinov2_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt
```

注意：完整 mIoU 评测需使用在线模式（Engine），离线模式仅测量 BPFP。

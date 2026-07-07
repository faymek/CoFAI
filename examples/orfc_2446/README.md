# ORFC-2446 — Soft-PQ (Differentiable Product Quantization)

## 目录结构

```
examples/orfc_2446/
├── run_eval_orfc_2446.py            # 在线模式入口（Engine 评测）
├── offline/
│   ├── train_soft_pq.py             # 离线训练（frozen-tail ΔL_ref + 可选 rate loss）
│   ├── test_cls.py                  # 离线分类测试（加载 codec, 评测 Acc@1 + rate）
│   ├── test_seg.py                  # 离线分割测试（加载 codec, 评测 mIoU + rate）
│   ├── weights_paths.py             # backbone → 权重子目录映射
│   ├── extract_features.py          # 特征提取 → orfc/offline/extract_features.py
│   ├── utils.py                     # 工具函数 → orfc/offline/utils.py
│   ├── backbone/
│   │   └── wrapper.py               # Backbone 封装 → orfc/offline/backbone/wrapper.py
│   └── cfg/                         # 配置文件 → orfc/offline/cfg/
└── scripts/
    ├── run_online_eval_all.sh       # 批量在线评测（对照 CSV Ours 配置）
    ├── train_all.sh                 # 批量训练（多 GPU 并行）
    ├── test_cls_all.sh              # 批量分类测试
    └── test_seg_all.sh              # 批量分割测试
```

核心算法：`cofai/entropy_models/soft_pq.py`

## 与 ORFC 的区别

| 项目 | ORFC (`examples/orfc`) | Soft-PQ (`examples/orfc_2446`) |
|------|------------------------|-------------------------------|
| 算法 | OPQ 交替优化（无梯度） | 可微分 PQ + 正交变换（Cayley 参数化）|
| 损失函数 | MSE 重建误差 | ΔL_ref (frozen-tail) + λ·Rate |
| 权重格式 | `.npz` (R + codebooks + pmf) | `.pt` (FeatureCodec state_dict) |
| 额外参数 | — | λ (rate), τ (temperature), lr, epochs |
| 权重路径 | `weights/orfc/{backbone}/` | `weights/orfc_2446/{backbone}_ori/`（ViT-L/G 均为 `dinov2_vitl14_ori/`、`dinov2_vitg14_ori/`） |

## 前置依赖

1. **Pre-extracted features**: 与 ORFC 共用同一套 features，位于 `features/orfc/`
2. **Backbone 权重**: 训练时需要 backbone（用于构建 frozen tail）
3. **Python 环境**: 需要 `cofai` 包及其依赖（torch, numpy, compressai 等）

## 训练

### 单个配置

```bash
cd examples/orfc_2446/offline

# DINOv2 ViT-L/14, blk10, K=64, 标准配置
python train_soft_pq.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --K 64 --embedding_dim 32 \
    --lmbda 0.5 --tau_start 0.5 --lr 0.0003 --epochs 100 \
    --warm_start_opq

# DINOv2 ViT-G/14, blk19, K=8
python train_soft_pq.py \
    --backbone dinov2_vitg14 --layer blk19 \
    --K 8 --embedding_dim 32 \
    --lmbda 0.5 --tau_start 0.5 --lr 0.0003 --epochs 100 \
    --warm_start_opq
```

### 批量训练

```bash
# 4 GPU 并行训练全部配置
NUM_GPUS=4 bash examples/orfc_2446/scripts/train_all.sh

# 指定 Python 解释器
PYTHON=.venv/bin/python NUM_GPUS=4 bash examples/orfc_2446/scripts/train_all.sh
```

训练日志保存在 `examples/orfc_2446/logs/train/`。

### Checkpoint 命名规则

```
{layer}_K{K}_emb{emb}_bt{bt}_{ws/km}[_lmbda{λ}]_tau{τ}_lr{lr}_ep{epochs}_n{max_train}_s{seed}.pt
```

示例: `weights/orfc_2446/dinov2_vitl14_ori/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt`

> ViT-L/14 的 codec 权重存放在 `dinov2_vitl14_ori/` 子目录；`--backbone dinov2_vitl14` 会自动映射到该路径。

## 在线评测（Engine 模式）

通过 `cofai.engine.run_eval` 端到端评测，无需预提取特征。每个 job 指定一个 `.pt` checkpoint，并 override `model.dino_codec.codec_path` 与本地 backbone 权重。

### 单次评测

```bash
# 分类 (ImageNet sel500)
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk10 --task cls \
    --ckpt_path weights/orfc_2446/dinov2_vitl14_ori/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt \
    --cuda

# 分割 (VOC2012 sel100)
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/run_eval_orfc_2446.py \
    --backbone dinov2_vitg14 --layer blk29 --task seg \
    --ckpt_path weights/orfc_2446/dinov2_vitg14_ori/blk29_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt \
    --cuda
```

### 批量在线评测

```bash
# 对照 CSV Ours 配置，全量评测（默认 GPU 0-3）
GPU_IDS=0,1,2,3 bash examples/orfc_2446/scripts/run_online_eval_all.sh

PYTHON=.venv/bin/python GPU_IDS=0,1,2,3 bash examples/orfc_2446/scripts/run_online_eval_all.sh
```

结果按层级保存在 `eval_results/SoftPQ/{dataset}/{backbone}/{slot}/{K}e{emb}/`，例如：

```
eval_results/SoftPQ/voc2012-sel100/dinov2-vitl14-slide/slot11/K64e32/result.json
eval_results/SoftPQ/imagenet-sel500/dinov2-vitl14/slot11/K64e32/result.json
```

运行日志：`eval_results/SoftPQ/.../{slot}/logs/K64e32.log`

> 若报 `No module named 'compressai._CXX'`，说明 `PYTHONPATH` 指向了未编译的 `ORFC/coding/CompressAI` 源码树。请 `unset PYTHONPATH` 或改用 `.venv/bin/python`；`run_eval_orfc_2446.py` 会自动过滤该路径。

## 离线评测

### 分类测试（单个）

```bash
cd examples/orfc_2446/offline

# 指定 checkpoint 路径
python test_cls.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --ckpt_path ../../weights/orfc_2446/dinov2_vitl14_ori/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt

# 或通过参数自动推断 checkpoint 路径
python test_cls.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --K 64 --embedding_dim 32 --lmbda 0.5 --tau_start 0.5
```

### 分割测试（单个）

```bash
python test_seg.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --ckpt_path ../../weights/orfc_2446/dinov2_vitl14_ori/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.pt
```

### 批量测试

```bash
# 自动发现所有 checkpoint 并评测
NUM_GPUS=4 bash examples/orfc_2446/scripts/test_cls_all.sh
NUM_GPUS=4 bash examples/orfc_2446/scripts/test_seg_all.sh
```

测试日志保存在 `examples/orfc_2446/logs/test_cls/` 和 `logs/test_seg/`。
结果 JSON 保存在 `results/orfc_2446/{backbone}/`。

## 关键超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lmbda` | Rate-Distortion 拉格朗日乘子 (J = R + λD) | 0.5 |
| `--tau_start` | 初始 soft-PQ 温度（>0 启用 STE 梯度） | 0.5 |
| `--tau_end` | 最终温度（退火到） | 0.005 |
| `--warm_start_opq` | 用 OPQ 解初始化旋转矩阵和码本 | True |
| `--lr` | 学习率 | 3e-4 |
| `--epochs` | 训练轮数 | 100 |
| `--bottleneck_dim` | 变换维度（0=自动=D，即 OrthogonalTransform） | 0 (=D) |

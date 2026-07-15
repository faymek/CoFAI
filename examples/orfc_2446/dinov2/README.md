# ORFC-2446 Soft-PQ — DINOv2

## 目录结构

```
examples/orfc_2446/dinov2/
├── run_eval_orfc_2446.py              # 在线模式入口
├── dinov2-orfc_2446.manifest.txt      # 数据与权重下载清单
├── offline/
│   ├── train_soft_pq.py               # 离线训练（结束写出 .npz）
│   ├── test_cls.py                    # 离线分类测试
│   ├── test_seg.py                    # 离线分割测试
│   ├── export_softpq_npz.py         # 从可选 .pt 导出评测 .npz
│   ├── extract_features.py            # 特征提取（→ examples/orfc/offline）
│   ├── utils.py / backbone / cfg      # → examples/orfc/offline
│   └── weights_paths.py
└── scripts/
    ├── unzip_softpq_weights.sh        # 解压 SoftPQ zip
    ├── train_all.sh                   # 批量训练
    ├── test_cls_all.sh / test_seg_all.sh
    └── run_online_eval_all.sh         # 批量在线评测
```

核心算法：`cofai/entropy_models/soft_pq.py`  
在线 codec：`cofai/latent_codecs/SoftPQFeatureCodec`（只加载 `.npz`）

## 下载数据与权重

权重与数据集通过清单一键拉取（默认基址
[CoFAI-share](https://medialab.sjtu.edu.cn/files/CoFAI-share/)）。
SoftPQ 压缩包：[`weights/orfc_2446/`](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/orfc_2446/)。
在仓库根目录执行：

```bash
poetry run cofai-download examples/orfc_2446/dinov2/dinov2-orfc_2446.manifest.txt
bash examples/orfc_2446/dinov2/scripts/unzip_softpq_weights.sh
```

数据集以压缩包下载到 `data/`，需**手动解压**：

```bash
unzip data/ImageNet_val_sel500.zip -d data/
unzip data/VOC2012_sel100.zip -d data/
```

解压后的最终目录：

```
CoFAI/
│
├─ data/
│   ├─ ImageNet_val_sel500/
│   └─ VOC2012_sel100/
│
├─ weights/
│   ├─ dinov2/                 # backbone + 分类/分割任务头
│   └─ orfc_2446/
│       ├─ dinov2_vitl14_ori/  # SoftPQ .npz（ViT-L/14）
│       └─ dinov2_vitg14_ori/  # SoftPQ .npz（ViT-G/14）
```

---

## 在线模式（Engine）

从原始图像出发，经 `cofai.engine`：backbone → SoftPQ 量化 → 任务头。**无需预提取特征。**

### 前置准备

1. **环境**：按项目 `README_DEV.md` 配置
2. **测试数据**（网盘下载）：
   - `ImageNet_val_sel500.zip` → `$PROJECT_ROOT/data/ImageNet_val_sel500/`
   - `VOC2012_sel100.zip` → `$PROJECT_ROOT/data/VOC2012_sel100/`
3. **权重**（网盘下载 + 解压 SoftPQ zip）：
   - `weights/dinov2/backbone/` — DINOv2 预训练 backbone
   - `weights/dinov2/cls_head/` — 分类线性头
   - `weights/dinov2/semseg_head/` — 分割线性头
   - `weights/orfc_2446/dinov2_*_ori/*.npz` — SoftPQ 量化权重

> 网盘：https://medialab.sjtu.edu.cn/files/CoFAI-share/  
> SoftPQ zip：https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/orfc_2446/

### 运行命令

在仓库根目录，并设置 `export PROJECT_ROOT=$(pwd)`。

**单配置测试**（按 `K` / `embedding_dim` 自动解析 `.npz`；同层同 K/e 有多文件时优先 `lmbda0.5`，也可显式 `--ckpt_path`）：

```bash
# 分类
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32 \
    --task cls --real --cuda

# 分割
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitg14 --layer blk09 --K 8 --embedding_dim 32 \
    --task seg --real --cuda
```

**全量测试（扫描 plan 中全部 SoftPQ 条目）：**

```bash
# ViT-L/14 分类（blk05/blk10/blk15/blk20）
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk05 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk10 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk15 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk20 --task cls --multi-run --real --cuda

# ViT-G/14 分类（blk09/blk19/blk29）
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitg14 --layer blk09 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitg14 --layer blk19 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitg14 --layer blk29 --task cls --multi-run --real --cuda

# ViT-L/14 分割
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk05 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk10 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk15 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitl14 --layer blk20 --task seg --multi-run --real --cuda

# ViT-G/14 分割
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitg14 --layer blk09 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitg14 --layer blk19 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov2/run_eval_orfc_2446.py \
    --backbone dinov2_vitg14 --layer blk29 --task seg --multi-run --real --cuda
```

或批量脚本（默认使用 `CoFAI/.venv/bin/python`，勿用系统/conda 的 `python`）：

```bash
GPU_IDS=0,1,2,3 bash examples/orfc_2446/dinov2/scripts/run_online_eval_all.sh
```

Plan：`conf/plan/dinov2/*__SoftPQ__*.yaml`（`codec_path` 均为 `.npz`）。

---

## 离线模式（Offline）

基于预提取的 `.npy` 特征训练与评测（特征目录与 ORFC 共用 `features/orfc/`）。

### 前置准备

1. **环境**：`README_DEV.md`；另需 `compressai`（rANS）
2. **测试数据与权重**：
   - `weights/orfc_2446/dinov2_*_ori/*.npz` — SoftPQ 权重（仅测试时）
   - `weights/dinov2/backbone/`、`weights/dinov2/semseg_head/` — 分割评测需要
   - `data/VOC2012_sel100/` — 分割评测需要
3. **测试特征**：解压至 `$PROJECT_ROOT/features/orfc/`（与 ORFC 相同；可从网盘 ORFC 说明获取）
4. **训练特征**（本地提取）：

```bash
python examples/orfc_2446/dinov2/offline/extract_features.py \
    --backbone dinov2_vitl14 --layer blk10 --split train
```

> 网盘：https://medialab.sjtu.edu.cn/files/CoFAI-share/

### 运行命令

**训练**（结束写出官方 `.npz`；可选 `--keep_pt` 保留 resume）：

```bash
python examples/orfc_2446/dinov2/offline/train_soft_pq.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --K 64 --embedding_dim 32 --lmbda 0.5 --tau_start 0.5 --lr 0.0003 --epochs 100

bash examples/orfc_2446/dinov2/scripts/train_all.sh
```

**分类测试：**

```bash
python examples/orfc_2446/dinov2/offline/test_cls.py \
    --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32

bash examples/orfc_2446/dinov2/scripts/test_cls_all.sh
```

**分割测试：**

```bash
python examples/orfc_2446/dinov2/offline/test_seg.py \
    --backbone dinov2_vitg14 --layer blk09 --K 8 --embedding_dim 32

bash examples/orfc_2446/dinov2/scripts/test_seg_all.sh
```

**环境变量覆盖：**

```bash
FEAT_ROOT=/path/to/features NUM_GPUS=8 bash examples/orfc_2446/dinov2/scripts/train_all.sh
NUM_GPUS=6 bash examples/orfc_2446/dinov2/scripts/test_cls_all.sh
```

---

## 支持的配置

| Backbone | 层 | 特征维度 D | 权重目录 |
|----------|------|---|----------|
| `dinov2_vitl14` | blk05, blk10, blk15, blk20 | 1024 | `weights/orfc_2446/dinov2_vitl14_ori/` |
| `dinov2_vitg14` | blk09, blk19, blk29 | 1536 | `weights/orfc_2446/dinov2_vitg14_ori/` |

| 超参 | 说明 |
|------|------|
| K | 码本大小（发布包常见 4/8/16/32/64/256） |
| embedding_dim | 子向量维度（常见 16/32） |
| lmbda / tau / lr / ep | SoftPQ 训练超参；写入文件名。同 K/e 多文件时解析优先 `lmbda0.5` |

**命名与 ORFC 的差异：** ORFC 为 `blk10_K64_e32.npz`；SoftPQ 为
`blk10_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz`
（含 bottleneck / warm-start / 训练超参）。发布包共 **55** 个 `.npz`（L:32 + G:23）。

**复现全部在线配置：**

| 方式 | 覆盖范围 |
|------|----------|
| 各层 `--multi-run`（见上文） | 与 `conf/plan/dinov2/*SoftPQ*` 中 `multi_run` 条目一致 |
| `GPU_IDS=0,1,2,3 bash examples/orfc_2446/dinov2/scripts/run_online_eval_all.sh` | `run_online_eval_all.sh` 中的 JOBS（默认 `.venv` Python） |

码率与 ORFC 相同思路：有 `pmf` 时 `--real` 走真实 rANS；理论上限约 `(D / embedding_dim) × log₂(K)` bits/token。

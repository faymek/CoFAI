# ORFC — Optimized Rotation for Feature Compression

## 目录结构

```
examples/orfc/
├── run_eval_orfc.py              # 在线模式入口
├── offline/
│   ├── train_orfc.py             # 离线训练
│   ├── test_cls.py               # 离线分类测试
│   ├── test_seg.py               # 离线分割测试
│   ├── extract_features.py       # 特征提取（离线训练前置步骤）
│   ├── utils.py
│   ├── backbone/
│   │   └── wrapper.py
│   └── cfg/
└── scripts/
    ├── train_all.sh              # 批量训练（多 GPU 并行）
    ├── test_cls_all.sh           # 批量分类测试
    └── test_seg_all.sh           # 批量分割测试
```

核心数学操作：`cofai/ops/orfc.py`；在线 codec：`cofai/latent_codecs/orfc.py`。

## 下载数据与权重

权重与数据集压缩包均通过下载清单 `dinov2-orfc.manifest.txt` 一键拉取（默认从 `https://medialab.sjtu.edu.cn/files/CoFAI-share/` 按相对路径下载）。清单覆盖全部在线评测所需的数据集 zip、DINOv2 backbone（ViT-L/14、ViT-G/14）、分类/分割任务头与 ORFC 量化权重（.npz）。在仓库根目录执行：

```bash
poetry run cofai-download examples/orfc/dinov2-orfc.manifest.txt
```

数据集以压缩包形式下载到 `data/`，目前需**手动解压**到对应位置：

```bash
# ImageNet sel500 分类
unzip data/ImageNet_val_sel500.zip -d data/
# VOC2012 sel100 分割
unzip data/VOC2012_sel100.zip -d data/
```

解压后的最终目录结构如下：

```
CoFAI/
│
├─ data/                                  # 评测所用数据子集
│   ├─ ImageNet_val_sel500/
│   └─ VOC2012_sel100/
│
├─ weights/                               # 预训练权重
│   ├─ dinov2/                            # backbone + 分类/分割任务头
│   └─ orfc/                              # ORFC 量化权重 (.npz)
```

---

## 在线模式（Engine）

从原始图像出发，通过 `cofai.engine` 管线完成 backbone 推理 → ORFC 量化 → 任务头预测。**无需提取特征。**

### 前置准备

1. **环境**：按项目 `README_DEV.md` 配置
2. **测试数据**（网盘下载）：
   - `ImageNet_val_sel500.zip` → 解压至 `$PROJECT_ROOT/data/ImageNet_val_sel500/`
   - `VOC2012_sel100.zip` → 解压至 `$PROJECT_ROOT/data/VOC2012_sel100/`
3. **权重**（网盘下载）：
   - `weights/dinov2/backbone/` — DINOv2 预训练 backbone
   - `weights/dinov2/cls_head/` — 分类线性头
   - `weights/dinov2/seg_head/` — 分割线性头
   - `weights/orfc/` — ORFC 量化权重（.npz）

> 网盘地址：https://medialab.sjtu.edu.cn/files/CoFAI-share/

### 运行命令

**单配置测试：**

```bash
# 分类
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32 \
    --task cls --real --cuda

# 分割
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitg14 --layer blk09 --K 8 --embedding_dim 32 \
    --task seg --real --cuda
```

**全量测试（扫描所有 K 值）：**

```bash
# ViT-L/14 分类（blk05/blk10/blk15/blk20）
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk05 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk10 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk15 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk20 --task cls --multi-run --real --cuda

# ViT-G/14 分类（blk09/blk19/blk29）
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitg14 --layer blk09 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitg14 --layer blk19 --task cls --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitg14 --layer blk29 --task cls --multi-run --real --cuda

# ViT-L/14 分割（blk05/blk10/blk15/blk20）
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk05 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk10 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk15 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitl14 --layer blk20 --task seg --multi-run --real --cuda

# ViT-G/14 分割（blk09/blk19/blk29）
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitg14 --layer blk09 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitg14 --layer blk19 --task seg --multi-run --real --cuda
CUDA_VISIBLE_DEVICES=0 python examples/orfc/run_eval_orfc.py \
    --backbone dinov2_vitg14 --layer blk29 --task seg --multi-run --real --cuda
```

---

## 离线模式（Offline）

基于预提取的 `.npy` 特征文件进行训练和评测，适合快速迭代。

### 前置准备

1. **环境**：按项目 `README_DEV.md` 配置，额外需要 `compressai`（rANS 编码）
2. **测试数据和权重**（网盘下载）：
   - `weights/orfc/` — 已训练的 ORFC 权重（如仅需测试）
   - `weights/dinov2/backbone/` — DINOv2 预训练 backbone（分割评测需要）
   - `weights/dinov2/semseg_head/` — 分割线性头（分割评测需要）
   - `data/VOC2012_sel100/` — VOC 分割子集（分割评测需要）
3. **测试特征**（网盘下载）：
   - 从网盘 `weights/orfc/` 目录的说明获取预提取特征
   - 解压至 `$PROJECT_ROOT/features/orfc/`
4. **训练特征**（本地提取，无法通过网盘下载）：
   ```bash
   python examples/orfc/offline/extract_features.py \
       --backbone dinov2_vitl14 --layer blk10 --split train
   ```
   需要 ImageNet 训练集（5000 张），提取后保存至 `$PROJECT_ROOT/features/orfc/train/`

> 网盘地址：https://medialab.sjtu.edu.cn/files/CoFAI-share/

### 运行命令

**训练：**

```bash
# 单配置
python examples/orfc/offline/train_orfc.py \
    --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32

# 全量批量训练
bash examples/orfc/scripts/train_all.sh
```

**分类测试：**

```bash
# 单配置
python examples/orfc/offline/test_cls.py \
    --backbone dinov2_vitl14 --layer blk10 --K 64 --embedding_dim 32

# 全量批量测试
bash examples/orfc/scripts/test_cls_all.sh
```

**分割测试：**

```bash
# 单配置
python examples/orfc/offline/test_seg.py \
    --backbone dinov2_vitg14 --layer blk09 --K 8 --embedding_dim 32

# 全量批量测试
bash examples/orfc/scripts/test_seg_all.sh
```

**环境变量覆盖（批量脚本支持）：**

```bash
FEAT_ROOT=/path/to/features NUM_GPUS=8 bash examples/orfc/scripts/train_all.sh
NUM_GPUS=6 bash examples/orfc/scripts/test_cls_all.sh
```

---

## 支持的配置

| Backbone | 层 | 特征维度 D |
|----------|------|---|
| `dinov2_vitl14` | blk05, blk10, blk15, blk20 | 1024 |
| `dinov2_vitg14` | blk09, blk19, blk29 | 1536 |
| `clip_vitl14`（仅离线） | blk05, blk10, blk15, blk20 | 1024 |

| 超参 | 可选值 |
|------|--------|
| K（码本大小） | 4, 8, 16, 32, 64, 256, 512 |
| embedding_dim（子向量维度） | 8, 16, 32 |

固定码率公式：`(D / embedding_dim) × log₂(K)` bits/token

# ORFC-2446 — Soft-PQ 特征压缩

Soft-PQ（可微乘积量化）特征压缩，覆盖 **DINOv2** 与 **DINOv3**。评测唯一权重格式为单个 `.npz`
（`R` + `codebooks` + `pmf`，可选 `norm_mode` / `n_prefix`），与 [`examples/orfc`](../orfc/) 对齐。

```text
examples/orfc_2446/
├── README.md                 # 本文件：总览
├── offline/                  # DINOv2/DINOv3 共用的训练与 artifact 工具
├── plan/                     # 标准 cofai-eval 计划
│   ├── dinov2/
│   └── dinov3/
├── dinov2/                   # DINOv2（ImageNet 分类 / VOC 分割）— 主复现入口
│   ├── README.md
│   ├── dinov2-orfc_2446.manifest.txt
│   ├── offline/
│   └── scripts/
└── dinov3/                   # DINOv3（ADE20K / NYUv2）
    ├── README.md
    ├── dinov3-orfc_2446.manifest.txt
    ├── configs/
    ├── lib/
    ├── offline/
    └── scripts/
```

| 子目录 | 任务 | 网盘 SoftPQ 权重 |
|--------|------|------------------|
| [dinov2/](dinov2/README.md) | ImageNet cls / VOC seg | 有（见下方下载） |
| [dinov3/](dinov3/README.md) | ADE20K semseg / NYUv2 depth | 有（见下方下载） |

核心训练算法：`examples/orfc_2446/offline/soft_pq.py`；在线 codec：
`cofai.latent_codecs.OrthoRotationFeatureCodec`。

## 权重格式

| 产物 | 用途 |
|------|------|
| **`.npz`** | **唯一评测/发布格式** — 在线 Engine 与离线 test/replay 均只读此文件 |
| `.pt` | 可选训练 resume（`--keep_pt`）；**不参与评测、不上传** |

## 下载数据与权重

清单默认从 [CoFAI-share](https://medialab.sjtu.edu.cn/files/CoFAI-share/) 按相对路径拉取。
SoftPQ zip 位于 [`weights/orfc_2446/`](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/orfc_2446/)。

```bash
# DINOv2
poetry run cofai-download examples/orfc_2446/dinov2/dinov2-orfc_2446.manifest.txt
bash examples/orfc_2446/dinov2/scripts/unzip_softpq_weights.sh

# DINOv3
poetry run cofai-download examples/orfc_2446/dinov3/dinov3-orfc_2446.manifest.txt
mkdir -p weights/orfc_2446/dinov3_vitl16_ori
unzip -jo weights/orfc_2446/dinov3_vitl16_ori.zip \
  'blk23_*.npz' \
  -d weights/orfc_2446/dinov3_vitl16_ori

# 数据集 zip 仍需手动解压
unzip data/ImageNet_val_sel500.zip -d data/
unzip data/VOC2012_sel100.zip -d data/
unzip data/ADE20K.zip -d data/
unzip data/NYU_subset_for_training_depth_head.zip -d data/
```

解压后目录：

```text
CoFAI/
├─ data/
│   ├─ ImageNet_val_sel500/
│   └─ VOC2012_sel100/
└─ weights/
    ├─ dinov2/                 # backbone + 分类/分割头
    └─ orfc_2446/
        ├─ dinov2_vitl14_ori/*.npz
        ├─ dinov2_vitg14_ori/*.npz
        └─ dinov3_vitl16_ori/*.npz
```

详细在线/离线命令见 [dinov2/README.md](dinov2/README.md)。DINOv3 见
[dinov3/README.md](dinov3/README.md)。

## 与经典 ORFC（OPQ）的关系

[`examples/orfc/`](../orfc/) 实现经典 ORFC：在归一化特征上交替执行硬分配
K-means 与正交 Procrustes，学习旋转矩阵 \(R\) 和分组 PQ 码本 \(C\)，直接最小化
量化前后特征的重建误差：

\[
\min_{R,C}\left\|Y-Q_C(YR)R^\mathsf{T}\right\|_2^2,\qquad
R^\mathsf{T}R=I.
\]

ORFC-2446 保留相同的正交旋转与分组 PQ 结构，但改变了参数学习方式：

1. 先用经典 ORFC/OPQ 得到 \(R\) 与码本的 warm start；
2. 训练时前向仍选择硬码字，反向通过带温度退火的 softmax
   straight-through estimator 传播梯度；
3. 启用率失真训练时，码字选择同时考虑欧氏距离和熵代价：
   \[
   \operatorname*{arg\,min}_k
   \left(\lVert z-c_k\rVert_2^2+\frac{-\log_2 p_k}{\lambda}\right),
   \]
   总体优化目标为 \(J=R_{\text{bits}}+\lambda D\)；
4. 默认的 \(D\) 不是输入特征 MSE，而是原始特征与重建特征经过
   `FrozenTail` 后的输出差异；`--mse_loss` 可退化为直接特征 MSE。

训练完成后，两者都将 \(R\)、`codebooks` 和按硬分配统计的 `pmf` 导出到
`.npz`。因此它们在评测阶段共享
`cofai.latent_codecs.OrthoRotationFeatureCodec`，执行相同的
“归一化 → 正交旋转 → 分组 PQ → 静态熵编码”流程；主要区别在训练目标与优化方法，
而不是在线码流结构。

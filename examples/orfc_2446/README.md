# ORFC-2446 — Soft-PQ 特征压缩

Soft-PQ（可微乘积量化）特征压缩，覆盖 **DINOv2** 与 **DINOv3**。评测唯一权重格式为单个 `.npz`
（`R` + `codebooks` + `pmf`，可选 `norm_mode` / `n_prefix`），与 [`examples/orfc`](../orfc/) 对齐。

```
examples/orfc_2446/
├── README.md                 # 本文件：总览
├── dinov2/                   # DINOv2（ImageNet 分类 / VOC 分割）— 主复现入口
│   ├── README.md
│   ├── run_eval_orfc_2446.py
│   ├── dinov2-orfc_2446.manifest.txt
│   ├── offline/
│   └── scripts/
└── dinov3/                   # DINOv3（ADE20K / NYUv2）— 需自备 SoftPQ 权重
    ├── README.md
    ├── run_orfc_dinov3.py
    ├── run_eval_orfc_2446_dinov3.py
    ├── configs/ lib/ offline/ scripts/
    └── results/
```

| 子目录 | 任务 | 网盘 SoftPQ 权重 |
|--------|------|------------------|
| [dinov2/](dinov2/README.md) | ImageNet cls / VOC seg | 有（见下方下载） |
| [dinov3/](dinov3/README.md) | ADE20K semseg / NYUv2 depth | **无**（需自训 `.npz`） |

核心算法：`cofai/entropy_models/soft_pq.py`；在线 codec：`cofai/latent_codecs/soft_pq.py`。

## 权重格式

| 产物 | 用途 |
|------|------|
| **`.npz`** | **唯一评测/发布格式** — 在线 Engine 与离线 test/replay 均只读此文件 |
| `.pt` | 可选训练 resume（`--keep_pt`）；**不参与评测、不上传** |

## 下载数据与权重（DINOv2）

清单默认从 [CoFAI-share](https://medialab.sjtu.edu.cn/files/CoFAI-share/) 按相对路径拉取。
SoftPQ zip 位于 [`weights/orfc_2446/`](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/orfc_2446/)。

```bash
# 仓库根目录 (CoFAI/)
poetry run cofai-download examples/orfc_2446/dinov2/dinov2-orfc_2446.manifest.txt

# SoftPQ zip → 解压出 *.npz
bash examples/orfc_2446/dinov2/scripts/unzip_softpq_weights.sh

# 数据集 zip 需手动解压
unzip data/ImageNet_val_sel500.zip -d data/
unzip data/VOC2012_sel100.zip -d data/
```

解压后目录：

```
CoFAI/
├─ data/
│   ├─ ImageNet_val_sel500/
│   └─ VOC2012_sel100/
└─ weights/
    ├─ dinov2/                 # backbone + 分类/分割头
    └─ orfc_2446/
        ├─ dinov2_vitl14_ori/*.npz
        └─ dinov2_vitg14_ori/*.npz
```

详细在线/离线命令见 [dinov2/README.md](dinov2/README.md)。DINOv3 见 [dinov3/README.md](dinov3/README.md)。

## 相关

- 经典 ORFC（OPQ）：`examples/orfc/`
- PQFC（正交变换消融）：`examples/pqfc/`

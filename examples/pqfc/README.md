# PQFC — Product Quantization Feature Codec（DINOv3 离线）

离线训练 / 评测（**分割 + 深度估计**），对齐 CTC DINOv3。正交变换可开关并与 soft/hard 绑定；损失 `J = R + λ · D`（`D` = frozen-tail ΔL_ref）。**不**提供在线 plan。

## 开箱流程

```bash
cd /data4/workspace/zlt/featcodec/CoFAI
unset PYTHONPATH
export PROJECT_ROOT=$(pwd)

# 0) 环境（一次性）
poetry install
poetry run pip install --editable .
poetry run python install.py          # 写 .env / PROJECT_ROOT

# 1) 下载评测数据 + backbone + 任务头，并解压
bash examples/pqfc/scripts/prepare_dinov3.sh

# 2) 提取特征
# 训练：ImageNet val（需本地图像；清单默认 featcodec/utils/imagenet_selected_pathname5000.txt）
bash examples/pqfc/scripts/extract_train.sh --device cuda:0
# 评测：ADE20K / NYUv2（CTC 同口径，原生分辨率 pad×16）
bash examples/pqfc/scripts/extract_val.sh

# 3) 训练（参数全部透传，不写死配置）
bash examples/pqfc/scripts/train_dinov3.sh --K 256 --embedding_dim 32 --epochs 100
bash examples/pqfc/scripts/train_dinov3.sh --K 256 --embedding_dim 16 --no_transform --lmbda 0.0

# 4) 测试（semseg / depth；只读单个 .npz）
bash examples/pqfc/scripts/replay_dinov3.sh --task semseg \
  --ckpt_path weights/pqfc/dinov3_vitl16/<ckpt>.npz
bash examples/pqfc/scripts/replay_dinov3.sh --task depth \
  --ckpt_path weights/pqfc/dinov3_vitl16/<ckpt>.npz
```

## 目录

```
examples/pqfc/
├── README.md
├── manifests/dinov3-pqfc.manifest.txt   # cofai-download 清单
├── run_pqfc_dinov3.py                   # replay / rate
├── configs/dinov3_blk23.yaml
├── offline/
│   ├── train_pqfc_dinov3.py
│   └── extract_train_features.py        # ImageNet → train .npy
└── scripts/
    ├── prepare_dinov3.sh                # 下载 + 解压 + 检查
    ├── extract_train.sh / extract_val.sh
    ├── train_dinov3.sh                  # 透传训练参数
    └── replay_dinov3.sh                 # 透传评测参数
```

## 准备 ImageNet（仅训练特征）

评测数据走 CTC 网盘；**训练特征**需要 ImageNet val 图像：

```bash
# 默认列表（5000）：../utils/imagenet_selected_pathname5000.txt
# 默认图像根：../data/imagenet/images/val/<wnid>/*.JPEG
IMAGENET_ROOT=/path/to/imagenet/val \
  bash examples/pqfc/scripts/extract_train.sh --max_images 5000
```

输出默认：`features/train/dinov3_vitl16/blk23/`（`T=201 = 1 CLS + 4 reg + 196`）。  
若已有 `../ORFC/features/train/dinov3_vitl16/blk23`，`prepare_dinov3.sh` 会自动软链过来。

## 归一化 / 权重格式

`--norm_mode`：`per_image`（默认）| `split_cls_patch` | `split_reg_cls_patch` | `per_token_ln`

评测只读单个 **`.npz`**（`R` + `codebooks` + `pmf` + `norm_mode` / `n_prefix`）。训练默认只写 `.npz`；`--keep_pt` 可额外存 resume 用 `.pt`。

## 脚本参数约定

| 脚本 | 说明 |
|------|------|
| `train_dinov3.sh` | 后面所有参数原样传给 `train_pqfc_dinov3.py`；`GPU=` / `FEAT_DIR=` / `CONFIG=` 可用环境变量 |
| `replay_dinov3.sh` | 默认子命令 `replay`；也可 `rate`；参数透传给 `run_pqfc_dinov3.py` |
| `extract_train.sh` / `extract_val.sh` | 参数透传到对应 Python；`extract_val` 未传 `--task` 时跑 semseg+depth |

示例：

```bash
GPU=0 bash examples/pqfc/scripts/train_dinov3.sh \
  --K 64 --embedding_dim 32 --lmbda 0.5 --epochs 30 \
  --batch_size 16 --max_train 1000

GPU=1 bash examples/pqfc/scripts/replay_dinov3.sh rate --task depth \
  --ckpt_path weights/pqfc/dinov3_vitl16/<ckpt>.npz
```

## Soft / Hard

| 模式 | 变换 | 分配 |
|------|------|------|
| `--use_transform`（默认） | `OrthogonalTransform` | soft |
| `--no_transform` | 无 R | hard（`tau_start=0`） |

# PQFC — Product Quantization Feature Codec

基于 `examples/orfc_2446` 的离线训练/评测示例。正交变换可开关，并与 soft/hard 分配绑定；损失为 `J = R + λ · D`（`D` = frozen-tail ΔL_ref）。目录内支持 CTC DINOv3（离线训练 + replay），**不**提供 CoFAI 在线 plan。

## 目录结构

```
examples/pqfc/
├── README.md
├── run_pqfc_dinov3.py              # DINOv3 CTC replay (semseg / depth)
├── configs/dinov3_blk23.yaml
├── lib/                            # DINOv3 工具（部分 symlink 自 orfc_2446_dinov3）
├── offline/
│   ├── train_pqfc.py               # DINOv2 / CLIP 训练
│   ├── train_pqfc_dinov3.py        # DINOv3 CTC 训练
│   ├── test_cls.py / test_seg.py   # DINOv2 离线评测
│   ├── weights_paths.py
│   ├── utils.py / backbone / cfg   # → examples/orfc/offline
└── scripts/
    ├── train_all.sh / train_vitl14_k4.sh
    ├── test_cls_all.sh / test_seg_all.sh
    └── train_dinov3.sh
```

核心算法：`cofai/entropy_models/soft_pq.py`

## Soft / Hard 与正交变换绑定

| 模式 | 变换 | 分配 | 初始化 |
|------|------|------|--------|
| `--use_transform`（**默认**） | `OrthogonalTransform` | **soft**（τ: 0.5→0.005） | OPQ warm-start（与 orfc_2446 一致） |
| `--no_transform` | 无 R | **hard**（强制 `tau_start=0`） | k-means 初始化码本 |

入口会按开关自动设置 tau，避免有变换却 hard / 无变换却 soft。

## 损失

- `D`：`‖FrozenTail(X) − FrozenTail(X̂)‖²`（ΔL_ref）
- `J = R + λ · D`（默认 `λ=0.5`）
- **不**使用特征空间 `--mse_loss` 作为默认训练模式

## DINOv2 训练

```bash
cd /data4/workspace/zlt/featcodec/CoFAI
unset PYTHONPATH
export PROJECT_ROOT=$(pwd)

# 默认：有正交变换 + soft PQ（对齐 orfc_2446）
poetry run python examples/pqfc/offline/train_pqfc.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --K 64 --embedding_dim 32 --lmbda 0.5

# 无正交变换 + hard PQ
poetry run python examples/pqfc/offline/train_pqfc.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --K 64 --embedding_dim 32 --no_transform

# 批量
NUM_GPUS=4 bash examples/pqfc/scripts/train_all.sh
EXTRA_ARGS='--no_transform' NUM_GPUS=4 bash examples/pqfc/scripts/train_all.sh
```

权重：`weights/pqfc/{dinov2_vitl14,dinov2_vitg14}/`

## DINOv2 评测

```bash
poetry run python examples/pqfc/offline/test_cls.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --ckpt_path weights/pqfc/dinov2_vitl14/<ckpt>.pt

poetry run python examples/pqfc/offline/test_seg.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --ckpt_path weights/pqfc/dinov2_vitl14/<ckpt>.pt

NUM_GPUS=4 bash examples/pqfc/scripts/test_cls_all.sh
NUM_GPUS=4 bash examples/pqfc/scripts/test_seg_all.sh

# 一致性检验：加载 orfc_2446 已有 vitl14 全部码本配置（blk05/10/15/20），0-3 卡并行
GPU_IDS=0,1,2,3 PYTHON=.venv/bin/python \
    bash examples/pqfc/scripts/test_vitl14_all_ckpts.sh

# 有变换 + soft：K4e32 训练并测试（超参对齐 orfc_2446 train_vitl14_k4）
GPU_IDS=0,1,2,3 PYTHON=.venv/bin/python \
    bash examples/pqfc/scripts/train_test_vitl14_k4.sh
```

结果 JSON：`results/pqfc/{backbone}/`

## CTC DINOv3（文件夹内）

前置：ImageNet blk23 训练特征（`ORFC/features/train/dinov3_vitl16/blk23`，T≈201）、slot24 val 特征、DINOv3 backbone/head 权重（见 `configs/dinov3_blk23.yaml`）。COCO 长序列（T≈5445）可用 `FEAT_DIR=.../dinov3_vitl16_coco/blk23` 覆盖，但 24GB 卡上 SoftPQ 很重。

```bash
cd /data4/workspace/zlt/featcodec/CoFAI
unset PYTHONPATH
export PROJECT_ROOT=$(pwd)

# 8 configs: K256 e32/e16 × with/without R × λ∈{0.5,0.0}，GPU 3,4,5,7
bash examples/pqfc/scripts/train_dinov3.sh
# 或
poetry run python examples/pqfc/offline/train_pqfc_dinov3.py --K 256 --embedding_dim 16

# 无 R + hard（脚本内已含 noR 配置；单跑示例）
poetry run python examples/pqfc/offline/train_pqfc_dinov3.py --K 256 --embedding_dim 16 --no_transform

# Replay（semseg / depth）
poetry run python examples/pqfc/run_pqfc_dinov3.py replay \
    --task semseg --ckpt weights/pqfc/dinov3_vitl16/<ckpt>.pt --gpu 0
poetry run python examples/pqfc/run_pqfc_dinov3.py replay \
    --task depth --ckpt weights/pqfc/dinov3_vitl16/<ckpt>.pt --gpu 0
```

权重：`weights/pqfc/dinov3_vitl16/`  
结果：`examples/pqfc/results/dinov3/`

## Checkpoint 命名

- 有变换：`{layer}_K{K}_emb{d}_bt{D}_{ws|km}_lmbda{λ}_tau{τ}_lr{lr}_ep{ep}_n{n}_s{seed}.pt`
- 无变换：`{layer}_K{K}_emb{d}_noR_km_lmbda{λ}_lr{lr}_ep{ep}_n{n}_s{seed}.pt`

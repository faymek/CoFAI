# PQFC — Product Quantization Feature Codec

离线训练/评测示例，与 SoftPQ（`examples/orfc_2446`）同族：正交变换可开关并与 soft/hard 分配绑定；损失 `J = R + λ · D`（`D` = frozen-tail ΔL_ref）。目录内支持 CTC DINOv3（离线 train + replay），**不**提供 CoFAI 在线 plan。

对比 SoftPQ 主线见 [`examples/orfc_2446/dinov3/`](../orfc_2446/dinov3/)（单 `.npz` 评测）。PQFC 评测口径尽量对齐「单 npz」：训练结束写出含 `R`/`codebooks`/`pmf` 的 `.npz` 时，replay 优先加载该文件。

## 目录结构

```
examples/pqfc/
├── README.md
├── run_pqfc_dinov3.py              # DINOv3 CTC replay (semseg / depth)
├── configs/dinov3_blk23.yaml
├── lib/                            # DINOv3 工具（部分 symlink 自 orfc_2446/dinov3）
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

# 默认：有正交变换 + soft PQ（对齐 orfc_2446/dinov2）
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
    --ckpt_path weights/pqfc/dinov2_vitl14/<ckpt>.npz

poetry run python examples/pqfc/offline/test_seg.py \
    --backbone dinov2_vitl14 --layer blk10 \
    --ckpt_path weights/pqfc/dinov2_vitl14/<ckpt>.npz

NUM_GPUS=4 bash examples/pqfc/scripts/test_cls_all.sh
NUM_GPUS=4 bash examples/pqfc/scripts/test_seg_all.sh

# 一致性检验：加载 orfc_2446 SoftPQ vitl14 码本（见 examples/orfc_2446/dinov2）
GPU_IDS=0,1,2,3 PYTHON=.venv/bin/python \
    bash examples/pqfc/scripts/test_vitl14_all_ckpts.sh

GPU_IDS=0,1,2,3 PYTHON=.venv/bin/python \
    bash examples/pqfc/scripts/train_test_vitl14_k4.sh
```

结果 JSON：`results/pqfc/{backbone}/`

## CTC DINOv3（文件夹内）

与 SoftPQ DINOv3（[`orfc_2446/dinov3`](../orfc_2446/dinov3/)）共用特征/backbone 约定，但权重目录独立（`weights/pqfc/dinov3_vitl16/`）。

前置：ImageNet blk23 训练特征、slot24 val 特征、DINOv3 backbone/head 权重（见 `configs/dinov3_blk23.yaml`）。

```bash
cd /data4/workspace/zlt/featcodec/CoFAI
unset PYTHONPATH
export PROJECT_ROOT=$(pwd)

bash examples/pqfc/scripts/train_dinov3.sh
# 或
poetry run python examples/pqfc/offline/train_pqfc_dinov3.py --K 256 --embedding_dim 16

poetry run python examples/pqfc/offline/train_pqfc_dinov3.py --K 256 --embedding_dim 16 --no_transform

# Replay：优先传单个 .npz（含 pmf）；若仅有历史 .pt，需先导出 npz
poetry run python examples/pqfc/run_pqfc_dinov3.py replay \
    --task semseg --ckpt weights/pqfc/dinov3_vitl16/<ckpt>.npz --gpu 0
poetry run python examples/pqfc/run_pqfc_dinov3.py replay \
    --task depth --ckpt weights/pqfc/dinov3_vitl16/<ckpt>.npz --gpu 0
```

权重：`weights/pqfc/dinov3_vitl16/`  
结果：`examples/pqfc/results/dinov3/`

## 与 SoftPQ（orfc_2446）对照

| | SoftPQ `orfc_2446` | PQFC |
|--|--------------------|------|
| DINOv2 | `orfc_2446/dinov2/` | `pqfc/offline/train_pqfc.py` |
| DINOv3 | `orfc_2446/dinov3/` | `pqfc/`（本目录） |
| 评测权重 | **仅 `.npz`** | 尽量对齐单 `.npz` |
| 在线 Engine | SoftPQ plan 已接线 | 不提供 |

## Checkpoint 命名

- 有变换：`{layer}_K{K}_emb{d}_bt{D}_{ws|km}_lmbda{λ}_tau{τ}_lr{lr}_ep{ep}_n{n}_s{seed}`
- 无变换：`{layer}_K{K}_emb{d}_noR_km_lmbda{λ}_lr{lr}_ep{ep}_n{n}_s{seed}`
- 发布/评测优先 `.npz`；`.pt` 仅作可选 resume

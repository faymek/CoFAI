# PQFC — Product Quantization Feature Codec（DINOv3）

DINOv3 ViT-L/16 **slot24 / blk23** 特征压缩，CTC 任务：**ADE20K 语义分割**、**NYUv2 深度估计**。

- **在线 codec**：`cofai.latent_codecs.OrthoRotationFeatureCodec`（`.npz`：`R` + `codebooks` + `pmf`）。
- **离线训练**：与 ORFC-2446 同一套 SoftPQ（`examples.orfc_2446.offline`），**不**写入 `cofai/entropy_models`。`--use_transform` / `--no_transform` 绑定 soft / hard。

| 变体 | Plan 标记 | 权重 |
|------|-----------|------|
| **有正交变换** | `__PQFC__` | [ORFC-2446 DINOv3 release](../orfc_2446/dinov3/README.md) `weights/orfc_2446/dinov3_vitl16_ori/blk23_*.npz` |
| **无正交变换** | `__PQFC-noR__` | 训练导出到 `weights/pqfc/dinov3_vitl16_noR/`（plan 占位路径；未训练前评测会缺文件） |

```text
examples/pqfc/
├── README.md
├── dinov3-pqfc.manifest.txt
├── configs/dinov3_blk23.yaml
├── plan/dinov3/                 # cofai-eval 在线 plan
├── lib/                         # 离线 helpers
├── offline/train_pqfc_dinov3.py
├── run_pqfc_dinov3.py           # 离线 replay / rate
└── scripts/
    ├── run_eval_ctc.sh
    ├── prepare_dinov3.sh
    ├── extract_train.sh / extract_val.sh
    ├── train_dinov3.sh
    └── replay_dinov3.sh
```

## 准备

```bash
SOURCE_ROOT="$PWD"
PROJECT_ROOT="$(awk -F= '/^[[:space:]]*PROJECT_ROOT[[:space:]]*=/ {
  sub(/^[^=]*=[[:space:]]*/, ""); print; exit
}' .env)"
export SOURCE_ROOT PROJECT_ROOT PYTHONPATH="$SOURCE_ROOT"

bash examples/pqfc/scripts/prepare_dinov3.sh
```

有变换 8 档与 ORFC-2446 相同（见 `plan/dinov3/*__PQFC__*.yaml`）。

## 离线训练

训练脚本会把 `PYTHONPATH` 指到本仓库，SoftPQ 从 `examples.orfc_2446.offline` 导入。

```bash
# ImageNet val 特征（T=201）
IMAGENET_ROOT=/path/to/imagenet/val \
  bash examples/pqfc/scripts/extract_train.sh --max_images 5000

# 有变换（默认 OrthogonalTransform + soft PQ）
GPU=0 bash examples/pqfc/scripts/train_dinov3.sh --K 256 --embedding_dim 32 --epochs 100

# 无变换（hard PQ，导出到 noR 目录）
GPU=0 WEIGHTS_DIR=weights/pqfc/dinov3_vitl16_noR \
  bash examples/pqfc/scripts/train_dinov3.sh \
    --K 256 --embedding_dim 32 --no_transform --lmbda 0.0
```

离线 replay（需先 `extract_val.sh`）：

```bash
bash examples/pqfc/scripts/extract_val.sh
bash examples/pqfc/scripts/replay_dinov3.sh --task semseg \
  --ckpt_path weights/pqfc/dinov3_vitl16/<ckpt>.npz
```

## 在线评测

```bash
GPU_IDS=0,1 bash examples/pqfc/scripts/run_eval_ctc.sh
TASK=semseg GPU_IDS=0 bash examples/pqfc/scripts/run_eval_ctc.sh
VARIANT=noR GPU_IDS=0 bash examples/pqfc/scripts/run_eval_ctc.sh
```

单 plan：

```bash
PYTHONPATH="$SOURCE_ROOT" CUDA_VISIBLE_DEVICES=0 \
poetry -C "$PROJECT_ROOT" run cofai-eval \
  "$SOURCE_ROOT/examples/pqfc/plan/dinov3/ade20k-val__dinov3-vitl16-slot24__PQFC__semseg.yaml" \
  args.cuda=true args.real=true args.multi_run=true \
  "args.output_dir=$PROJECT_ROOT/logs/pqfc/dinov3"
```

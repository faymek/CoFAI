# ORFC-2446 Soft-PQ — DINOv3

DINOv3 ViT-L/16 **slot24 / blk23** SoftPQ：离线训练 / replay，以及在线 Engine CTC 评测（ADE20K 分割 + NYUv2 深度）。

## 目录结构

```
examples/orfc_2446/dinov3/
├── run_orfc_dinov3.py                 # 离线 replay / rate（只读 .npz）
├── run_eval_orfc_2446_dinov3.py       # 在线 Engine 入口（支持 multi-run + 多卡）
├── configs/dinov3_blk23.yaml
├── lib/
├── offline/
│   ├── train_soft_pq_dinov3.py
│   └── ...
├── scripts/
│   ├── run_train_frozen_tail.sh
│   ├── run_eval_tasks.sh              # 离线 replay 批量评测
│   └── run_eval_release_ctc.sh        # 在线 CTC 一键复现（多卡）
```

## 权重格式

评测唯一格式为单个 **`.npz`**（`R` + `codebooks` + `pmf` + `norm_mode` / `n_prefix`）。

| `norm_mode` | 含义 | 典型 `n_prefix` |
|-------------|------|-----------------|
| `per_image` | 整图统一 mean/std | 忽略 |
| `split_cls_patch` | cls+reg 一组；patch 一组 | 5 |
| `split_reg_cls_patch` | reg 一组；cls+patch 一组 | 5 |

发布权重目录：`weights/orfc_2446_dinov3/`（含 Engine plan 引用的 SoftPQ `.npz`）。

## 下载数据与权重

1. DINOv3 backbone / 任务头（见 `examples/ctc/README-DINOv3.md`）
2. 任务数据：ADE20K val、NYUv2（路径见 plan / `configs/dinov3_blk23.yaml`）
3. SoftPQ `.npz`：`weights/orfc_2446_dinov3/`（自训或发布包）

> 网盘总址：https://medialab.sjtu.edu.cn/files/CoFAI-share/

---

## 在线模式（Engine CTC）

Plan（含 8 档 SoftPQ `multi_run`）：

- `conf/plan/dinov3/ade20k-val__dinov3-vitl16-slot24__SoftPQ__semseg.yaml`
- `conf/plan/dinov3/nyuv2-val__dinov3-vitl16-slot24__SoftPQ__depth.yaml`

码本配置：K16e32、K256e32、K1024e32、K512e16、K1024e16、K64e8、K256e8、K512e8。

### 一键复现（多卡并行）

```bash
cd /data4/workspace/zlt/featcodec/CoFAI
export PROJECT_ROOT=$(pwd)

# 默认 GPUS=0,1,2,3，扫 plan 全部 multi_run × semseg+depth
bash examples/orfc_2446/dinov3/scripts/run_eval_release_ctc.sh

# 指定 GPU / 单任务
GPUS=0,1,2,3 TASK=semseg bash examples/orfc_2446/dinov3/scripts/run_eval_release_ctc.sh
```

等价 Python：

```bash
poetry run python examples/orfc_2446/dinov3/run_eval_orfc_2446_dinov3.py \
    --task both --multi-run --gpus 0,1,2,3 --cuda --real
```

结果目录：`eval_results/SoftPQ/dinov3-vitl16-slot24/{semseg,depth}/q*/`。

### 单档评测

```bash
CUDA_VISIBLE_DEVICES=0 poetry run python examples/orfc_2446/dinov3/run_eval_orfc_2446_dinov3.py \
    --task semseg \
    --ckpt_path weights/orfc_2446_dinov3/release/blk23_K256_e32.npz \
    --cuda --real
```

无 SoftPQ 的 Bypass 基线见 `examples/ctc/README-DINOv3.md`。

---

## 离线模式（Offline）

### 训练

```bash
FEAT_DIR=/data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_ade \
NORM_MODE=split_reg_cls_patch TRAIN=5000 TOKEN_HW=32,43 \
K=256 EMB=32 GPU=0 \
bash examples/orfc_2446/dinov3/scripts/run_train_frozen_tail.sh
```

写出 `weights/orfc_2446_dinov3/*.npz`。

### Replay

```bash
CKPTS="weights/orfc_2446_dinov3/<ckpt>.npz" \
TASKS=semseg,depth GPUS=0,1,2,3 NORM=split_reg_cls_patch \
bash examples/orfc_2446/dinov3/scripts/run_eval_tasks.sh
```

---

## 与 DINOv2 对照

| | dinov2 | dinov3 |
|--|--------|--------|
| 任务 | ImageNet cls / VOC seg | ADE20K semseg / NYUv2 depth |
| Slot | blk05/10/15/20（L）或 blk09/19/29（G） | slot24 / blk23 |
| 默认发布 norm | `per_image` | `split_reg_cls_patch` |
| SoftPQ 权重 | `weights/orfc_2446/` | `weights/orfc_2446_dinov3/` |

# ORFC-2446 Soft-PQ — DINOv3

DINOv3 ViT-L/16 **slot24 / blk23** 的 SoftPQ 离线训练与 replay，以及在线 Engine 的 **代码模板**。
**本次不提供网盘 SoftPQ 权重**；需自训得到 `.npz` 后再跑评测。

## 目录结构

```
examples/orfc_2446/dinov3/
├── run_orfc_dinov3.py                 # 离线 replay / rate（只读 .npz）
├── run_eval_orfc_2446_dinov3.py       # 在线 Engine 入口（代码模板）
├── configs/dinov3_blk23.yaml
├── lib/                               # 配置、codec、rate 等
├── offline/
│   ├── train_soft_pq_dinov3.py        # 训练 → 写出 .npz
│   ├── export_npz_dinov3.py
│   ├── extract_features_dinov3.py
│   └── ...
├── scripts/
│   ├── run_train_pipeline.sh          # 官方训练入口
│   ├── run_replay_eval.sh             # 官方 replay 入口
│   └── ...                            # RAEtail / 诊断脚本（可选实验）
└── results/
```

## 权重格式

评测唯一格式为单个 **`.npz`**（`R` + `codebooks` + `pmf` + `norm_mode` / `n_prefix`）。

| `norm_mode` | 含义 | 典型 `n_prefix` |
|-------------|------|-----------------|
| `per_image` | 整图统一 mean/std | 忽略 |
| `split_cls_patch` | cls+reg 一组；patch 一组 | 5 |
| `split_reg_cls_patch` | reg 一组；cls+patch 一组 | 5 |

默认训练/评测常用 `split_cls_patch`。权重目录：`weights/orfc_2446_dinov3/`。

## 下载数据与权重

DINOv3 SoftPQ **无** CoFAI-share 发布包。需自行准备：

1. DINOv3 backbone / 任务头（可参考 `examples/ctc/README-DINOv3.md` 的 manifest）
2. 训练/验证特征（`extract_features_dinov3.py` 或配置中的路径）
3. SoftPQ `.npz`（本目录训练产出）

ADE20K / NYUv2 等任务数据路径见 `configs/dinov3_blk23.yaml`。

> 网盘总址：https://medialab.sjtu.edu.cn/files/CoFAI-share/

---

## 在线模式（Engine，代码模板）

Plan：

- `conf/plan/dinov3/ade20k-val__dinov3-vitl16-slot24__SoftPQ__semseg.yaml`
- `conf/plan/dinov3/nyuv2-val__dinov3-vitl16-slot24__SoftPQ__depth.yaml`

```bash
export PROJECT_ROOT=$(pwd)

# 需已有 SoftPQ .npz；可用 --dry-run 检查接线
CUDA_VISIBLE_DEVICES=0 python examples/orfc_2446/dinov3/run_eval_orfc_2446_dinov3.py \
    --task semseg \
    --ckpt_path weights/orfc_2446_dinov3/<ckpt>.npz \
    --cuda --real
```

无 SoftPQ 的 Bypass 基线见 `examples/ctc/README-DINOv3.md`。

---

## 离线模式（Offline）

### 训练

```bash
poetry run python examples/orfc_2446/dinov3/offline/train_soft_pq_dinov3.py \
    --K 4 --embedding_dim 32 --norm_mode split_cls_patch --gpu 0

bash examples/orfc_2446/dinov3/scripts/run_train_pipeline.sh
```

写出 `weights/orfc_2446_dinov3/*.npz`。仅 resume 时加 `--keep_pt`。

### Replay / Rate

```bash
poetry run python examples/orfc_2446/dinov3/run_orfc_dinov3.py \
    --config examples/orfc_2446/dinov3/configs/dinov3_blk23.yaml \
    replay --task semseg \
    --ckpt_path weights/orfc_2446_dinov3/<ckpt>.npz --gpu 0

bash examples/orfc_2446/dinov3/scripts/run_replay_eval.sh
```

官方入口：`scripts/run_train_pipeline.sh`、`scripts/run_replay_eval.sh`。

---

## 与 DINOv2 对照

| | dinov2 | dinov3 |
|--|--------|--------|
| 任务 | ImageNet cls / VOC seg | ADE20K semseg / NYUv2 depth |
| Slot | blk05/10/15/20（L）或 blk09/19/29（G） | slot24 / blk23 |
| 默认 norm | `per_image` | `split_cls_patch` |
| 网盘 SoftPQ | [`weights/orfc_2446/`](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/orfc_2446/) | 不发布 |

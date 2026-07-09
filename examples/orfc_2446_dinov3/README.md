# DINOv3 ORFC-2446 (SoftPQ) — Offline Pipeline

DINOv3 ViT-L/16 **blk23** (= slot24 encode) offline SoftPQ training and semseg/depth replay.

## Directory layout

```
examples/orfc_2446_dinov3/
├── configs/dinov3_blk23.yaml
├── run_orfc_dinov3.py              # replay | rate
├── offline/train_soft_pq_dinov3.py # training
├── lib/
│   ├── config_utils.py
│   ├── dataset_utils.py            # re-export from examples/vtm
│   ├── dinov3_frozen_tail.py
│   ├── orfc_codec.py
│   └── rate_eval.py
└── scripts/run_train_pipeline.sh
```

Core algorithm: `cofai/entropy_models/soft_pq.py`

## Prerequisites

1. **COCO train features** (blk23): `ORFC/features/train/dinov3_vitl16_coco/blk23`
2. **Val features** (slot24): run VTM extract first:
   ```bash
   cd /data4/workspace/zlt/featcodec/CoFAI
   unset PYTHONPATH
   export PROJECT_ROOT=$(pwd)
   poetry run python examples/vtm/run_vtm_dinov3.py extract --task semseg --gpu 0
   poetry run python examples/vtm/run_vtm_dinov3.py extract --task depth --gpu 0
   ```
3. Always `unset PYTHONPATH` before running (avoid `compressai._CXX` import errors).

## Training

```bash
cd /data4/workspace/zlt/featcodec/CoFAI
unset PYTHONPATH
export PROJECT_ROOT=$(pwd)

# Single config (split_cls_patch — matches compression_vit COCO recipe)
poetry run python examples/orfc_2446_dinov3/offline/train_soft_pq_dinov3.py \
  --K 4 --embedding_dim 32 --norm_mode split_cls_patch --gpu 0

# per_image norm (matches orfc_2446 DINOv2 default)
poetry run python examples/orfc_2446_dinov3/offline/train_soft_pq_dinov3.py \
  --K 4 --embedding_dim 32 --norm_mode per_image --gpu 0
```

Checkpoints: `weights/orfc_2446_dinov3/blk23_K{K}_emb{d}_...pt` + sidecar `.npz` (R, codebooks, pmf).

### Batch training

```bash
bash examples/orfc_2446_dinov3/scripts/run_train_pipeline.sh
```

Override via env: `EPOCHS=30 TRAIN=1000 NORM=split_cls_patch K configs in script`.

## Replay (semseg / depth)

```bash
# Bypass baseline (no codec)
poetry run python examples/orfc_2446_dinov3/run_orfc_dinov3.py \
  replay --task semseg --mode bypass --gpu 0

# ORFC replay
poetry run python examples/orfc_2446_dinov3/run_orfc_dinov3.py \
  replay --task semseg --mode orfc \
  --ckpt_path weights/orfc_2446_dinov3/blk23_K4_emb32_bt1024_ws_tau0.5_lr0.0003_ep30_n1000_s42.pt \
  --norm_mode split_cls_patch --gpu 0

# Smoke subset
poetry run python examples/orfc_2446_dinov3/run_orfc_dinov3.py \
  replay --task semseg --mode orfc --ckpt_path <ckpt> \
  --subset examples/vtm/subsets/ade20k_smoke.txt --gpu 0
```

## Rate-only evaluation

```bash
poetry run python examples/orfc_2446_dinov3/run_orfc_dinov3.py \
  rate --task semseg --ckpt_path <ckpt> --norm_mode split_cls_patch --gpu 0
```

## Norm modes

| `--norm_mode` | `--n_prefix` | Notes |
|---------------|--------------|-------|
| `split_cls_patch` | auto 5 if 0 | CLS+reg vs patch separate stats (compression_vit) |
| `split_reg_cls_patch` | auto 5 if 0 | reg alone vs cls+patch shared stats (same 64b side info) |
| `per_image` | ignored | Global per-image mean/std (orfc_2446 default) |
| `per_token_ln` | ignored | Per-token layer norm stats |

## vs `examples/orfc_2446` (DINOv2)

| | orfc_2446 | orfc_2446_dinov3 |
|--|-----------|------------------|
| Backbone | DINOv2 ViT-L/G | DINOv3 ViT-L/16 |
| Train data | ImageNet 5k | COCO 5k blk23 |
| Eval tasks | cls / VOC seg | ADE20K seg / NYUv2 depth |
| Frozen tail | generic | RoPE-aware (norm-only at blk23) |

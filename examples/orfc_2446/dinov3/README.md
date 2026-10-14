# ORFC-2446 — DINOv3

The DINOv3 integration evaluates ViT-L/16 **slot24** (ADE20K / NYUv2) and **slot6** (ADE20K / NYUv2 / COCO val 1k detection).

All plans use the standard `DinoFeatureCodecModel` and
`OrthoRotationFeatureCodec`.

## Environment and assets

From the worktree root:

```bash
SOURCE_ROOT="$PWD"
PROJECT_ROOT="$(awk -F= '/^[[:space:]]*PROJECT_ROOT[[:space:]]*=/ {
  sub(/^[^=]*=[[:space:]]*/, ""); print; exit
}' .env)"

poetry -C "$PROJECT_ROOT" run cofai-download \
  "$SOURCE_ROOT/examples/orfc_2446/dinov3/dinov3-orfc_2446.manifest.txt"

mkdir -p "$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_ori"
unzip -jo "$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_ori.zip" \
  'blk23_*.npz' \
  -d "$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_ori"
unzip "$PROJECT_ROOT/data/ADE20K.zip" -d "$PROJECT_ROOT/data"
unzip "$PROJECT_ROOT/data/NYU_subset_for_training_depth_head.zip" -d "$PROJECT_ROOT/data"
unzip "$PROJECT_ROOT/data/coco_val_1k.zip" -d "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_slot6"
unzip -jo "$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_slot6.zip" \
  'blk05_*.npz' \
  -d "$PROJECT_ROOT/weights/orfc_2446/dinov3_vitl16_slot6"
```

The manifest installs the DINOv3 backbone, datasets, task heads, and SoftPQ zips.

slot24 artifacts (`dinov3_vitl16_ori/`):

```text
weights/orfc_2446/dinov3_vitl16_ori/
├── blk23_K16_e32.npz
├── blk23_K256_e32.npz
├── blk23_K1024_e32.npz
├── blk23_K512_e16.npz
├── blk23_K1024_e16.npz
├── blk23_K64_e8.npz
├── blk23_K256_e8.npz
└── blk23_K512_e8.npz
```

slot6 artifacts (`dinov3_vitl16_slot6/`): `blk05_K{4,8,16,256}_e32.npz`, mapped to qualities 1–4.

Each artifact contains `R`, `codebooks`, `pmf`, `norm_mode`, and `n_prefix`.

The two slot24 plans use the same integer quality mapping:

| quality | artifact |
| ---: | --- |
| 1 | `blk23_K16_e32.npz` |
| 2 | `blk23_K256_e32.npz` |
| 3 | `blk23_K1024_e32.npz` |
| 4 | `blk23_K512_e16.npz` |
| 5 | `blk23_K1024_e16.npz` |
| 6 | `blk23_K64_e8.npz` |
| 7 | `blk23_K256_e8.npz` |
| 8 | `blk23_K512_e8.npz` |

## Online evaluation

Run one full task plan:

```bash
PYTHONPATH="$SOURCE_ROOT" CUDA_VISIBLE_DEVICES=0 \
poetry -C "$PROJECT_ROOT" run cofai-eval \
  "$SOURCE_ROOT/examples/orfc_2446/plan/dinov3/ade20k-val__dinov3-vitl16-slot24__ORFC__semseg.yaml" \
  args.cuda=true \
  args.real=true \
  args.multi_run=true \
  "args.output_dir=$PROJECT_ROOT/logs/orfc_2446/dinov3"
```

Run both slot24 plans:

```bash
GPU_IDS=0,1 \
bash examples/orfc_2446/dinov3/scripts/run_eval_release_ctc.sh
```

Results are written below
`$PROJECT_ROOT/logs/orfc_2446/dinov3/<plan>/q<quality>/`.

## Slot6 evaluation

Slot6 codes the output of block 5 and runs blocks 6–23 on the decoding side. The three plans use the common [DINOv3 test conditions](../../ctc/README-DINOv3.md) with the ADE20K-trained SoftPQ artifacts. Segmentation and depth use the final-layer features; detection uses taps at blocks 5, 11, 17, and 23.

Run the four real rANS qualities for each task:

```bash
for plan in \
  ade20k-val__dinov3-vitl16-slot6__ORFC__semseg \
  nyuv2-val__dinov3-vitl16-slot6__ORFC__depth \
  coco-val1k__dinov3-vitl16-slot6__ORFC__det
do
  PYTHONPATH="$SOURCE_ROOT" CUDA_VISIBLE_DEVICES=0 \
  poetry -C "$PROJECT_ROOT" run cofai-eval \
    "$SOURCE_ROOT/examples/orfc_2446/plan/dinov3/$plan.yaml" \
    args.multi_run=true args.cuda=true args.real=true
done
```

Aggregate results are written to `$PROJECT_ROOT/logs/<plan>/summary.json`.

### Author-reported slot6 results

The following values were reported in the original contribution. They predate the COCO evaluation-set and padding-coordinate corrections and are not the corrected reproduction results.

| q | Codebook | ADE bpp / mIoU | NYUv2 bpp / RMSE | COCO 1k bpp / bpfp / AP / AP50 |
|---|------|----------------|------------------|--------------------------------|
| 1 | K4e32 | 0.252 / 0.4918 | 0.234 / 0.3973 | 1.266 / 0.0553 / 0.4489 / 0.6544 |
| 2 | K8e32 | 0.377 / 0.5110 | 0.360 / 0.3904 | 1.984 / 0.0867 / 0.4932 / 0.6982 |
| 3 | K16e32 | 0.504 / 0.5140 | 0.482 / 0.3797 | 2.711 / 0.1183 / 0.5060 / 0.7155 |
| 4 | K256e32 | 1.024 / 0.5275 | 0.993 / 0.3626 | 5.633 / 0.2459 / 0.5359 / 0.7437 |

## Offline training and diagnostics

Extract a deterministic 5,050-image ADE20K feature corpus (5,000 train + 50
validation):

```bash
PYTHONPATH="$SOURCE_ROOT" CUDA_VISIBLE_DEVICES=0 \
poetry -C "$PROJECT_ROOT" run python \
  "$SOURCE_ROOT/examples/orfc_2446/dinov3/offline/extract_features_dinov3.py" \
  --dataset ade \
  --slot 24 \
  --max_images 5050 \
  --output_dir "$PROJECT_ROOT/features/orfc_2446/dinov3/ade"
```

Train the eight published configurations. Each run writes its runtime `.npz`
under `weights/orfc_2446/dinov3_vitl16/`:

```bash
for spec in 16:32 256:32 1024:32 512:16 1024:16 64:8 256:8 512:8; do
  K="${spec%:*}" EMB="${spec#*:}" GPU=0 \
  bash examples/orfc_2446/dinov3/scripts/run_train_frozen_tail.sh
done
```

Validate their identity metadata and package the exact filenames used by the
plans:

```bash
PYTHONPATH="$SOURCE_ROOT" poetry -C "$PROJECT_ROOT" run python \
  "$SOURCE_ROOT/examples/orfc_2446/dinov3/offline/package_release.py"
```

For a single development run, call the trainer directly:

```bash
PYTHONPATH="$SOURCE_ROOT" CUDA_VISIBLE_DEVICES=0 \
poetry -C "$PROJECT_ROOT" run python \
  "$SOURCE_ROOT/examples/orfc_2446/dinov3/offline/train_soft_pq_dinov3.py" \
  --config "$SOURCE_ROOT/examples/orfc_2446/dinov3/configs/dinov3_blk23.yaml" \
  --K 256 \
  --embedding_dim 32 \
  --feat_dir "$PROJECT_ROOT/features/orfc_2446/dinov3/ade"
```

Formal reference results always come from the plans and the standard
`cofai-eval` entrypoint.

The proposal-local replay command is an offline diagnostic for pre-extracted
features. It uses the same runtime codec and counts the actual label and
normalization streams:

```bash
CKPTS=weights/orfc_2446/dinov3_vitl16_ori/blk23_K256_e32.npz \
TASKS=semseg GPUS=0 \
bash examples/orfc_2446/dinov3/scripts/run_offline_replay.sh
```

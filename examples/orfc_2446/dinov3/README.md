# ORFC-2446 — DINOv3

The DINOv3 integration evaluates ViT-L/16 slot24 on:

- ADE20K semantic segmentation
- NYUv2 depth estimation

Both plans use the standard `DinoFeatureCodecModel` and
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
```

The manifest installs the DINOv3 backbone, datasets, task heads, and the eight
released ORFC-2446 artifacts consumed by the plans:

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

Each artifact contains `R`, `codebooks`, `pmf`, `norm_mode`, and `n_prefix`.

The two plans use the same integer quality mapping:

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

Run both plans:

```bash
GPU_IDS=0,1 \
bash examples/orfc_2446/dinov3/scripts/run_eval_release_ctc.sh
```

Results are written below
`$PROJECT_ROOT/logs/orfc_2446/dinov3/<plan>/q<quality>/`.

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

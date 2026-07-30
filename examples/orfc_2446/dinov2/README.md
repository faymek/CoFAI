# ORFC-2446 — DINOv2

The DINOv2 release covers seven split points:

- ViT-L/14: slots 06, 11, 16, and 21
- ViT-G/14: slots 10, 20, and 30

Each split point has one classification plan and one VOC segmentation plan.
All qualities for that split point are declared in the plan's `multi_run`.
Keys are consecutive integer quality IDs; the selected `K`, embedding size,
and training variant remain explicit in each artifact path.

## Environment and assets

Run commands from the worktree root while reusing the verified main environment:

```bash
SOURCE_ROOT="$PWD"
PROJECT_ROOT="$(awk -F= '/^[[:space:]]*PROJECT_ROOT[[:space:]]*=/ {
  sub(/^[^=]*=[[:space:]]*/, ""); print; exit
}' .env)"

poetry -C "$PROJECT_ROOT" run cofai-download \
  "$SOURCE_ROOT/examples/orfc_2446/dinov2/dinov2-orfc_2446.manifest.txt"

WEIGHTS_ROOT="$PROJECT_ROOT/weights/orfc_2446" \
bash "$SOURCE_ROOT/examples/orfc_2446/dinov2/scripts/unzip_softpq_weights.sh"
unzip "$PROJECT_ROOT/data/ImageNet_val_sel500.zip" -d "$PROJECT_ROOT/data"
unzip "$PROJECT_ROOT/data/VOC2012_sel100.zip" -d "$PROJECT_ROOT/data"
```

The manifest downloads the backbone, task heads, datasets, and ORFC-2446
archives below `PROJECT_ROOT`; the extraction command installs the `.npz`
artifacts there.

## Online evaluation

Run every quality in one split-point plan:

```bash
PYTHONPATH="$SOURCE_ROOT" CUDA_VISIBLE_DEVICES=0 \
poetry -C "$PROJECT_ROOT" run cofai-eval \
  "$SOURCE_ROOT/examples/orfc_2446/plan/dinov2/imagenet-sel500__dinov2-vitl14-slot06__ORFC__cls.yaml" \
  args.cuda=true \
  args.real=true \
  args.multi_run=true \
  "args.output_dir=$PROJECT_ROOT/logs/orfc_2446/dinov2"
```

Run all 14 plans and all 90 quality points:

```bash
GPU_IDS=0,1,2,3 \
bash examples/orfc_2446/dinov2/scripts/run_online_eval_all.sh
```

Set `DRY_RUN=1` to inspect the generated commands without inference. Results
are written below `$PROJECT_ROOT/logs/orfc_2446/dinov2/<plan>/q<quality>/`.

## Offline training and export

Training implementation and checkpoint serialization live under the proposal,
not in `cofai/entropy_models`:

```bash
PYTHONPATH="$SOURCE_ROOT" CUDA_VISIBLE_DEVICES=0 \
poetry -C "$PROJECT_ROOT" run python \
  "$SOURCE_ROOT/examples/orfc_2446/dinov2/offline/train_soft_pq.py" \
  --backbone dinov2_vitl14 \
  --layer blk10 \
  --K 64 \
  --embedding_dim 32 \
  --lmbda 0.5
```

The training entrypoint exports the runtime `.npz` at completion. To convert an
optional training checkpoint explicitly:

```bash
PYTHONPATH="$SOURCE_ROOT" poetry -C "$PROJECT_ROOT" run python \
  "$SOURCE_ROOT/examples/orfc_2446/dinov2/offline/export_softpq_npz.py" \
  --ckpt_path "$PROJECT_ROOT/weights/orfc_2446/<checkpoint>.pt" \
  --feat_root "$PROJECT_ROOT/features/orfc"
```

`offline/test_cls.py` and `offline/test_seg.py` replay pre-extracted features
through the same `OrthoRotationFeatureCodec` used by the online engine.

# PQFC — Product Quantization Feature Codec (M2488)

This example implements the transform-optional product-quantization route from **AI M2488**, which integrates the no-transform method from M2353 and the learnable orthogonal-transform method from M2446. It evaluates DINOv3 ViT-L/16 **slot24 / block 23** features on ADE20K segmentation and NYUv2 depth estimation.

The original [M2446 Soft-PQ examples](../orfc_2446/README.md) and [M2411 classic ORFC examples](../orfc/README.md) retain their own training recipes and evaluation guides. Common evaluation settings are documented in the [DINOv3 test conditions](../ctc/README-DINOv3.md).

Online evaluation uses `cofai.latent_codecs.OrthoRotationFeatureCodec`. The runtime artifact is a single `.npz` containing `R`, `codebooks`, and `pmf`, with normalization metadata; `use_transform=False` disables both rotations. Optional `.pt` files are training checkpoints, not evaluation artifacts.

Offline training reuses the existing SoftPQ and artifact tools in `examples/orfc_2446/offline/`. M2488-specific configuration, feature extraction and replay helpers live in this directory. The transform-enabled plans use the published M2446 weights without changing their paths or format.

## Layout

```text
examples/pqfc/
├── offline/                    # M2488 feature extraction and training entrypoints
├── plan/dinov3/                # PQFC and PQFC-noR evaluation plans
├── configs/dinov3_blk23.yaml    # M2488 configuration
├── lib/                        # M2488 offline helpers
├── dinov3-pqfc.manifest.txt     # M2488 slot24 resources
├── run_pqfc_dinov3.py           # M2488 offline replay and rate measurement
└── scripts/                    # M2488 prepare, extract, train, replay and evaluation
```

## DINOv3 slot24

| Variant | Plan marker | Weights |
|---|---|---|
| With orthogonal transform | `__PQFC__` | `weights/orfc_2446/dinov3_vitl16_ori/blk23_K{K}_e{d}.npz` (M2446 release) |
| Without orthogonal transform | `__PQFC-noR__` | `weights/pqfc/dinov3_vitl16_noR/blk23_K{K}_e{d}_no_transform.npz` |

The transform-enabled `__PQFC__` plans use the same published weights and evaluation settings as the M2446 slot24 release.

### Prepare

```bash
SOURCE_ROOT="$PWD"
PROJECT_ROOT="$(awk -F= '/^[[:space:]]*PROJECT_ROOT[[:space:]]*=/ {
  sub(/^[^=]*=[[:space:]]*/, ""); print; exit
}' .env)"
export SOURCE_ROOT PROJECT_ROOT PYTHONPATH="$SOURCE_ROOT"

bash examples/pqfc/scripts/prepare_dinov3.sh
```

The prepare script downloads the slot24 evaluation resources and extracts the transform-enabled release ZIP to the directory required by the plans. The seven no-transform NPZ files are downloaded directly to `weights/pqfc/dinov3_vitl16_noR/`. Data and weight paths are resolved against `PROJECT_ROOT`.

There are eight transform-enabled qualities and seven no-transform qualities (no `K512_e8`):

| quality | With transform | Without transform |
| ---: | --- | --- |
| 1 | `blk23_K16_e32.npz` | `blk23_K16_e32_no_transform.npz` |
| 2 | `blk23_K256_e32.npz` | `blk23_K256_e32_no_transform.npz` |
| 3 | `blk23_K1024_e32.npz` | `blk23_K1024_e32_no_transform.npz` |
| 4 | `blk23_K512_e16.npz` | `blk23_K512_e16_no_transform.npz` |
| 5 | `blk23_K1024_e16.npz` | `blk23_K1024_e16_no_transform.npz` |
| 6 | `blk23_K64_e8.npz` | `blk23_K64_e8_no_transform.npz` |
| 7 | `blk23_K256_e8.npz` | `blk23_K256_e8_no_transform.npz` |
| 8 | `blk23_K512_e8.npz` | — |

### Offline training

The scripts set `PYTHONPATH` to this checkout and import SoftPQ from `examples.orfc_2446.offline`. In this M2488 training recipe, `--use_transform` selects soft assignments and `--no_transform` selects hard assignments.

```bash
# ImageNet validation features (T=201)
IMAGENET_ROOT=/path/to/imagenet/val PATHNAME_LIST=/path/to/imagenet_selected_pathname5000.txt \
  bash examples/pqfc/scripts/extract_train.sh --max_images 5000

# With transform (default: OrthogonalTransform + soft PQ)
GPU=0 bash examples/pqfc/scripts/train_dinov3.sh --K 256 --embedding_dim 32 --epochs 100

# Without transform (hard PQ, exported to the noR directory)
GPU=0 WEIGHTS_DIR=weights/pqfc/dinov3_vitl16_noR \
  bash examples/pqfc/scripts/train_dinov3.sh \
    --K 256 --embedding_dim 32 --no_transform --lmbda 0.0
```

Offline replay requires validation features extracted by `extract_val.sh`:

```bash
bash examples/pqfc/scripts/extract_val.sh
bash examples/pqfc/scripts/replay_dinov3.sh --task semseg \
  --ckpt_path weights/pqfc/dinov3_vitl16/<ckpt>.npz
```

### Online evaluation

Each GPU runs one plan at a time. CoFAI `multi_run` evaluates the qualities sequentially within each plan. `PROJECT_ROOT` points to the data and weights; `PYTHONPATH` points to this worktree.

```bash
# With transform
GPU_IDS=0,1 bash examples/pqfc/scripts/run_eval_ctc.sh

# Without transform
VARIANT=noR GPU_IDS=0,1 bash examples/pqfc/scripts/run_eval_ctc.sh

TASK=semseg GPU_IDS=0 bash examples/pqfc/scripts/run_eval_ctc.sh

# Preview selected plans and full commands
DRY_RUN=1 VARIANT=both GPU_IDS=0,1 bash examples/pqfc/scripts/run_eval_ctc.sh
```

CoFAI writes results to `$OUTPUT_DIR/<plan>/q<quality>/result.json` and `<plan>/summary.json`, with runner logs in `$OUTPUT_DIR/runner_logs/`. Missing weights or failed evaluations return a nonzero exit code.

To run one plan directly:

```bash
PYTHONPATH="$SOURCE_ROOT" CUDA_VISIBLE_DEVICES=0 \
poetry -C "$PROJECT_ROOT" run cofai-eval \
  "$SOURCE_ROOT/examples/pqfc/plan/dinov3/ade20k-val__dinov3-vitl16-slot24__PQFC__semseg.yaml" \
  args.cuda=true args.real=true args.multi_run=true \
  "args.output_dir=$PROJECT_ROOT/logs/pqfc/dinov3"
```

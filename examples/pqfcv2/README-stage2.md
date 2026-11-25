# PQFCv2 stage 2: frozen stage-1 baseline

Stage 2 trains a token analysis adapter and a synthesis adapter. The stage-1
rotation, PQ codebooks, probability tables, and prefix/patch RMS are fixed.
For the DINOv3 ViT-L/16 features used here, the five prefix tokens are kept:
the full sequence is RMS-normalized and split into blocks of 241 tokens, each
mapped to N latent tokens. The last block is zero-padded when needed; decoded
output is cropped to the original sequence length. Training checkpoints bind
the stage-1 artifact and backbone weights by SHA-256.

## Requirements and inputs

Install the project using the repository's installation instructions, e.g.
`pip install -e .`, in an environment with the required PyTorch/CUDA and rANS
dependencies. The data, model weights and feature files are external artifacts.
You need:

- A stage-1 `.npz` containing `R`, `codebooks`, `pmf`,
  `prefix_position_channel_rms` and `patch_channel_rms`.
- DINOv3 ViT-L/16 backbone weights in safetensors format, matching the
  backbone used to extract the features and train stage 1.
- ADE and NYU extracted features, each a directory of `.npy` files with shape
  `[tokens, 1024]`, including the five prefix tokens. An existing
  `indexed-npy-tar-v1` JSON index may be used instead of a directory.
- The ADE/NYU training image lists. By default these are the lists tracked at
  `examples/vq_ufc/dinov3/data_lists/ade_1w.txt` and `nyu_train_1w.txt`.
  Feature filenames must match these lists; use `--ade-list` and `--nyu-list`
  to select different lists. Names across both domains must be unique.
- For evaluation: NYUv2 and ADE20K data, plus the matching DINOv3 depth and
  segmentation head checkpoints. NYU's root must include `nyu_test.txt` and
  its referenced samples; ADE's root must contain `images/validation` and
  `annotations/validation`.

The existing `examples/pqfcv2/extract_features.py` extracts `.npy` features
from images. For example, from the repository root, replace the placeholders
below with local paths:

```bash
python examples/pqfcv2/extract_features.py \
  --project-root "$PWD" \
  --ade-list examples/vq_ufc/dinov3/data_lists/ade_1w.txt \
  --nyu-list examples/vq_ufc/dinov3/data_lists/nyu_train_1w.txt \
  --ade-root /path/to/ADEChallengeData2016/images/training \
  --nyu-root /path/to/NYU \
  --ade-out /path/to/ade_features --nyu-out /path/to/nyu_features \
  --backbone /path/to/dinov3_backbone.safetensors
```

The supplied NPZ does not necessarily record its training split. Check stage-1
split and feature provenance separately before treating validation as held out.

## Training

Set paths to your own artifacts, then choose N and lambda:

```bash
export STAGE1=/path/to/stage1.npz
export BACKBONE=/path/to/dinov3_backbone.safetensors
export ADE_FEATURES=/path/to/ade_features
export NYU_FEATURES=/path/to/nyu_features
export OUTPUT_DIR=/path/to/output/stage2_N72_L100
bash examples/pqfcv2/scripts/train_stage2.sh 72 --lmbda 100
```

`ADE_FEATURES` and `NYU_FEATURES` accept either feature directories or index
JSON files. If omitted, they default to
`$PROJECT_ROOT/features/pqfcv2/dinov3/ade` and `.../nyu`; `ADE_INDEX` and
`NYU_INDEX` remain supported as aliases. The script's default `PROJECT_ROOT`
is the repository root and can be overridden. `PYTHON_BIN` selects the Python
interpreter. All paths may be outside the repository.

Training defaults are in `plan/stage2_baseline.yaml`. The launcher reads that
file; set `CONFIG` to use a different YAML file, or pass CLI options (such as
`--lmbda`) to override its values. The original baseline uses a 50-image
validation split (seed 42). N values used
in experiments are 20, 48, 72, 96, 120, 144, 192, and 240. Defaults: AdamW,
learning rate 0.001, temperature 0.5, 8 epochs, batch size 4, gradient clipping
at 1.0, lambda 100. The objective is estimated bits plus lambda times the sum
of squared differences after backbone LayerNorm. `best.pt` is selected by this
validation loss; downstream task metrics can rank checkpoints differently.

Use a distinct output directory per run. To resume, add `--resume` with the
same inputs, hyperparameters and output directory. The trainer checks artifact
hashes, token dimensions and split on resume; it does not check every
hyperparameter. The output contains `best.pt`, `last.pt`, `split.json`, and
`history.jsonl`.

## Evaluation

The following uses the same `STAGE1` and `BACKBONE` variables as training and
runs both NYU and ADE tests with real rANS streams:

```bash
export CHECKPOINT="$OUTPUT_DIR/best.pt"
export DEPTH_HEAD=/path/to/dinov3_depth_head.pth
export SEMSEG_HEAD=/path/to/dinov3_semseg_head.pth
export NYU_ROOT=/path/to/NYU
export ADE_ROOT=/path/to/ADEChallengeData2016
export EVAL_OUTPUT=/path/to/output/eval_N72_L100
bash examples/pqfcv2/scripts/eval_stage2.sh 72
```

Alternatively, `DINOV3_WEIGHTS` may point to a directory with the default
`backbone/`, `dpt_head/`, and `semseg_head/` layout. `CHECKPOINT`,
`EVAL_OUTPUT`, `STAGE1`, `BACKBONE`, `DEPTH_HEAD`, `SEMSEG_HEAD`, `NYU_ROOT`
and `ADE_ROOT` can each be overridden independently. Additional Hydra overrides
may be passed after N, for example `args.max_samples=1` for a smoke test.

The eval plans are shared across N values; the checkpoint determines the
actual N. The eval script supplies the required checkpoint path. Evaluation reports
actual coded-stream bpp and downstream metrics; model and image metadata bytes
are not included. Estimated rate and rANS rate can differ.

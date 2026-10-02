# VQ-UFC

VQ-UFC encodes DINOv3 features, including the five CLS/REG prefix tokens. Each prefix position has its own per-channel RMS; patch positions share one per-channel RMS vector. The model uses one shared VQ codebook and reuses CoFAI's `SoftmaxPrior` and `DiscreteEntropyModel` for rANS coding.

## Layout

```text
cofai/entropy_models/vq_ufc_model.py       # Quantization and token transforms
cofai/latent_codecs/vq_ufc.py              # CoFAI codec adapter
cofai/utils/utils_vqufc.py                 # Feature splits and RMS helpers
examples/vq_ufc/dinov3/plan/*VQUFC*.yaml   # Evaluation plans
examples/vq_ufc/dinov3/all_token_rms.py    # RMS command-line entry point
examples/vq_ufc/dinov3/train_vq_ufc_dinov3.py
examples/vq_ufc/train_stage{1,2}.sh
```

## Online Evaluation (Engine)

The engine runs the DINOv3 backbone, VQ-UFC, and a downstream task head. Invoke `cofai-eval`, which maps to `cofai.engine.run_eval:main`. The codec and its Python dependencies are imported from `cofai`, not from `examples`.

### Requirements

1. Set up the environment as described in `README_DEV.md`. Provide the evaluation dataset, DINOv3 backbone weights, the corresponding task-head weights, and the VQ-UFC checkpoints.
2. Use a CoFAI worktree containing `DinoFeatureCodecModel`, the DINOv3 backbone and task head, the dataset loader, and a matching evaluation plan.
3. The checkpoint must contain the model configuration, codebook, entropy logits, and prefix/patch RMS. The separate training `.npz` is not needed for evaluation.

VQ-UFC checkpoints: [download link](https://pan.quark.cn/s/0eeea25c643c).

### Commands

Run from a CoFAI worktree with the DINOv3 evaluation components. Choose one of the plans in `examples/vq_ufc/dinov3/plan`; the plan is a YAML configuration file, not an `examples` Python module. Place the flat checkpoint files `T01_N20.pth.tar` through `T08_N240.pth.tar` and `S01_N20.pth.tar` through `S08_N240.pth.tar` under `$PROJECT_ROOT/weights/vq_ufc/dinov3/rd_points/`. Each plan explicitly names all 16 checkpoint paths and runs them sequentially by default.

| Dataset / task | Plan in this branch |
| --- | --- |
| NYUv2 / depth | `examples/vq_ufc/dinov3/plan/nyuv2-val__dinov3-vitl16-slot24__VQUFC__depth.yaml` |
| ADE20K / semantic segmentation | `examples/vq_ufc/dinov3/plan/ade20k-val__dinov3-vitl16-slot24__VQUFC__semseg.yaml` |

```bash
export PROJECT_ROOT="$PWD"
PLAN="nyuv2-val__dinov3-vitl16-slot24__VQUFC__depth"
RUN_NAME="dataset-task"

cofai-eval --config-dir="$PROJECT_ROOT/examples/vq_ufc/dinov3/plan" --config-name="$PLAN" \
  args.cuda=true args.real=true \
  args.output_dir="$PROJECT_ROOT/logs/vq_ufc/$RUN_NAME"
```

Each plan evaluates all 16 checkpoints sequentially for its task. Results are written to `args.output_dir/<plan-name>/qT01_N20/result.json` (and similarly for the other points), with an aggregated `summary.json` in the plan directory. Set `args.multi_run=false` to run only the plan's default checkpoint, or override `model.dino_codec.checkpoint_path` to run another single point. `args.real=true` calls the codec's `compress()` and `decompress()` methods and counts the actual bitstream size. `bpp` is common to both plans; task metrics include depth RMSE/Abs Rel/δ₁ for NYUv2 and mIoU for ADE20K. Add `args.max_samples=1` for a one-image-per-point smoke test.

## Offline Training

Training uses pre-extracted DINOv3 `.npy` features. `train_vq_ufc_dinov3.py` is the configurable training program; both shell scripts call it with preset arguments. First run `all_token_rms.py` to produce an RMS artifact bound to the training split. `train_stage1.sh` trains the shared codebook and entropy prior without a token transform (NYU defaults). `train_stage2.sh <nyu|ade> <latent-tokens>` loads the Stage 1 checkpoint, freezes the codebook and prior, and trains the token transform. Its `BASE_CHECKPOINT` defaults to the NYU Stage 1 result even for ADE; override it when needed. RMS artifacts and checkpoints are generated files and are not committed.

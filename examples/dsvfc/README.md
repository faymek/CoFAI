# AI M2482 DS-VFC

DS-VFC (DeepStack-aware Visual Feature Codec) compresses the Qwen3-VL-8B-Instruct visual tower at slot 9, immediately after block 8 and before the first DeepStack merger. It combines two pre/post ViT blocks, convolutional analysis/synthesis transforms, a hyperprior, and 16 uniform space-channel context groups. The codec has 108,796,736 parameters (`h_dim=1152`, `y_dim=256`, `z_dim=192`).

Source: **AI M2482: 2026年8月, “Qwen3-VL 视觉塔中间层特征编码方法”**, revision `AI M2482_v2.docx`, submitted by Shanghai Jiao Tong University and Ant Group. This integration retains the proposal's `loss_1` checkpoints. The author's later `loss_2` training variant is outside this example.

The implementation reuses CoFAI's existing `Qwen3VLBackbone`, `Qwen3vlFeatureCodecModel`, checkpoint loader, and evaluation engine. MMStar and ODinW-13 use the same DS-VFC codec and checkpoints. `cofai/latent_codecs/dsvfc.py` exposes the standard `(1,H*W,C)` / `(H,W)` codec interface and preserves the checkpoint parameter names. It emits the engine's standard per-unit `y`/`z` streams; batch size 1 is required.

## Resources

Follow the [Qwen3-VL CTC guide](../ctc/README-QWEN3VL.md) for MMStar, the Qwen3-VL-8B-Instruct backbone, and the existing Bypass reference. Set `PROJECT_ROOT` as documented there. Download the five proposal checkpoints from CoFAI-share:

```bash
poetry run cofai-download examples/dsvfc/dsvfc.manifest.txt
```

The original checkpoint files are saved under `${PROJECT_ROOT}/weights/DS-VFC/loss_1/`; no extraction is needed. Each contains codec parameters under `qwen_codec.*`, plus optimizer/RNG state. The plan uses the existing engine checkpoint loader with `weights_only: false` and `strict: false`, because the pretrained backbone is not stored in those files.

| Checkpoint | Epoch |
| --- | ---: |
| `DS-VFC-qwen3vl-8b-ins-slot9-Q0.6-epoch012.pth` | 12 |
| `DS-VFC-qwen3vl-8b-ins-slot9-Q0.6-epoch016.pth` | 16 |
| `DS-VFC-qwen3vl-8b-ins-slot9-Q0.4-epoch016.pth` | 16 |
| `DS-VFC-qwen3vl-8b-ins-slot9-Q0.2-epoch016.pth` | 16 |
| `DS-VFC-qwen3vl-8b-ins-slot9-Q0.2-epoch028.pth` | 28 |

`Q` denotes the independent training rate weight, not an inference QP. The duplicate Q0.2/Q0.6 files are different epochs.

## MMStar evaluation

The default runs all 1500 questions with the Q0.4 epoch16 checkpoint and real entropy coding:

```bash
poetry run cofai-eval examples/dsvfc/plan/mmstar__qwen3vl-8b-ins-slot9__DSVFC__vqa.yaml
```

To evaluate all five checkpoints, use the existing multi-run mechanism:

```bash
poetry run cofai-eval examples/dsvfc/plan/mmstar__qwen3vl-8b-ins-slot9__DSVFC__vqa.yaml \
  args.multi_run=true
```

Results follow the standard `logs/<plan-name>/result.json` layout; multi-run writes per-checkpoint subdirectories and a `summary.json`. `args.real=true` counts the actual `y` and `z` entropy payload bytes. `args.real=false` uses likelihood estimates. Use the existing CTC plan for the uncompressed reference:

```bash
poetry run cofai-eval conf/plan/mmstar__qwen3vl-8b-ins-slot9__Bypass__vqa.yaml
```

The proposal used likelihood-estimated rates; the default plan measures actual entropy payloads. MMStar keeps CoFAI's existing CTC backbone and attention path, which differs from the proposal's SDPA/Transformers 4.x inference environment.

## ODinW-13 detection

Download the complete dataset ZIP, then extract it under the resource root. Export `PROJECT_ROOT` in your shell to the same resource directory used by the plans:

```bash
poetry run cofai-download examples/dsvfc/odinw13.manifest.txt
unzip -n "${PROJECT_ROOT}/data/ODinW-13.zip" -d "${PROJECT_ROOT}/data"
```

The archive creates `${PROJECT_ROOT}/data/ODinW-13/`, including all 13 source sub-datasets, their original splits, and `odinw13_config.py`. The plans evaluate all 4938 validation images. They reuse the five codec files listed above; no task-specific codec or detection-head weights are needed.

```bash
poetry run cofai-eval examples/dsvfc/plan/odinw13__qwen3vl-8b-ins-slot9__Bypass__det.yaml
poetry run cofai-eval examples/dsvfc/plan/odinw13__qwen3vl-8b-ins-slot9__DSVFC__det.yaml
```

The default DS-VFC checkpoint is Q0.4 epoch16 for both tasks. `_dsvfc-slot9.yaml` defines their shared codec, load path, and five-checkpoint `args.multi_run=true` sweep. `Q` remains a training coefficient rather than an inference QP.

Detection retains the author's prompt, 768–12800 visual-token budget, 2048-token generation limit, greedy decoding, and SDPA attention. MMStar keeps its existing CTC attention setting. Generated boxes use normalized 0–1000 coordinates; the metric maps them to original-image pixels and averages COCO mAP across the 13 sub-datasets, with the proposal's `maxDets=[100,300,1000]` and per-subset rounding. Results include per-subset metrics and generated text for cross-checking.

CoFAI is a PyTorch library and evaluation platform for multi-purpose compression research, providing a public testing environment for evaluating feature coding methods.

CoFAI currently provides:

* Multi-Purpose Compression (MPC) framework - a coding architecture designed to prioritize machine vision while retaining compatibility with human visual perception
* Feature Coding for Large Models (LaMoFC) framework - a feature coding framework for distributed large model deployments
* a **unified evaluation engine** (`cofai-eval`, Hydra plans under `conf/plan/`) for reproducible benchmarks—see [docs/engine.md](docs/engine.md)
* examples and test setups for comparing compression methods

## Installation

CoFAI supports Python 3.10+, PyTorch 2.4.0 and CUDA 12.1.

### Using Poetry (Recommended)

Poetry helps manage version-pinned virtual environments. First, install [Poetry](https://python-poetry.org/docs/#installation):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then, create the virtual environment and install the required Python packages:

```bash
cd CoFAI

# Install Python packages to new virtual environment.
poetry install
echo "Virtual environment created in $(poetry env list --full-path)"

# Link to local CoFAI source code.
poetry run pip install --editable .
```

The project uses a `.env` file (via `PROJECT_ROOT`) to avoid hardcoding absolute paths in configs. After `poetry install`, generate it with:

```bash
poetry run python install.py
```


## Documentation

* [Hosted documentation](https://faymek.github.io/CoFAI)
* [Evaluation engine](docs/engine.md) — plans, `cofai-eval`, data contracts
* [Migration Guide (mpcompress → cofai)](MIGRATION.md)

## Dataset and Weights Preparation

We provide publicly available datasets and weights via the following link:

Link: https://medialab.sjtu.edu.cn/files/CoFAI-share/

Please refer to the examples to download the needed resources and extract them into the current directory. The resulting directory structure should look like this:

```
CoFAI/
├─ cofai/
├─ data/
│   ├─ ADEChallengeData2016/
│   ├─ ImageNet_val_sel2k/
│   └─ VOC2012/
├─ weights/
│   ├─ dinov2/
│   └─ MPC/
```

The full directory structure is organized as follows:
```
CoFAI/
├─ cofai/        # source code
├─ conf/         # Hydra defaults and evaluation plans (`plan/*.yaml`)
├─ data/         # dataset libraries
├─ features/     # extracted features
├─ logs/         # eval logs
├─ runs/         # training logs and checkpoints
├─ models/       # vision model libraries, reserved for future use
├─ weights/      # pre-trained weights and released checkpoints
```

## Usage

Coding examples can be found in the `examples/` directory.

### Eval Engine

We recently added a unified evaluation **engine** (`cofai/engine/`) so that datasets, preprocessing, tasks and metrics, models, and checkpoints are wired together through declarative **plans**—making it easier to reproduce runs and compare methods under the same protocol. Command-line usage (`poetry run cofai-eval`), plan layout (`conf/plan/`), and the full data/runtime contract are documented in **[docs/engine.md](docs/engine.md)**.

### Testing LaMoFC

> "Feature Coding in the Era of Large Models: Dataset, Test Conditions, and Benchmark"

Large models are often partitioned and deployed across multiple devices. In such distributed setups, intermediate features must be encoded and transmitted between nodes. LaMoFC is a feature coding framework designed to minimize the required bitrate under a specified task accuracy constraint, or conversely, to maximize task accuracy under a given bitrate limit.

For implementation details and usage examples, please refer to the directory `examples/lamofc/` and its dedicated [README](examples/lamofc/README.md).

This test script is adapted from the original [LaMoFC repository](https://github.com/chansongoal/LaMoFC). Note that results may vary slightly depending on the versions of Python libraries used.

### Testing MPC

The Multi-Purpose Compression (MPC) framework is a coding architecture designed to prioritize machine vision while retaining compatibility with human visual perception. It extracts general-purpose visual features at the encoder and enables low-complexity decoding of task-relevant information at the decoder. The framework adopts a multi-branch structure to support on-demand bitstream extraction, making it adaptable to a variety of downstream tasks.

For implementation details and usage examples, please refer to the directory `examples/mpc/` and its dedicated [README](examples/mpc/README.md).

### Testing RAE (AITISA AI M2417)

RAE is currently provided as an extra capability under the MPC evaluation setup: it uses the **same codec** as MPC, but decodes images with an **RAE decoder** (image reconstruction from the coded representation).

RAE benchmarks can be run with **`poetry run cofai-eval`** and plans under `conf/plan/`; see **[examples/mpc/README.md](examples/mpc/README.md)** for examples.

### Testing VQFC (AITISA AI M2353)

> "Transform-Free Feature Coding via Entropy-Constrained Vector Quantization" (AAAI 2026)

VQFC proposes a **transform-free** pipeline that directly encodes features via vector quantization and an entropy model, jointly learned for end-to-end optimization. The method achieves comparable performance compared to transform-based baselines (LaMoFC) while **significantly reducing encoding and decoding complexity**. Original Code: [VQFC](https://github.com/xxii111/FCVQ). It is integrated into the LaMoFC pipeline to align the current framework.

For implementation details and usage examples, please refer to the directory `examples/vqfc/` and its dedicated [README](examples/vqfc/README.md).

### Testing VTC (AITISA AI M2418)

The **Visual Token Codec (VTC)** compresses global tokens and patch tokens separately. The key is that, for the dominant patch tokens, the method uses a spatial–channel context model to explicitly capture their spatial correlation. Experiments on image classification and segmentation tasks show that VTC significantly outperforms existing methods.

See [`conf/model/VTC-*.yaml`](conf/model/) for VTC model presets; run **`poetry run cofai-eval`** following [docs/engine.md](docs/engine.md). Codec implementation: [`cofai/latent_codecs/vtc.py`](cofai/latent_codecs/vtc.py).

### Testing CAVC (AITISA AI M2328)

CAVC is a video-feature joint coding framework proposed by AI M2328. It regulates video pixel distributions through context prompts and visual feature guidance, enabling targeted adaptation to different coding scenarios, such as human perceptual optimization, objective fidelity optimization, and machine vision tasks including object detection. During training, it adopts an advanced end-to-end compression proxy network to ensure effective gradient backpropagation; during inference, the compressor can be replaced with any end-to-end network or standard coding tool, providing strong compatibility.

For implementation details and usage examples, please refer to the directory `examples/cavc/` and its dedicated [README](examples/cavc/README.md).

### Testing RFC (AITISA AI M2268)

RFC studies a feature disentanglement and compression approach for multi-task ViT models. To extract task-relevant knowledge, it introduces rate constraints and task-specific losses to encourage the network to discard irrelevant information while preserving task-related information. To improve flexibility, it adopts a simple strategy: fine-tune a task-specific client-side network for each task, while different tasks share the same cloud-side network. This design can be conveniently deployed and adjusted by distributing different client-side model parameters.

For implementation details and usage examples, please refer to the directory `examples/rfc/` and its dedicated [README](examples/rfc/README.md).

## Acknowledgement

Special thanks to Donghui Feng, Bo Gao, Qingyue Ling, Fengxi Zhang and Zekai Liu, for their valuable contributions in building this test platform.

Special thanks to Yifan Ma, Qiaoxi Chen, and Yenan Xu, for their valuable contributions in building LaMoFC.

## Related Links

* [CompressAI repository](https://github.com/microsoft/CompressAI)
* [LaMoFC repository](https://github.com/chansongoal/LaMoFC)
* [DINOv2 repository](https://github.com/facebookresearch/dinov2)
* [VTM repository](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM)

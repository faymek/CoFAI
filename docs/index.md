# CoFAI Documentation

CoFAI (Coding for AI) is a framework for coding three types of visual representation: structured data, foundation-model features, and image pixels. These representations can be coded independently or together to reduce storage and transmission costs while preserving the information needed for AI perception, understanding, and generation.

![CoFAI framework: structured data, foundation-model features, image pixels, context, and downstream tasks](assets/cofai-framework.svg)

Feature coding is the current primary research focus of the reference software. It compresses intermediate features at a model split point for transmission between devices or storage and later reuse.

## Feature Coding and Deployment

![Feature coding pipeline and transmission-oriented or storage-oriented deployment](assets/cofai-feature-deployment.svg)

A model prefix extracts features, a codec compresses and reconstructs them, and the model suffix and task module produce the downstream result. Transmission-oriented deployment exchanges features between a device and a server; storage-oriented deployment saves them for later use. The split point and feature reuse capability depend on the model and coding method.

The documentation below separates framework concepts, current implementations, evaluation contracts, and API references from method-specific reproduction instructions.

## Start Here

| Guide | What it explains |
|---|---|
| [CoFAI Framework](framework.md) | Representation types, context, coding scheme levels, DU/AU organization, and encoder semantic conventions |
| [Reference software implementation](reference_software.md) | Backbones, codecs, heads, slots, multi-stream DUs, and token grouping in current code |
| [Evaluation engine](engine.md) | Plans, component building, data contracts, and reproducible evaluation |

Method-specific data, weights, training instructions, and commands are documented under `examples/<method>/`. The repository's root `README.md` provides installation instructions, a first evaluation example, and an overview of supported capabilities.

## Library API

- [cofai.models](api/models.md)
- [cofai.backbone](api/backbone.md)
- [cofai.latent_codecs](api/latent_codecs.md)
- [cofai.index_codecs](api/index_codecs.md)
- [cofai.token_grouping](api/token_grouping.md)
- [cofai.transforms](api/transforms.md)
- [cofai.engine](api/engine.md)
- [cofai.heads](api/heads.md)
- [cofai.layers](api/layers.md)
- [cofai.datasets](api/datasets.md)
- [cofai.losses](api/losses.md)
- [cofai.metrics](api/metrics.md)
- [cofai.utils](api/utils/utils.md)
- [Data and weight downloads](api/utils/download.md)

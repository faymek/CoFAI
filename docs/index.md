# CoFAI Documentation

CoFAI is a multi-representation coding framework and PyTorch reference software for AI perception, understanding, and generation. The documentation separates framework concepts, current implementations, evaluation contracts, and API references from method-specific reproduction instructions.

## Start Here

| Guide | What it explains |
|---|---|
| [Framework concepts](framework.md) | Representation types, context, coding scheme levels, DU/AU organization, and encoder semantic conventions |
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

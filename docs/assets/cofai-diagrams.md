# README framework diagrams

| View | Display asset | Editable source |
|---|---|---|
| README: overall three-branch framework | `cofai-framework.svg` | `cofai-framework.pptx` |
| README: feature coding and deployment | `cofai-feature-deployment.svg` | `cofai-feature-deployment.pptx` |
| Framework guide: earlier overall schematic | `cofai-overview.svg` | `cofai-overview.drawio` |
| Framework guide: feature and index interfaces | `cofai-feature-branch.svg` | `cofai-feature-branch.drawio` |

The README uses the two user-approved native PowerPoint drawings. Their text,
blocks, icons, and connectors are individually editable. The SVG exports are
fully vector, with text outlined. Edit the PPT files and regenerate the matching
SVGs together when changing the diagrams.

The framework guide retains the earlier diagrams, including the detailed
feature/index and optional processing interfaces. Open their `.drawio` sources
in diagrams.net to edit them. Implementation status is explained in
`../reference_software.md`.

The layout and content were adapted from the supplied drawing sources:

- `CoFAI-Framework-2606.drawio`: structured data, visual-model features, pixels,
  encoding/decoding, context, and AI tasks.
- `MPC-I2-API.drawio`: model prefix/suffix, the split feature interface, codec,
  and task module.
- `02-Framework-New3.drawio`: transmission and storage/reuse scenarios, optional
  preparation, and continuation of visual and vision-language tasks.

The README overview shows three representation branches, optional context,
Training/Inference scenarios, and Perception/Understanding/Generation tasks.
Its feature view shows a split ViT coding pipeline and two deployment patterns:
transmission for cloud analysis and storage for downstream reuse. Frozen-model
symbols describe the illustrated setup; shallow/deep splits and reuse are
examples, not requirements for every method. Feature/index streams and optional
pre/post-processing interfaces remain in the framework guide. Coded-unit syntax,
context availability, and current interface limitations remain in the detailed
documentation.

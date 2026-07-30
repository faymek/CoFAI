from .base import BackboneProtocol
from .frozen_tail import (
    Dinov3FrozenTail,
    FrozenTail,
)
from .tokenizer import VqganBackbone
from .timm import (
    Dinov2TimmBackbone,
    MAETimmBackbone,
    SigLIP2TimmBackbone,
    Dinov3TimmBackbone,
)
from .transformers import (
    Dinov2TransformersBackbone,
    MAETransformersBackbone,
    SigLIP2TransformersBackbone,
)
from .dinov2_org import Dinov2OrgBackbone
from .vgg import VGGBackbone, setup_vgg_feature_extractor, extract_vgg_features
from .qwen3vl import Qwen3VLBackbone

__all__ = [
    "BackboneProtocol",
    "FrozenTail",
    "Dinov3FrozenTail",
    "VqganBackbone",
    "Dinov2TimmBackbone",
    "Dinov3TimmBackbone",
    "MAETimmBackbone",
    "SigLIP2TimmBackbone",
    "Dinov2TransformersBackbone",
    "MAETransformersBackbone",
    "SigLIP2TransformersBackbone",
    "Dinov2OrgBackbone",
    "VGGBackbone",
    "setup_vgg_feature_extractor",
    "extract_vgg_features",
    "Qwen3VLBackbone",
]

backbone_tools = {
    "setup_vgg_feature_extractor": setup_vgg_feature_extractor,
    "extract_vgg_features": extract_vgg_features,
}

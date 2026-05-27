"""Project-level transform recipes and builders.

This package is the preferred place to define *composable* evaluation/training
transform pipelines (YAML-friendly dicts + small Python helpers).

Dataset-specific adapters may still live under `cofai.datasets`, but reusable
transform composition should be imported from here.
"""

from cofai.transforms.core import (
    AddIgnoreRegions,
    CenterCropImage,
    Normalize,
    PadImage,
    PadToMultiple,
    PhotoMetricDistortion,
    RandomCrop,
    RandomHorizontalFlip,
    RandomScaling,
    ResizeImage,
    ResizeToFit,
    SetImageAsOriginal,
    ToTensor,
)

from cofai.transforms.mlore import (  # legacy builders used by cofai.datasets.mlore
    MLoRETransformsConfig,
    get_mlore_transforms,
    get_mlore_transforms_cfg,
    get_mlore_train_transforms,
    get_mlore_val_transforms,
)


__all__ = [
    # core blocks (YAML-friendly)
    "CenterCropImage",
    "RandomScaling",
    "ResizeImage",
    "ResizeToFit",
    "RandomCrop",
    "RandomHorizontalFlip",
    "PhotoMetricDistortion",
    "Normalize",
    "PadImage",
    "PadToMultiple",
    "AddIgnoreRegions",
    "SetImageAsOriginal",
    "ToTensor",
    # legacy builders
    "MLoRETransformsConfig",
    "get_mlore_transforms_cfg",
    "get_mlore_train_transforms",
    "get_mlore_val_transforms",
    "get_mlore_transforms",
    "PadToMultiplePIL",
]

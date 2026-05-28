"""MLoRE/RFC transform *builders*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "MLoRETransformsConfig",
    "MLORE_ALIGNED_KEYS",
    "get_mlore_transforms_cfg",
    "get_mlore_train_transforms",
    "get_mlore_val_transforms",
    "get_mlore_transforms",
]

# Image + dense maps that must share geometry (pad/crop/flip/scale) for batched MLoRE/RFC loaders.
MLORE_ALIGNED_KEYS: Tuple[str, ...] = (
    "img",
    "semseg",
    "edge",
    "human_parts",
    "normals",
    "sal",
    "depth",
)


@dataclass(frozen=True)
class MLoRETransformsConfig:
    """
    Configurable transform recipe for RFC/MLoRE datasets.

    This is intentionally lightweight: it only controls *which* building blocks
    are included and their key parameters. The underlying blocks live in
    `cofai.transforms` (preferred YAML namespace; implementation in `cofai.transforms.core`).
    """

    # Target canvas sizes (H, W). For training, the crop size is usually the canvas.
    train_scale: Tuple[int, int] = (512, 512)
    test_scale: Tuple[int, int] = (512, 512)

    # Train-only augmentations
    enable_random_scaling: bool = True
    random_scaling_factors: Tuple[float, float] = (0.5, 2.0)
    random_scaling_discrete: bool = False

    enable_random_crop: bool = True
    random_crop_cat_max_ratio: float = 0.75

    enable_hflip: bool = True
    hflip_p: float = 0.5

    enable_photometric: bool = True

    # Common normalization
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def _as_hw(size: Union[int, Sequence[int], Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return (int(size[0]), int(size[1]))
    # Best-effort fallback for OmegaConf ListConfig-like
    try:
        seq = list(size)  # type: ignore[arg-type]
        if len(seq) == 2:
            return (int(seq[0]), int(seq[1]))
    except Exception:
        pass
    raise ValueError(f"Expected size as int or (H,W), got: {size!r}")


def _resolve_cfg(p: Any = None, cfg: Optional[MLoRETransformsConfig] = None) -> MLoRETransformsConfig:
    if cfg is not None:
        return cfg
    train_scale = (512, 512)
    test_scale = (512, 512)
    try:
        if p is not None and hasattr(p, "TRAIN") and hasattr(p.TRAIN, "SCALE"):
            train_scale = _as_hw(p.TRAIN.SCALE)
        if p is not None and hasattr(p, "TEST") and hasattr(p.TEST, "SCALE"):
            test_scale = _as_hw(p.TEST.SCALE)
    except Exception:
        # Keep defaults if p has unexpected shape.
        pass
    return MLoRETransformsConfig(train_scale=train_scale, test_scale=test_scale)


def get_mlore_train_transforms(p: Any = None, cfg: Optional[MLoRETransformsConfig] = None):
    """Return the RFC/MLoRE *training* transform pipeline (augment + pad + tensor)."""
    import torchvision
    from cofai import transforms

    c = _resolve_cfg(p, cfg)
    ops: List[Any] = []

    _aligned_keys = MLORE_ALIGNED_KEYS

    if c.enable_random_scaling:
        ops.append(
            transforms.RandomScaling(
                scale_factors=list(c.random_scaling_factors),
                discrete=bool(c.random_scaling_discrete),
                keys=_aligned_keys,
            )
        )
    if c.enable_random_crop:
        ops.append(
            transforms.RandomCrop(
                size=c.train_scale,
                cat_max_ratio=float(c.random_crop_cat_max_ratio),
                keys=_aligned_keys,
            )
        )
    if c.enable_hflip:
        ops.append(
            transforms.RandomHorizontalFlip(p=float(c.hflip_p), keys=_aligned_keys)
        )
    if c.enable_photometric:
        ops.append(transforms.PhotoMetricDistortion())

    ops.extend(
        [
            transforms.Normalize(mean=list(c.mean), std=list(c.std)),
            transforms.PadImage(size=c.train_scale, keys=_aligned_keys),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(keys=_aligned_keys),
        ]
    )
    return torchvision.transforms.Compose(ops)


def get_mlore_val_transforms(p: Any = None, cfg: Optional[MLoRETransformsConfig] = None):
    """Return the RFC/MLoRE *validation/test* transform pipeline (normalize + pad + tensor)."""
    import torchvision
    from cofai import transforms

    c = _resolve_cfg(p, cfg)
    _aligned_keys = MLORE_ALIGNED_KEYS
    return torchvision.transforms.Compose(
        [
            transforms.Normalize(mean=list(c.mean), std=list(c.std)),
            transforms.PadImage(size=c.test_scale, keys=_aligned_keys),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(keys=_aligned_keys),
        ]
    )


def get_mlore_transforms_cfg(
    p: Any = None, split: str = "train", cfg: Optional[MLoRETransformsConfig] = None
) -> Dict[str, Any]:
    """
    Return a *config dict* (YAML-friendly) describing the RFC/MLoRE transform pipeline.

    This enables composing fine-grained blocks (Normalize/PadImage/ToTensor/...) via config files.
    The returned dict follows this project's convention:

    - type: torchvision.transforms.Compose
    - transforms: [{type: <callable>, ...}, ...]

    Notes:
    - For val/test, this pipeline does **not** resize; it pads to `test_scale` and uses ignore labels.
    - For train, resize-like behavior comes from RandomScaling (data augmentation).
    """
    c = _resolve_cfg(p, cfg)
    split_l = str(split).lower()
    seg_keys = list(MLORE_ALIGNED_KEYS)

    # Prefer `cofai.transforms.*` namespace for YAML `type:` strings.
    T = "cofai.transforms"
    compose: Dict[str, Any] = {"type": "torchvision.transforms.Compose", "transforms": []}

    if split_l == "train":
        if c.enable_random_scaling:
            compose["transforms"].append(
                {
                    "type": f"{T}.RandomScaling",
                    "scale_factors": list(c.random_scaling_factors),
                    "discrete": bool(c.random_scaling_discrete),
                    "keys": seg_keys,
                }
            )
        if c.enable_random_crop:
            compose["transforms"].append(
                {
                    "type": f"{T}.RandomCrop",
                    "size": list(c.train_scale),
                    "cat_max_ratio": float(c.random_crop_cat_max_ratio),
                    "keys": seg_keys,
                }
            )
        if c.enable_hflip:
            compose["transforms"].append(
                {
                    "type": f"{T}.RandomHorizontalFlip",
                    "p": float(c.hflip_p),
                    "keys": seg_keys,
                }
            )
        if c.enable_photometric:
            compose["transforms"].append({"type": f"{T}.PhotoMetricDistortion"})

        compose["transforms"].extend(
            [
                {"type": f"{T}.Normalize", "mean": list(c.mean), "std": list(c.std)},
                {"type": f"{T}.PadImage", "size": list(c.train_scale), "keys": seg_keys},
                {"type": f"{T}.AddIgnoreRegions"},
                {"type": f"{T}.ToTensor", "keys": seg_keys},
            ]
        )
        return compose

    compose["transforms"].extend(
        [
            {"type": f"{T}.Normalize", "mean": list(c.mean), "std": list(c.std)},
            {"type": f"{T}.PadImage", "size": list(c.test_scale), "keys": seg_keys},
            {"type": f"{T}.AddIgnoreRegions"},
            {"type": f"{T}.ToTensor", "keys": seg_keys},
        ]
    )
    return compose


def get_mlore_transforms(p: Any = None, split: str = "train", cfg: Optional[MLoRETransformsConfig] = None):
    """
    Get data transforms for MLoRE datasets.

    Backward compatible API:
    - Old callers: `get_mlore_transforms(p, split)` keep working.
    - New callers: pass `cfg=MLoRETransformsConfig(...)` to configure behavior.

    Notes:
    - For `split != "train"` (val/test), this pipeline **does not resize**; it pads
      image + labels to `cfg.test_scale` and uses ignore regions for padded areas.
    """
    if str(split).lower() == "train":
        return get_mlore_train_transforms(p=p, cfg=cfg)
    return get_mlore_val_transforms(p=p, cfg=cfg)

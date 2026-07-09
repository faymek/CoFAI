"""Map backbone names to codec checkpoint subdirectories under weights/pqfc/."""

import os

WEIGHTS_SUBDIR = {
    "dinov2_vitl14": "dinov2_vitl14",
    "dinov2_vitg14": "dinov2_vitg14",
    "clip_vitl14": "clip_vitl14",
    "dinov3_vitl16": "dinov3_vitl16",
}


def codec_weights_dir(weights_root: str, backbone: str) -> str:
    subdir = WEIGHTS_SUBDIR.get(backbone, backbone)
    return os.path.join(weights_root, subdir)

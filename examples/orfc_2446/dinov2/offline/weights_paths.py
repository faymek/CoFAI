"""Map backbone names to codec checkpoint subdirectories under weights/orfc_2446/."""

import os

WEIGHTS_SUBDIR = {
    "dinov2_vitl14": "dinov2_vitl14_ori",
    "dinov2_vitg14": "dinov2_vitg14_ori",
}


def codec_weights_dir(weights_root: str, backbone: str) -> str:
    subdir = WEIGHTS_SUBDIR.get(backbone, backbone)
    return os.path.join(weights_root, subdir)

"""Build the checkpoint-compatible GPS feature-codec pipeline."""

from __future__ import annotations

from omegaconf import OmegaConf

from cofai.backbone import GPSTransReIDBackbone
from cofai.heads import GPSTransReIDHead
from cofai.models import CommonFeatureCodecModel
from examples.gps.reid.model import build_checkpoint_modules


def build_codec_model(
    cfg,
    *,
    camera_num: int,
    view_num: int,
):
    checkpoint_modules = build_checkpoint_modules(
        cfg,
        camera_num=camera_num,
        view_num=view_num,
    )
    checkpoint_modules.load_checkpoint(str(cfg.checkpoint))
    backbone = GPSTransReIDBackbone(
        checkpoint_modules.base,
        checkpoint_modules.b1,
        checkpoint_modules.b2,
    )
    reid_head = GPSTransReIDHead(
        checkpoint_modules.bottleneck,
        [
            checkpoint_modules.bottleneck_1,
            checkpoint_modules.bottleneck_2,
            checkpoint_modules.bottleneck_3,
            checkpoint_modules.bottleneck_4,
        ],
        neck_feature=cfg.model.neck_feature,
    )

    codec_model_cfg = OmegaConf.to_container(
        cfg.codec_model,
        resolve=True,
    )
    if codec_model_cfg.pop("type") != "cofai.models.CommonFeatureCodecModel":
        raise ValueError("GPS integration requires CommonFeatureCodecModel")
    return CommonFeatureCodecModel(
        backbone=backbone,
        heads={"reid": reid_head},
        **codec_model_cfg,
    )

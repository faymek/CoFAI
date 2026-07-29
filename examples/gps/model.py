"""Build the checkpoint-compatible GPS feature-codec pipeline."""

from __future__ import annotations

from omegaconf import OmegaConf

from cofai.models import CommonFeatureCodecModel
from examples.gps.reid.backbone import GPSReIDBackbone
from examples.gps.reid.head import GPSReIDHead
from examples.gps.reid.model import make_model


def build_codec_model(
    cfg,
    legacy_cfg,
    *,
    num_classes: int,
    camera_num: int,
    view_num: int,
):
    legacy_model = make_model(
        legacy_cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
    )
    legacy_model.load_param(str(cfg.checkpoint))
    backbone = GPSReIDBackbone(legacy_model)
    reid_head = GPSReIDHead(legacy_model)

    codec_model_cfg = OmegaConf.to_container(
        cfg.codec_model,
        resolve=True,
    )
    if codec_model_cfg.pop("type") != "CommonFeatureCodecModel":
        raise ValueError("GPS integration requires CommonFeatureCodecModel")
    return CommonFeatureCodecModel(
        backbone=backbone,
        heads={"reid": reid_head},
        **codec_model_cfg,
    )

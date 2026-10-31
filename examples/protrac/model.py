import torch
from omegaconf import OmegaConf

from cofai.heads import Dinov3SegmentationHead
from cofai.heads.vdt_reid import VDTReIDHead
from cofai.models import CommonFeatureCodecModel

from .reid.model import ProTRACReIDBackbone
from .semseg.model import ProTRACSegBackbone


def build_codec_model(cfg, device):
    if cfg.task == "semseg":
        head = (
            Dinov3SegmentationHead(
                in_channels=[1024],
                in_index=[0],
                input_transform="resize_concat",
                channels=1024,
                num_classes=150,
                patch_size=16,
                checkpoint=str(cfg.head),
            )
            .to(device)
            .eval()
        )
        backbone = ProTRACSegBackbone(cfg, head, device)
    else:
        head = VDTReIDHead().to(device).eval()
        backbone = ProTRACReIDBackbone(cfg, head, device).to(device)
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["model"]
        backbone_state = {
            key.removeprefix("backbone."): value for key, value in state.items() if key.startswith("backbone.")
        }
        backbone_state.update(
            {
                f"head.bottleneck.{key.removeprefix('heads.bottleneck.0.')}": value
                for key, value in state.items()
                if key.startswith("heads.bottleneck.0.")
            }
        )
        backbone.load_state_dict(backbone_state)
    model = CommonFeatureCodecModel(
        backbone=backbone,
        codec=OmegaConf.to_container(cfg.codec, resolve=True),
        heads={cfg.task: head},
        post_process=None,
    )
    return model.to(device).eval().requires_grad_(False)

import torch
import torch.nn as nn

from compressai.models.base import CompressionModel

from cofai.backbone import Dinov2TimmBackbone, Dinov3TimmBackbone
from cofai.latent_codecs import BypassLatentCodec
from cofai.engine.registry import instantiate_class, register


@register("DinoFeatureCodecModel")
class DinoFeatureCodecModel(CompressionModel):
    """Compress DINO backbone features for multiple downstream tasks.

    Extracts features with a DINO backbone (DINOv2 or DINOv3), compresses them with a
    latent codec, and reconstructs features for task heads. A single model can serve 
    several tasks at once -- classification, segmentation, depth estimation,
    and feature reconstruction -- depending on the heads provided.

    Args:
        dino_backbone (dict): Configuration for the DINO backbone. If a "type" key is
            present, uses instantiate_class for dynamic instantiation. Otherwise, uses
            Dinov3TimmBackbone with the provided config.
        dino_codec (dict): Configuration for the feature codec. If a "type" key is
            present, uses instantiate_class for dynamic instantiation. Otherwise, uses
            BypassLatentCodec (no compression) with the provided config.
        heads (dict | None): Optional mapping from task name (e.g. "cls", "semseg",
            "depth", "rec") to head configuration. A task that needs a head must have
            one registered here.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        dino: The DINO backbone model (Dinov3TimmBackbone or dynamically instantiated).
        dino_codec: The feature codec (BypassLatentCodec or dynamically instantiated).
        heads (nn.ModuleDict): Mapping from task name to head module.
        patch_size (int): Patch size used by the backbone model.
    """

    def __init__(
        self,
        dino_backbone={},
        dino_codec={},
        heads: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        if "type" in dino_backbone:
            self.dino = instantiate_class(dino_backbone)
        else:
            self.dino = Dinov3TimmBackbone(**dino_backbone)

        if "type" in dino_codec:
            self.dino_codec = instantiate_class(dino_codec)
        else:
            self.dino_codec = BypassLatentCodec(**dino_codec)

        self.patch_size = self.dino.patch_size

        self.heads = nn.ModuleDict()
        if heads:
            if not isinstance(heads, dict):
                raise TypeError("heads must be a dict mapping task -> head config")
            for task, hcfg in heads.items():
                if not isinstance(task, str) or hcfg is None:
                    continue
                if not isinstance(hcfg, dict) or "type" not in hcfg:
                    raise ValueError(f"heads.{task} must be a dict with a 'type' field")
                self.heads[task] = instantiate_class(hcfg)

    def _token_res(self, h, w):
        return (h // self.patch_size, w // self.patch_size)

    def _decode_tasks(self, h_hat, token_res, tasks):
        """Decode requested tasks from reconstructed tokens.

        ``seg`` returns raw backbone seg features (the seg head is applied by the
        caller). ``cls`` / ``semseg`` / ``depth`` / ``rae`` apply their configured
        head, which must be present in ``self.heads`` (``rae`` uses the ``rec`` head).
        """
        task_feats = {}
        if "cls" in tasks:
            feat = self.dino.decode_cls(h_hat)
            task_feats["cls"] = self.heads["cls"](feat)
        if "seg" in tasks:
            task_feats["seg"] = self.dino.decode_seg(h_hat, token_res)
        if "semseg" in tasks:
            feat = self.dino.decode_seg(h_hat, token_res)
            task_feats["semseg"] = self.heads["semseg"].predict(
                feat, scale=int(self.patch_size)
            )
        if "depth" in tasks:
            feat = self.dino.decode_depth(h_hat, token_res)
            size = (
                int(token_res[0]) * int(self.patch_size),
                int(token_res[1]) * int(self.patch_size),
            )
            task_feats["depth"] = self.heads["depth"].predict(feat, size=size)
        if "rae" in tasks:
            feat = self.dino.decode_rae(h_hat, token_res)
            task_feats["rae"] = self.heads["rec"].predict(
                feat,
                token_res=token_res,
                token_format="patch",
            )
        return task_feats

    def forward(self, x, qp=0, **kwargs):
        with torch.no_grad():
            h_dino = self.dino.encode(x)
            token_res = self._token_res(x.shape[2], x.shape[3])
            o_dino = self.dino.decode_whole(h_dino)[-1]

        dino_out = self.dino_codec(h_dino.clone(), token_res, qp=qp)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": dino_out["likelihoods"],
        }

    def offline_forward(self, data, device, qp=0, **kwargs):
        with torch.no_grad():
            h_dino = data["h_dino"].to(device).float()
            _, _, H, W = data["x_shape"]
            token_res = self._token_res(H, W)
            o_dino = self.dino.decode_whole(h_dino, token_res)[-1]

        dino_out = self.dino_codec(h_dino.clone(), token_res, qp=qp)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": dino_out["likelihoods"],
        }

    def extract_feature(self, x, **kwargs):
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            return {"h_dino": h_dino}

    def get_feature_numel(self, x):
        h_dino = self.dino.encode(x)
        return h_dino.numel()

    def forward_test(self, x, qp=0, tasks=[], **kwargs):
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = self._token_res(x.shape[2], x.shape[3])

            coded_unit = self.dino_codec(h_dino, token_res, qp=qp)
            h_dino_hat = coded_unit["h_hat"]

            task_feats = self._decode_tasks(h_dino_hat, token_res, tasks)
            return coded_unit, task_feats

    def compress(self, x, qp=0, **kwargs):
        h_dino = self.dino.encode(x)
        token_res = self._token_res(x.shape[2], x.shape[3])

        coded_unit = self.dino_codec.compress(h_dino, token_res, qp=qp)
        if "pstate" not in coded_unit:
            coded_unit["pstate"] = {}
        coded_unit["pstate"]["token_res"] = token_res
        return coded_unit

    def decompress(self, coded_unit, tasks=[], **kwargs):
        token_res = coded_unit["pstate"]["token_res"]
        decoded = self.dino_codec.decompress(**coded_unit)
        h_hat = decoded["h_hat"]
        return self._decode_tasks(h_hat, token_res, tasks)

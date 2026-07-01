import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models.base import CompressionModel

from cofai.backbone import Dinov2TimmBackbone, Dinov3TimmBackbone, Qwen3VLBackbone
from cofai.latent_codecs import BypassLatentCodec
from cofai.engine.registry import instantiate_class, register
from cofai.utils.complexity import latent_codec_complexity


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

    def codec_complexity(self, x, qp=0):
        """Latent-codec params (static) plus encode/decode FLOPs on ``x``'s tokens."""
        with torch.no_grad():
            h_dino = self.dino.encode(x)
            token_res = self._token_res(x.shape[2], x.shape[3])
        return latent_codec_complexity(self.dino_codec, h_dino, token_res, qp)

    def forward_test(self, x, qp=0, tasks=[], **kwargs):
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = self._token_res(x.shape[2], x.shape[3])

            t0 = time.time()
            coded_unit = self.dino_codec(h_dino, token_res, qp=qp)
            codec_t = time.time() - t0
            self._codec_time = {
                "codec_enc_time": codec_t / 2.0,
                "codec_dec_time": codec_t / 2.0,
            }
            h_dino_hat = coded_unit["h_hat"]

            task_feats = self._decode_tasks(h_dino_hat, token_res, tasks)
            return coded_unit, task_feats

    def compress(self, x, qp=0, **kwargs):
        h_dino = self.dino.encode(x)
        token_res = self._token_res(x.shape[2], x.shape[3])

        t0 = time.time()
        coded_unit = self.dino_codec.compress(h_dino, token_res, qp=qp)
        self._codec_time = {"codec_enc_time": time.time() - t0}
        if "pstate" not in coded_unit:
            coded_unit["pstate"] = {}
        coded_unit["pstate"]["token_res"] = token_res
        return coded_unit

    def decompress(self, coded_unit, tasks=[], **kwargs):
        token_res = coded_unit["pstate"]["token_res"]
        t0 = time.time()
        decoded = self.dino_codec.decompress(**coded_unit)
        codec_t = time.time() - t0
        self._codec_time = getattr(self, "_codec_time", {})
        self._codec_time["codec_dec_time"] = codec_t
        h_hat = decoded["h_hat"]
        return self._decode_tasks(h_hat, token_res, tasks)


@register("DinoSlideFeatureCodecModel")
class DinoSlideFeatureCodecModel(DinoFeatureCodecModel):
    """Sliding-window variant of :class:`DinoFeatureCodecModel` for large images.

    Like :class:`DinoFeatureCodecModel`, this model pairs a (timm-based) DINO
    backbone with a feature codec and optional task heads. The difference is that
    the input image is split into overlapping crops with a sliding window before
    being encoded, each crop's encoded feature is compressed independently, and
    the per-crop reconstructed features are returned for segmentation decoding.

    The sliding-window logic that previously lived on ``Dinov2OrgBackbone``
    (``slide_encode`` / ``slide_decode_seg``) is provided here as auxiliary
    methods of the model, so it works with any backbone exposing ``encode`` and
    ``decode_seg`` (e.g. ``Dinov2TimmBackbone`` / ``Dinov3TimmBackbone``).

    The model is built for the unified eval engine (``cofai.engine.run_eval``):
    each crop is encoded, compressed by ``dino_codec``, decoded to segmentation
    features, passed through the ``semseg`` head, upsampled to ``slide_size`` and
    fused into a full-resolution logits map. ``forward_test`` / ``decompress``
    therefore return ``task_feats["semseg"]`` as ``(B, num_classes, H, W)`` logits
    and report bitrate via a ``{"type": "slide_crops", "data": [...]}`` coded unit.

    Args:
        dino_backbone (dict): Backbone config, forwarded to
            :class:`DinoFeatureCodecModel`. Defaults to a timm backbone.
        dino_codec (dict): Feature codec config, forwarded to
            :class:`DinoFeatureCodecModel` (``BypassLatentCodec`` by default).
            Use :class:`~cofai.latent_codecs.VQFeatureCodec` for VQFC compression.
        heads (dict | None): Optional task heads, forwarded to the base class.
            A ``semseg`` head is required for the fused segmentation output.
        slide_size (list[int]): Crop size ``[h_crop, w_crop]``. Defaults to
            ``[518, 518]``.
        slide_stride (list[int]): Crop stride ``[h_stride, w_stride]``. Defaults
            to ``[259, 259]``.
        slide_codec_mode (str): How crops are handed to ``dino_codec``.

            - ``"per_crop"`` (default): each crop's tokens are compressed
              independently (one codec call per crop). Natural for per-token
              codecs such as :class:`~cofai.latent_codecs.VQFeatureCodec`.
            - ``"stacked"``: all crops are stacked into the batch dimension and
              compressed with a *single* codec call, then split back per crop.
              Required to reproduce the original sliding-window VTM rate, where
              every crop is packed into one image and encoded by one VTM call
              (see :class:`~cofai.latent_codecs.VtmLatentCodec`).

        **kwargs (dict): Additional keyword arguments forwarded to the base class.

    Attributes:
        slide_size (list[int]): Crop size used by the sliding window.
        slide_stride (list[int]): Crop stride used by the sliding window.
        slide_codec_mode (str): Crop-to-codec batching mode (see above).
    """

    def __init__(
        self,
        dino_backbone={},
        dino_codec={},
        heads: dict | None = None,
        slide_size=[518, 518],
        slide_stride=[259, 259],
        slide_codec_mode: str = "per_crop",
        **kwargs,
    ):
        super().__init__(
            dino_backbone=dino_backbone,
            dino_codec=dino_codec,
            heads=heads,
            **kwargs,
        )
        if slide_codec_mode not in ("per_crop", "stacked"):
            raise ValueError(
                "slide_codec_mode must be 'per_crop' or 'stacked', got "
                f"{slide_codec_mode!r}"
            )
        self.slide_size = list(slide_size)
        self.slide_stride = list(slide_stride)
        self.slide_codec_mode = slide_codec_mode

    def forward(self, x, qp=0, **kwargs):
        raise NotImplementedError(
            "DinoSlideFeatureCodecModel is inference-only; use forward_test / "
            "compress / decompress."
        )

    @property
    def _slide_res(self):
        """Token resolution (H, W) of a single crop."""
        return (
            self.slide_size[0] // self.patch_size,
            self.slide_size[1] // self.patch_size,
        )

    def _get_slide_crops(self, h_img, w_img):
        """Sliding-window crop coordinates ``(y1, x1, y2, x2)`` for an image."""
        h_crop, w_crop = self.slide_size
        h_stride, w_stride = self.slide_stride
        crops = []
        for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
            for w_idx in range(
                0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            ):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crops.append((y1, x1, y2, x2))
        return crops

    def slide_encode(self, img, slide_window=None, slide_stride=None):
        """Encode an image into per-crop backbone features with a sliding window.

        Crops the input into overlapping windows and runs ``self.dino.encode`` on
        each crop. Mirrors ``Dinov2OrgBackbone.slide_encode`` but lives on the
        model so any backbone with an ``encode`` method can be used.

        Args:
            img (torch.Tensor): Input image of shape (B, 3, H_img, W_img).
            slide_window (tuple | None): Window size ``(h_crop, w_crop)``. Falls
                back to ``self.slide_size`` when ``None``.
            slide_stride (tuple | None): Window stride ``(h_stride, w_stride)``.
                Falls back to ``self.slide_stride`` when ``None``.

        Returns:
            list[list[torch.Tensor]]: One ``[crop_feature]`` entry per crop, where
            ``crop_feature`` has shape (B, N, C).
        """
        if slide_window is not None or slide_stride is not None:
            sw = slide_window if slide_window is not None else self.slide_size
            ss = slide_stride if slide_stride is not None else self.slide_stride
            saved = (self.slide_size, self.slide_stride)
            self.slide_size, self.slide_stride = list(sw), list(ss)
            try:
                crops = self._get_slide_crops(img.shape[2], img.shape[3])
            finally:
                self.slide_size, self.slide_stride = saved
        else:
            crops = self._get_slide_crops(img.shape[2], img.shape[3])

        multi_crop_features = []
        for y1, x1, y2, x2 in crops:
            crop_features = self.dino.encode(img[:, :, y1:y2, x1:x2])
            multi_crop_features.append([crop_features])
        return multi_crop_features

    def slide_decode_seg(self, feature_list, slide_res=None):
        """Decode per-crop features into segmentation features.

        Args:
            feature_list (list[list[torch.Tensor]]): Per-crop encoded features as
                returned by :meth:`slide_encode` (each entry ``[crop_feature]``).
            slide_res (tuple | None): Token resolution ``(H, W)`` of a crop. Falls
                back to :attr:`_slide_res` when ``None``.

        Returns:
            list[list[torch.Tensor]]: Per-crop segmentation features, where each
            crop is the multi-layer list produced by ``self.dino.decode_seg``
            (shape (B, C, H, W) per layer).
        """
        slide_res = slide_res if slide_res is not None else self._slide_res
        return [self.dino.decode_seg(h[0], token_res=slide_res) for h in feature_list]

    def _fuse_seg_logits(self, seg_feats_per_crop, crops, img_hw):
        """Apply the ``semseg`` head per crop and fuse into full-resolution logits.

        Each crop's segmentation features are run through the head, upsampled to
        ``slide_size`` and averaged into the overlapping output canvas.

        Args:
            seg_feats_per_crop (list[list[torch.Tensor]]): Per-crop multi-layer
                segmentation features (from :meth:`slide_decode_seg`).
            crops (list[tuple[int, int, int, int]]): Crop coordinates.
            img_hw (tuple[int, int]): Output canvas size ``(H, W)``.

        Returns:
            torch.Tensor: Fused logits of shape ``(B, num_classes, H, W)``.
        """
        seg_head = self.heads["semseg"]
        num_classes = seg_head.num_classes
        h_img, w_img = img_hw
        b = seg_feats_per_crop[0][0].shape[0]
        device = seg_feats_per_crop[0][0].device

        preds = torch.zeros(b, num_classes, h_img, w_img, device=device)
        count = torch.zeros(b, 1, h_img, w_img, device=device)

        for (y1, x1, y2, x2), feats in zip(crops, seg_feats_per_crop):
            logits = seg_head.forward(feats)
            logits_up = F.interpolate(
                logits.float(),
                size=(self.slide_size[0], self.slide_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            logits_crop = logits_up[:, :, : y2 - y1, : x2 - x1]
            preds[:, :, y1:y2, x1:x2] += logits_crop
            count[:, :, y1:y2, x1:x2] += 1

        return preds / count.clamp_min(1.0)

    def _decode_slide_tasks(self, seg_feats_per_crop, crops, img_hw, tasks):
        """Build task outputs from per-crop reconstructed segmentation features.

        ``semseg`` returns fused full-resolution logits (head applied internally).
        ``seg`` returns the raw per-crop feature list (for an external seg head's
        ``slide_predict``). ``cls`` is not supported.
        """
        task_feats = {}
        if "cls" in tasks:
            raise NotImplementedError(
                "cls decoding is not supported in DinoSlideFeatureCodecModel"
            )
        if "seg" in tasks:
            task_feats["seg"] = seg_feats_per_crop
        if "semseg" in tasks:
            if "semseg" not in self.heads:
                raise KeyError(
                    "semseg task requires a 'semseg' head in DinoSlideFeatureCodecModel"
                )
            task_feats["semseg"] = self._fuse_seg_logits(
                seg_feats_per_crop, crops, img_hw
            )
        return task_feats

    @staticmethod
    def _crop_unit_for_bits(codec_out):
        """Extract just the bit-bearing fields from a codec output dict."""
        return {
            k: codec_out[k]
            for k in ("strings", "likelihoods", "bits")
            if k in codec_out
        }

    def get_feature_numel(self, x):
        crops = self._get_slide_crops(x.shape[2], x.shape[3])
        hw = self._slide_res[0] * self._slide_res[1]
        embed_dim = int(self.dino.model.embed_dim)
        return len(crops) * hw * embed_dim

    def codec_complexity(self, x, qp=0):
        """Latent-codec params and encode/decode FLOPs over all slide crops."""
        slide_res = self._slide_res
        crops = self._get_slide_crops(x.shape[2], x.shape[3])
        with torch.no_grad():
            if self.slide_codec_mode == "stacked":
                h = self._encode_crops_stacked(x, crops)
                return latent_codec_complexity(self.dino_codec, h, slide_res, qp)
            y1, x1, y2, x2 = crops[0]
            out = latent_codec_complexity(
                self.dino_codec,
                self.dino.encode(x[:, :, y1:y2, x1:x2]),
                slide_res,
                qp,
            )
            for y1, x1, y2, x2 in crops[1:]:
                comp = latent_codec_complexity(
                    self.dino_codec,
                    self.dino.encode(x[:, :, y1:y2, x1:x2]),
                    slide_res,
                    qp,
                )
                for key in ("codec_enc_flops", "codec_dec_flops"):
                    if key in comp:
                        out[key] = out.get(key, 0) + comp[key]
        return out

    def _encode_crops_stacked(self, x, crops):
        """Encode every crop and stack them along the batch dim -> ``(N_crop, N, C)``.

        Used by the ``"stacked"`` codec mode so the codec compresses all crops in
        a single call (e.g. one VTM bitstream for the whole sliding window).
        """
        feats = [self.dino.encode(x[:, :, y1:y2, x1:x2]) for (y1, x1, y2, x2) in crops]
        return torch.cat(feats, dim=0)

    def _seg_feats_from_stacked_hat(self, h_hat, slide_res, n_crops):
        """Split stacked reconstructed tokens back into per-crop seg features."""
        return [
            self.dino.decode_seg(h_hat[i : i + 1], token_res=slide_res)
            for i in range(n_crops)
        ]

    def forward_test(self, x, qp=0, tasks=[], **kwargs):
        with torch.inference_mode():
            slide_res = self._slide_res
            crops = self._get_slide_crops(x.shape[2], x.shape[3])

            codec_t = 0.0
            if self.slide_codec_mode == "stacked":
                h_stack = self._encode_crops_stacked(x, crops)
                t0 = time.time()
                out = self.dino_codec(h_stack, slide_res, qp=qp)
                codec_t += time.time() - t0
                seg_feats_per_crop = self._seg_feats_from_stacked_hat(
                    out["h_hat"], slide_res, len(crops)
                )
                crop_units = [self._crop_unit_for_bits(out)]
            else:
                crop_units = []
                seg_feats_per_crop = []
                for y1, x1, y2, x2 in crops:
                    h = self.dino.encode(x[:, :, y1:y2, x1:x2])
                    t0 = time.time()
                    out = self.dino_codec(h, slide_res, qp=qp)
                    codec_t += time.time() - t0
                    crop_units.append(self._crop_unit_for_bits(out))
                    seg_feats_per_crop.append(
                        self.dino.decode_seg(out["h_hat"], token_res=slide_res)
                    )

            self._codec_time = {
                "codec_enc_time": codec_t / 2.0,
                "codec_dec_time": codec_t / 2.0,
            }
            task_feats = self._decode_slide_tasks(
                seg_feats_per_crop, crops, (x.shape[2], x.shape[3]), tasks
            )
            coded_data = {"type": "slide_crops", "data": crop_units}
            return coded_data, task_feats

    def compress(self, x, qp=0, **kwargs):
        with torch.inference_mode():
            slide_res = self._slide_res
            crops = self._get_slide_crops(x.shape[2], x.shape[3])

            codec_t = 0.0
            if self.slide_codec_mode == "stacked":
                h_stack = self._encode_crops_stacked(x, crops)
                t0 = time.time()
                crop_units = [self.dino_codec.compress(h_stack, slide_res, qp=qp)]
                codec_t += time.time() - t0
            else:
                crop_units = []
                for y1, x1, y2, x2 in crops:
                    h = self.dino.encode(x[:, :, y1:y2, x1:x2])
                    t0 = time.time()
                    crop_units.append(self.dino_codec.compress(h, slide_res, qp=qp))
                    codec_t += time.time() - t0

            self._codec_time = {"codec_enc_time": codec_t}
            return {
                "type": "slide_crops",
                "data": crop_units,
                "pstate": {
                    "crops": crops,
                    "img_hw": (int(x.shape[2]), int(x.shape[3])),
                    "slide_res": slide_res,
                    "mode": self.slide_codec_mode,
                },
            }

    def decompress(self, coded_unit, tasks=[], **kwargs):
        with torch.inference_mode():
            crop_units = coded_unit["data"]
            pstate = coded_unit["pstate"]
            crops = [tuple(c) for c in pstate["crops"]]
            img_hw = tuple(pstate["img_hw"])
            slide_res = tuple(pstate["slide_res"])
            mode = pstate.get("mode", self.slide_codec_mode)

            codec_t = 0.0
            if mode == "stacked":
                t0 = time.time()
                decoded = self.dino_codec.decompress(**crop_units[0])
                codec_t += time.time() - t0
                seg_feats_per_crop = self._seg_feats_from_stacked_hat(
                    decoded["h_hat"], slide_res, len(crops)
                )
            else:
                seg_feats_per_crop = []
                for enc in crop_units:
                    t0 = time.time()
                    decoded = self.dino_codec.decompress(**enc)
                    codec_t += time.time() - t0
                    seg_feats_per_crop.append(
                        self.dino.decode_seg(decoded["h_hat"], token_res=slide_res)
                    )

            self._codec_time = getattr(self, "_codec_time", {})
            self._codec_time["codec_dec_time"] = codec_t
            return self._decode_slide_tasks(seg_feats_per_crop, crops, img_hw, tasks)


@register("Qwen3vlFeatureCodecModel")
class Qwen3vlFeatureCodecModel(CompressionModel):
    """Compress Qwen3VL visual features for VQA text generation.

    Extracts breakpoint tokens with :class:`~cofai.backbone.Qwen3VLBackbone`
    in ``(B, L, C)`` layout (same contract as :class:`DinoFeatureCodecModel`),
    compresses with a latent codec, and continues generation through the LM.
    """

    def __init__(
        self,
        qwen_backbone=None,
        qwen_codec=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        qwen_backbone = dict(
            qwen_backbone or {"type": "cofai.backbone.Qwen3VLBackbone"}
        )
        qwen_codec = dict(
            qwen_codec or {"type": "cofai.latent_codecs.BypassLatentCodec"}
        )
        if device is not None:
            qwen_backbone.setdefault("device", str(device))

        self.qwen = instantiate_class(qwen_backbone)
        self.qwen_codec = instantiate_class(qwen_codec)

    @staticmethod
    def _vqa_prompt(task_data):
        vqa = (task_data or {}).get("vqa")
        if not isinstance(vqa, dict):
            raise KeyError("Qwen VQA requires task_data['vqa'].")
        prompt = vqa.get("prompt")
        if not isinstance(prompt, str):
            raise TypeError("task_data['vqa']['prompt'] must be a string.")
        return prompt

    def forward_test(self, image, qp=0, tasks=None, task_data=None, **kwargs):
        with torch.inference_mode():
            tasks = tasks or []
            h_tokens, token_res = self.qwen.encode_image(image)
            self._feature_numel = h_tokens.numel()

            t0 = time.time()
            coded_unit = self.qwen_codec(h_tokens, token_res, qp=qp)
            codec_t = time.time() - t0
            self._codec_time = {
                "codec_enc_time": codec_t / 2.0,
                "codec_dec_time": codec_t / 2.0,
            }
            task_feats = {}
            if "vqa" in tasks:
                task_feats["vqa"] = self.qwen.decode_text(
                    coded_unit["h_hat"],
                    prompt=self._vqa_prompt(task_data),
                    token_res=token_res,
                )
            if "pstate" not in coded_unit:
                coded_unit["pstate"] = {}
            coded_unit["pstate"]["token_res"] = token_res
            return coded_unit, task_feats

    def compress(self, image, qp=0, **kwargs):
        h_tokens, token_res = self.qwen.encode_image(image)
        self._feature_numel = h_tokens.numel()

        t0 = time.time()
        coded_unit = self.qwen_codec.compress(h_tokens, token_res, qp=qp)
        self._codec_time = {"codec_enc_time": time.time() - t0}
        if "pstate" not in coded_unit:
            coded_unit["pstate"] = {}
        coded_unit["pstate"]["token_res"] = token_res
        return coded_unit

    def decompress(self, coded_unit, tasks=None, task_data=None, **kwargs):
        tasks = tasks or []
        token_res = tuple(coded_unit["pstate"]["token_res"])
        t0 = time.time()
        decoded = self.qwen_codec.decompress(**coded_unit)
        codec_t = time.time() - t0
        self._codec_time = getattr(self, "_codec_time", {})
        self._codec_time["codec_dec_time"] = codec_t
        task_feats = {}
        if "vqa" in tasks:
            task_feats["vqa"] = self.qwen.decode_text(
                decoded["h_hat"],
                prompt=self._vqa_prompt(task_data),
                token_res=token_res,
            )
        return task_feats

    def get_feature_numel(self, _image=None):
        return self._feature_numel

    def codec_complexity(self, image, qp=0):
        with torch.no_grad():
            h_tokens, token_res = self.qwen.encode_image(image)
        return latent_codec_complexity(self.qwen_codec, h_tokens, token_res, qp)

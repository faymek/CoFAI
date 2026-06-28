import torch
import torch.nn as nn

from compressai.models.base import CompressionModel
from compressai.models.utils import conv

from cofai.backbone import *
from cofai.token_codecs.base import UniformTokenCodec
from cofai.latent_codecs.vit_feature_codec import (
    VitUnionLatentCodec,
    VitUnionLatentCodecWithCtx,
    VitUnionLatentCodecCtxAsHyper,
    VbrVitUnionLatentCodec,
)
from cofai.engine.registry import instantiate_class, register


@register("MPC_I1")
class MPC_I1(CompressionModel):
    """
    Multi-Purpose Compression model using VQGAN backbone only.

    This is a single-layer compression model that uses VQGAN for feature extraction
    and uniform token codec for compression. It provides basic image reconstruction
    capabilities.

    Args:
        vqgan_config (dict): Configuration dictionary for the VQGAN backbone.
            Passed directly to VqganBackbone constructor.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        vqgan (VqganBackbone): The VQGAN backbone model.
        vqgan_codec (UniformTokenCodec): The uniform token codec for compression.
        patch_size (int): Patch size used by the model (fixed at 16).
    """

    def __init__(self, tokenizer, token_codec, patch_size=16, **kwargs):
        super().__init__()
        self.tokenizer = instantiate_class(tokenizer)
        self.token_codec = instantiate_class(token_codec)
        self.patch_size = patch_size

    def forward(self, x, **kwargs):
        """
        Forward pass for training.

        Encodes input image using VQGAN, compresses tokens, and decodes to
        reconstructed image.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            out (dict): Dictionary containing:

                - "likelihoods": Likelihoods from the codec
                - "x_hat": Reconstructed image tensor (legacy training key)
                - "rec": Same tensor as ``x_hat``
        """
        tk_enc = self.tokenizer.encode(x)
        tk_out = self.token_codec(tk_enc["tokens"])
        x_hat = self.tokenizer.decode(tk_enc["z_q"])
        # ``rec`` aligns eval protocol (``x_hat`` kept for existing training code).
        return {"likelihoods": tk_out["likelihoods"], "x_hat": x_hat, "rec": x_hat}

    def compress(self, x, **kwargs):
        """
        Compress input image to byte strings.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information
        """
        tk_enc = self.tokenizer.encode(x)
        coded_unit = self.token_codec.compress(tk_enc["tokens"])
        return coded_unit

    def decompress(self, coded_unit, **kwargs):
        """
        Decompress byte strings to reconstructed image and features.

        Args:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            task_feats (dict): Dictionary of task-specific features:

                - "z_q": Quantized features
                - "tokens": VQGAN tokens
                - "rec": Reconstructed image tensor
        """
        out = self.token_codec.decompress(**coded_unit)
        tokens = out["tokens"]
        z_q = self.tokenizer.tokens_to_features(tokens)
        rec = self.tokenizer.decode(z_q)
        task_feats = {"z_q": z_q, "tokens": tokens, "rec": rec}
        return task_feats


@register("MPC_I2")
class MPC_I2(CompressionModel):
    """
    Multi-Purpose Compression model using DINOv2 backbone only.

    This is a single-layer compression model that uses DINOv2 for feature extraction
    and ViT-based latent codec for compression. It supports multiple downstream tasks
    including classification and segmentation.

    Args:
        dino_backbone (dict): Configuration dictionary for the DINOv2 backbone.
            If "type" key is present, uses instantiate_class for dynamic instantiation.
            Otherwise, uses Dinov2TimmBackbone with provided config.
        dino_codec (dict): Configuration dictionary for the DINO codec.
            If "type" key is present in dino_backbone, uses instantiate_class.
            Otherwise, uses VitUnionLatentCodec with provided config.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        dino: The DINOv2 backbone model (Dinov2TimmBackbone or dynamically instantiated).
        dino_codec: The DINO codec (VitUnionLatentCodec or dynamically instantiated).
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
            self.dino_codec = instantiate_class(dino_codec)
        else:
            self.dino = Dinov2TimmBackbone(**dino_backbone)
            self.dino_codec = VitUnionLatentCodec(**dino_codec)
        self.patch_size = self.dino.patch_size

        self.heads = nn.ModuleDict()
        if heads:
            if not isinstance(heads, dict):
                raise TypeError("heads must be a dict mapping task -> head config")
            for task, hcfg in heads.items():
                if not isinstance(task, str):
                    continue
                if hcfg is None:
                    continue
                if not isinstance(hcfg, dict) or "type" not in hcfg:
                    raise ValueError(f"heads.{task} must be a dict with a 'type' field")
                self.heads[task] = instantiate_class(hcfg)

    def forward(self, x, qp=0, **kwargs):
        """
        Forward pass for training with learned image compression (LIC).

        Encodes input image using DINOv2, compresses features with codec,
        and returns reconstructed features and likelihoods for training.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Quantization parameter. Defaults to 0.
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            out (dict): Dictionary containing:

                - "h_dino_hat": Reconstructed DINO features
                - "h_dino": Original DINO features
                - "likelihoods": Likelihoods from the codec
        """
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            o_dino = self.dino.decode_whole(h_dino)[-1]

        h_dino = h_dino.clone()
        dino_out = self.dino_codec(h_dino, token_res, qp=qp)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": dino_out["likelihoods"],
        }

    def extract_feature(self, x, **kwargs):
        """
        Extract features from input image for offline training.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            out (dict): Dictionary containing:

                - "h_dino": Extracted DINO features
        """
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            return {
                "h_dino": h_dino,
            }

    def offline_forward(self, data, device, qp=0, **kwargs):
        """
        Offline forward pass for training with pre-extracted features.

        Processes pre-extracted DINO features for learned image compression training.
        This method is used when features are extracted separately to save memory.

        Args:
            data (dict): Dictionary containing:

                - "h_dino": Pre-extracted DINO features
                - "x_shape": Original image shape (B, C, H, W)
            device (torch.device): Device to move tensors to.
            qp (int): Quantization parameter. Defaults to 0.
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            out (dict): Dictionary containing:

                - "h_dino_hat": Reconstructed DINO features
                - "h_dino": Original DINO features
                - "likelihoods": Likelihoods from the codec
        """
        with torch.inference_mode():
            h_dino = data["h_dino"].to(device).float()
            _, _, H, W = data["x_shape"]
            token_res = (H // self.patch_size, W // self.patch_size)

            # x_uint8 = data["x_uint8"].to(device)
            # x = x_uint8 / 255.0
            # h_dino_ref = self.dino.encode(x).float()
            # print(torch.mean(torch.abs(h_dino - h_dino_ref)))
            # Note: The following commented code can be used for consistency checking:
            # x_uint8 = data["x_uint8"].to(device)
            # x = x_uint8 / 255.0
            # h_dino_ref = self.dino.encode(x).float()
            # assert torch.allclose(h_dino, h_dino_ref), "not consistent"

            o_dino = self.dino.decode_whole(h_dino, token_res)[-1]

        h_dino = h_dino.clone()
        dino_out = self.dino_codec(h_dino, token_res, qp=qp)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": dino_out["likelihoods"],
        }

    def forward_test(self, x, qp=0, tasks=[], **kwargs):
        """
        Forward pass for testing/inference with compression.

        Encodes input image, compresses features, and generates task-specific outputs.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Quantization parameter. Defaults to 0.
            tasks (list of str): List of tasks to perform. Supported tasks:

                - "cls": Classification task
                - "semseg": Semantic segmentation task
                - "rae": RAE task (if ``heads.rec``/``heads.rae`` is set, decoder image; else patch tokens)
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information
                - "h_hat": Reconstructed features

            task_feats (dict): Per-task outputs (only keys with a configured head or raw ``rae`` tokens)

                - "cls": logits if ``heads.cls`` is set
                - "semseg": logits if ``heads.semseg`` is set
                - "rae": see task description above
        """
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            coded_unit = self.dino_codec(h_dino, token_res, qp=qp)
            h_dino_hat = coded_unit["h_hat"]

            task_feats = {}
            if "cls" in tasks:
                feat_cls = self.dino.decode_cls(h_dino_hat)
                if "cls" in self.heads:
                    task_feats["cls"] = self.heads["cls"](feat_cls)
            if "semseg" in tasks:
                feat_semseg = self.dino.decode_seg(h_dino_hat, token_res)
                if "semseg" in self.heads:
                    task_feats["semseg"] = self.heads["semseg"].predict(
                        feat_semseg, scale=int(self.patch_size)
                    )
            if "rae" in tasks:
                feat_rae = self.dino.decode_rae(h_dino_hat, token_res)
                head_key = "rec" if "rec" in self.heads else ("rae" if "rae" in self.heads else None)
                if head_key is not None:
                    task_feats["rae"] = self.heads[head_key].predict(
                        feat_rae,
                        token_res=token_res,
                        token_format="patch",
                    )
                else:
                    task_feats["rae"] = feat_rae

            return coded_unit, task_feats

    def get_feature_numel(self, x):
        """
        Calculate the total number of elements in the extracted features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            numel (int): Total number of elements in the feature tensor.
        """
        h_dino = self.dino.encode(x)
        return h_dino.numel()

    def compress(self, x, qp=0, **kwargs):
        """
        Compress input image to byte strings.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Quantization parameter. Defaults to 0.
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information
        """
        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        coded_unit = self.dino_codec.compress(h_dino, token_res, qp=qp)
        return coded_unit

    def decompress(self, coded_unit, tasks=[], **kwargs):
        """
        Decompress byte strings to task-specific features.

        Args:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information

            tasks (list of str): List of tasks to perform. Supported tasks:

                - "cls": Classification task
                - "semseg": Semantic segmentation task
                - "rae": RAE task (see ``forward_test`` for head vs patch-token output)

            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            task_feats (dict): Same rules as :meth:`forward_test` (heads applied inside each task branch).
        """
        encoded = coded_unit
        token_res = coded_unit["pstate"]["token_res"]
        decoded = self.dino_codec.decompress(**encoded)
        task_feats = {}
        if "cls" in tasks:
            feat_cls = self.dino.decode_cls(decoded["h_hat"])
            if "cls" in self.heads:
                task_feats["cls"] = self.heads["cls"](feat_cls)
        if "semseg" in tasks:
            feat_semseg = self.dino.decode_seg(decoded["h_hat"], token_res)
            if "semseg" in self.heads:
                task_feats["semseg"] = self.heads["semseg"].predict(
                    feat_semseg, scale=int(self.patch_size)
                )
        if "rae" in tasks:
            feat_rae = self.dino.decode_rae(decoded["h_hat"], token_res)
            head_key = "rec" if "rec" in self.heads else ("rae" if "rae" in self.heads else None)
            if head_key is not None:
                task_feats["rae"] = self.heads[head_key].predict(
                    feat_rae,
                    token_res=token_res,
                    token_format="patch",
                )
            else:
                task_feats["rae"] = feat_rae
        return task_feats


@register("MPC_I12")
class MPC_I12(CompressionModel):
    """
    Multi-Purpose Compression model with two layers: VQGAN and DINOv2.

    This is a two-layer compression model that combines VQGAN (layer 1) and DINOv2
    (layer 2) for hierarchical compression. The DINOv2 codec uses VQGAN context
    for conditional compression. Supports multiple tasks including reconstruction,
    classification, and segmentation.

    Args:
        vqgan_backbone (dict): Configuration dictionary for the VQGAN backbone.
            Passed directly to VqganBackbone constructor.
        vqgan_codec (dict): Configuration dictionary for the VQGAN codec.
            Passed directly to UniformTokenCodec constructor.
        dino_backbone (dict): Configuration dictionary for the DINOv2 backbone.
            Passed directly to Dinov2TimmBackbone constructor.
        dino_codec (dict): Configuration dictionary for the DINO codec.
            Must contain "h_dim" and "ctx_dim" keys for the conditional decoder.
            Passed directly to VitUnionLatentCodecWithCtx constructor.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        vqgan (VqganBackbone): The VQGAN backbone model.
        vqgan_codec (UniformTokenCodec): The VQGAN codec.
        dino (Dinov2TimmBackbone): The DINOv2 backbone model.
        dino_codec (VitUnionLatentCodecWithCtx): The DINO codec with context.
        patch_size (int): Patch size used by the DINOv2 backbone.
        cond_dec_for_vqgan (nn.Sequential): Conditional decoder for enhancing
            VQGAN reconstruction using DINOv2 features.
    """

    def __init__(
        self,
        vqgan_backbone={},
        vqgan_codec={},
        dino_backbone={},
        dino_codec={},
        heads: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.vqgan = VqganBackbone(vqgan_backbone)
        self.vqgan_codec = UniformTokenCodec(**vqgan_codec)
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.dino_codec = VitUnionLatentCodecWithCtx(**dino_codec)
        self.patch_size = self.dino.patch_size

        self.heads = nn.ModuleDict()
        if heads:
            if not isinstance(heads, dict):
                raise TypeError("heads must be a dict mapping task -> head config")
            for task, hcfg in heads.items():
                if not isinstance(task, str):
                    continue
                if hcfg is None:
                    continue
                if not isinstance(hcfg, dict) or "type" not in hcfg:
                    raise ValueError(f"heads.{task} must be a dict with a 'type' field")
                self.heads[task] = instantiate_class(hcfg)

        # Additional branch for enhancing VQGAN reconstruction using DINOv2 features
        D_DINO = dino_codec["h_dim"]
        D_VQGAN = dino_codec["ctx_dim"]
        self.cond_dec_for_vqgan = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
        )

    def _apply_heads(self, task_feats: dict, *, tasks: list[str], token_res) -> dict:
        out: dict = {}
        if "cls" in tasks and "cls" in task_feats and "cls" in self.heads:
            out["cls"] = self.heads["cls"](task_feats["cls"])

        if "semseg" in tasks and "semseg" in task_feats and "semseg" in self.heads:
            out["semseg"] = self.heads["semseg"].predict(
                task_feats["semseg"], scale=int(self.patch_size)
            )

        if "rec" in tasks and "rec" in task_feats:
            out["rec"] = task_feats["rec"]
        return out

    def forward(self, x, qp=0, **kwargs):
        """
        Forward pass for training.

        Processes input through both VQGAN and DINOv2 layers, performs conditional
        compression, and returns features and likelihoods for training.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            out (dict): Dictionary containing:

                - "h_vqgan": Original VQGAN features
                - "h_vqgan_hat": Enhanced VQGAN features
                - "h_dino": Original DINO features
                - "h_dino_hat": Reconstructed DINO features
                - "likelihoods": Likelihoods from the DINO codec
                - "x_hat": Reconstructed image (only in eval mode, None during training)
        """
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            # Note: vqgan_codec is not used here as it only provides constant likelihoods
            h_vqgan = vqgan_enc["z"]
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            o_dino = self.dino.decode_whole(h_dino)[-1]

        h_dino = h_dino.clone()
        h_vqgan_ctx = h_vqgan_ctx.clone()
        dino_out = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        h_hat_for_vqgan = self.cond_dec_for_vqgan(
            torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
        )

        if not self.training:
            with torch.no_grad():
                x_hat = self.vqgan.decode(h_hat_for_vqgan)
        else:
            x_hat = None

        return {
            "h_vqgan": h_vqgan.clone(),
            "h_vqgan_hat": h_hat_for_vqgan,
            "h_dino": o_dino.clone(),
            "h_dino_hat": o_dino_hat,
            "likelihoods": dino_out["likelihoods"],
            "x_hat": x_hat,
        }

    def extract_feature(self, x, **kwargs):
        """
        Extract features from input image for offline training.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            out (dict): Dictionary containing:

                - "tokens": VQGAN tokens
                - "h_dino": Extracted DINO features
        """
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            h_dino = self.dino.encode(x)
            return {
                "tokens": vqgan_enc["tokens"],
                "h_dino": h_dino,
            }

    def offline_forward(self, data, device, **kwargs):
        """
        Offline forward pass for training with pre-extracted features.

        Processes pre-extracted VQGAN tokens and DINO features for training.
        This method is used when features are extracted separately to save memory.

        Args:
            data (dict): Dictionary containing:

                - "h_dino": Pre-extracted DINO features
                - "tokens": Pre-extracted VQGAN tokens
                - "x_shape": Original image shape (B, C, H, W)
            device (torch.device): Device to move tensors to.
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            out (dict): Dictionary containing:

                - "h_vqgan": VQGAN context features
                - "h_vqgan_hat": Enhanced VQGAN features
                - "h_dino": Original DINO features
                - "h_dino_hat": Reconstructed DINO features
                - "likelihoods": Likelihoods from the DINO codec
                - "x_hat": Reconstructed image (only in eval mode, None during training)
        """
        with torch.inference_mode():
            h_dino = data["h_dino"].to(device).float()
            tokens = data["tokens"].to(device).long()
            _, _, H, W = data["x_shape"]
            token_res = (H // self.patch_size, W // self.patch_size)

            # Note: The following commented code can be used for consistency checking:
            # x_uint8 = data["x_uint8"].to(device)
            # x = x_uint8 / 255.0
            # h_dino_ref = self.dino.encode(x).to(torch.float16).float()
            # assert torch.allclose(h_dino, h_dino_ref)

            o_dino = self.dino.decode_whole(h_dino, token_res)[-1]

        h_dino = h_dino.clone()
        h_vqgan_ctx = self.vqgan.tokens_to_features(tokens.clone())

        dino_out = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        h_hat_for_vqgan = self.cond_dec_for_vqgan(
            torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
        )

        if not self.training:
            with torch.no_grad():
                x_hat = self.vqgan.decode(h_hat_for_vqgan)
        else:
            x_hat = None

        return {
            "h_vqgan": h_vqgan_ctx.clone(),
            "h_vqgan_hat": h_hat_for_vqgan,
            "h_dino": o_dino.clone(),
            "h_dino_hat": o_dino_hat,
            "likelihoods": dino_out["likelihoods"],
            "x_hat": x_hat,
        }

    def forward_test(self, x, tasks, **kwargs):
        """
        Forward pass for testing/inference with compression.

        Processes input through both layers, compresses features, and generates
        task-specific outputs.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            tasks (list of str): List of tasks to perform. Supported tasks:

                - "rec": Reconstruction; branch chosen by ``recon_branch`` in kwargs (1=VQGAN-only, 2=enhanced).
                - "cls": Classification task
                - "semseg": Semantic segmentation task

            **kwargs (dict): Optional ``recon_branch`` (int, default 2): for ``rec`` only, 1=basic VQGAN, 2=enhanced.

        Returns:
            coded_data (dict): Dictionary containing compressed data:

                - "type": "frame"
                - "data": Dictionary with:
                    - "layer1": VQGAN compressed data
                    - "layer2": DINO compressed data

            task_feats (dict): Dictionary of task-specific features:

                - "rec": Reconstruction (if "rec" in tasks)
                - "cls": Classification features (if "cls" in tasks)
                - "semseg": Segmentation features (if "semseg" in tasks)
        """
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            # Note: vqgan_codec only provides constant likelihoods
            vqgan_cu = self.vqgan_codec(vqgan_enc["tokens"])
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            dino_cu = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
            h_dino_hat = dino_cu["h_hat"]

            task_feats = {}
            recon_branch = int(kwargs.get("recon_branch", 2))

            # Kind-driven reconstruction: if task_specs declares any reconstruction tasks,
            # emit outputs keyed by their labels (label naming is free at framework level).
            task_specs = kwargs.get("task_specs")
            recon_labels = []
            if isinstance(task_specs, list):
                for sp in task_specs:
                    if isinstance(sp, dict) and sp.get("kind") == "rec" and sp.get("label"):
                        recon_labels.append(str(sp["label"]))

            if ("rec" in tasks) or recon_labels:
                if recon_branch == 1:
                    rec_img = self.tokenizer.decode(h_vqgan_ctx)
                else:
                    h_hat_for_vqgan = self.cond_dec_for_vqgan(
                        torch.cat([dino_cu["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
                    )
                    rec_img = self.tokenizer.decode(h_hat_for_vqgan)
                if "rec" in tasks:
                    task_feats["rec"] = rec_img
                for lb in recon_labels:
                    task_feats[lb] = rec_img
            if "cls" in tasks:
                task_feats["cls"] = self.dino.decode_cls(h_dino_hat)
            if "semseg" in tasks:
                task_feats["semseg"] = self.dino.decode_seg(h_dino_hat, token_res)
            task_feats = self._apply_heads(task_feats, tasks=list(tasks), token_res=token_res)

            coded_data = {
                "type": "frame",
                "data": {
                    "layer1": vqgan_cu,
                    "layer2": dino_cu,
                },
            }
            return coded_data, task_feats

    def compress(self, x, **kwargs):
        vqgan_enc = self.vqgan.encode(x)
        vqgan_cu = self.vqgan_codec.compress(vqgan_enc["tokens"])
        h_vqgan_ctx = vqgan_enc["z_q"]

        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        dino_cu = self.dino_codec.compress(h_dino, h_vqgan_ctx, token_res)

        coded_data = {
            "type": "frame",
            "data": {
                "layer1": vqgan_cu,
                "layer2": dino_cu,
            },
        }
        return coded_data

    def decompress(self, coded_data, tasks=[], **kwargs):
        vqgan_cu = coded_data["data"]["layer1"]
        dino_cu = coded_data["data"]["layer2"]

        token_res = dino_cu["pstate"]["token_res"]
        vqgan_decoded = self.vqgan_codec.decompress(**vqgan_cu)
        h_vqgan_ctx = self.vqgan.tokens_to_features(vqgan_decoded["tokens"])
        dino_decoded = self.dino_codec.decompress(**dino_cu, ctx=h_vqgan_ctx)

        task_feats = {}
        recon_branch = int(kwargs.get("recon_branch", 2))

        task_specs = kwargs.get("task_specs")
        recon_labels = []
        if isinstance(task_specs, list):
            for sp in task_specs:
                if isinstance(sp, dict) and sp.get("kind") == "rec" and sp.get("label"):
                    recon_labels.append(str(sp["label"]))

        if ("rec" in tasks) or recon_labels:
            if recon_branch == 1:
                rec_img = self.tokenizer.decode(h_vqgan_ctx)
            else:
                h_hat_for_vqgan = self.cond_dec_for_vqgan(
                    torch.cat([dino_decoded["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
                )
                rec_img = self.tokenizer.decode(h_hat_for_vqgan)
            if "rec" in tasks:
                task_feats["rec"] = rec_img
            for lb in recon_labels:
                task_feats[lb] = rec_img
        if "cls" in tasks:
            task_feats["cls"] = self.dino.decode_cls(dino_decoded["h_hat"])
        if "semseg" in tasks:
            task_feats["semseg"] = self.dino.decode_seg(dino_decoded["h_hat"], token_res)

        return self._apply_heads(task_feats, tasks=list(tasks), token_res=token_res)


@register("MPC_I12_CtxAsHyper")
class MPC_I12_CtxAsHyper(CompressionModel):
    """
    Multi-Purpose Compression model with context as hyperprior.

    This is a variant of MPC_I12 where the VQGAN context is treated as hyperprior
    for the DINOv2 codec. Similar to MPC_I12, it combines VQGAN (layer 1) and DINOv2
    (layer 2) for hierarchical compression, but uses a different codec architecture
    that treats context as hyperprior.

    Args:
        vqgan_backbone (dict): Configuration dictionary for the VQGAN backbone.
            Passed directly to VqganBackbone constructor.
        vqgan_codec (dict): Configuration dictionary for the VQGAN codec.
            Passed directly to UniformTokenCodec constructor.
        dino_backbone (dict): Configuration dictionary for the DINOv2 backbone.
            Passed directly to Dinov2TimmBackbone constructor.
        dino_codec (dict): Configuration dictionary for the DINO codec.
            Must contain "h_dim" and "ctx_dim" keys for the conditional decoder.
            Passed directly to VitUnionLatentCodecCtxAsHyper constructor.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        vqgan (VqganBackbone): The VQGAN backbone model.
        vqgan_codec (UniformTokenCodec): The VQGAN codec.
        dino (Dinov2TimmBackbone): The DINOv2 backbone model.
        dino_codec (VitUnionLatentCodecCtxAsHyper): The DINO codec with context as hyperprior.
        patch_size (int): Patch size used by the DINOv2 backbone.
        cond_dec_for_vqgan (nn.Sequential): Conditional decoder for enhancing
            VQGAN reconstruction using DINOv2 features.
    """

    def __init__(
        self,
        vqgan_backbone={},
        vqgan_codec={},
        dino_backbone={},
        dino_codec={},
        **kwargs,
    ):
        super().__init__()
        self.vqgan = VqganBackbone(vqgan_backbone)
        self.vqgan_codec = UniformTokenCodec(**vqgan_codec)
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.dino_codec = VitUnionLatentCodecCtxAsHyper(**dino_codec)
        self.patch_size = self.dino.patch_size

        # Additional branch for enhancing VQGAN reconstruction using DINOv2 features
        D_DINO = dino_codec["h_dim"]
        D_VQGAN = dino_codec["ctx_dim"]
        self.cond_dec_for_vqgan = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
        )

    def forward(self, x, **kwargs):
        """
        Forward pass for training.

        Processes input through both VQGAN and DINOv2 layers, performs conditional
        compression with context as hyperprior, and returns features and likelihoods.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            out (dict): Dictionary containing:

                - "h_vqgan": Original VQGAN features
                - "h_vqgan_hat": Enhanced VQGAN features
                - "h_dino": Original DINO features
                - "h_dino_hat": Reconstructed DINO features
                - "likelihoods": Likelihoods from the DINO codec
                - "x_hat": Reconstructed image (only in eval mode, None during training)
        """
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            # Note: vqgan_codec is not used here as it only provides constant likelihoods
            h_vqgan = vqgan_enc["z"]
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            o_dino = self.dino.decode_whole(h_dino)[-1]

        h_dino = h_dino.clone()
        h_vqgan_ctx = h_vqgan_ctx.clone()
        dino_out = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        h_hat_for_vqgan = self.cond_dec_for_vqgan(
            torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
        )

        if not self.training:
            with torch.no_grad():
                x_hat = self.vqgan.decode(h_hat_for_vqgan)
        else:
            x_hat = None

        return {
            "h_vqgan": h_vqgan.clone(),
            "h_vqgan_hat": h_hat_for_vqgan,
            "h_dino": o_dino.clone(),
            "h_dino_hat": o_dino_hat,
            "likelihoods": dino_out["likelihoods"],
            "x_hat": x_hat,
        }

    def forward_test(self, x, tasks, **kwargs):
        """
        Forward pass for testing/inference with compression.

        Processes input through both layers, compresses features with context as
        hyperprior, and generates task-specific outputs.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            tasks (list of str): List of tasks to perform. Supported tasks:

                - "rec": Reconstruction; branch chosen by ``recon_branch`` in kwargs (1=VQGAN-only, 2=enhanced).
                - "cls": Classification task
                - "semseg": Semantic segmentation task
            **kwargs (dict): Optional ``recon_branch`` (int, default 2): for ``rec`` only.

        Returns:
            coded_data (dict): Dictionary containing compressed data:

                - "type": "frame"
                - "data": Dictionary with:
                    - "layer1": VQGAN compressed data
                    - "layer2": DINO compressed data

            task_feats (dict): Dictionary of task-specific features:

                - "rec": Reconstruction (if "rec" in tasks)
                - "cls": Classification features (if "cls" in tasks)
                - "semseg": Segmentation features (if "semseg" in tasks)
        """
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            # Note: vqgan_codec only provides constant likelihoods
            vqgan_cu = self.vqgan_codec(vqgan_enc["tokens"])
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            dino_cu = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
            h_dino_hat = dino_cu["h_hat"]

            task_feats = {}
            recon_branch = int(kwargs.get("recon_branch", 2))
            if "rec" in tasks:
                if recon_branch == 1:
                    task_feats["rec"] = self.tokenizer.decode(h_vqgan_ctx)
                else:
                    h_hat_for_vqgan = self.cond_dec_for_vqgan(
                        torch.cat([dino_cu["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
                    )
                    task_feats["rec"] = self.tokenizer.decode(h_hat_for_vqgan)
            if "cls" in tasks:
                task_feats["cls"] = self.dino.decode_cls(h_dino_hat)
            if "semseg" in tasks:
                task_feats["semseg"] = self.dino.decode_seg(h_dino_hat, token_res)

            coded_data = {
                "type": "frame",
                "data": {
                    "layer1": vqgan_cu,
                    "layer2": dino_cu,
                },
            }
            return coded_data, task_feats

    def compress(self, x, **kwargs):
        """
        Compress input image to byte strings using both layers.

        Processes input through both VQGAN and DINOv2 layers, compresses features
        with context as hyperprior, and returns compressed data.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            coded_data (dict): Dictionary containing compressed data:

                - "type": "frame"
                - "data": Dictionary with:
                    - "layer1": VQGAN compressed data
                    - "layer2": DINO compressed data
        """
        vqgan_enc = self.vqgan.encode(x)
        vqgan_cu = self.vqgan_codec.compress(vqgan_enc["tokens"])
        h_vqgan_ctx = vqgan_enc["z_q"]

        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        dino_cu = self.dino_codec.compress(h_dino, h_vqgan_ctx, token_res)

        coded_data = {
            "type": "frame",
            "data": {
                "layer1": vqgan_cu,
                "layer2": dino_cu,
            },
        }
        return coded_data

    def decompress(self, coded_data, tasks=[], **kwargs):
        """
        Decompress byte strings to task-specific features.

        Decompresses both VQGAN and DINOv2 layers, uses context as hyperprior for
        DINO decompression, and generates task-specific outputs.

        Args:
            coded_data (dict): Dictionary containing compressed data:

                - "type": "frame"
                - "data": Dictionary with:
                    - "layer1": VQGAN compressed data
                    - "layer2": DINO compressed data

            tasks (list of str): List of tasks to perform. Supported tasks:

                - "rec": Reconstruction; branch chosen by ``recon_branch`` in kwargs (1=VQGAN-only, 2=enhanced).
                - "cls": Classification task
                - "semseg": Semantic segmentation task

            **kwargs (dict): Optional ``recon_branch`` (int, default 2): for ``rec`` only.

        Returns:
            task_feats (dict): Dictionary of task-specific features:

                - "rec": Reconstruction (if "rec" in tasks)
                - "cls": Classification features (if "cls" in tasks)
                - "semseg": Segmentation features (if "semseg" in tasks)
        """
        vqgan_cu = coded_data["data"]["layer1"]
        dino_cu = coded_data["data"]["layer2"]
        token_res = dino_cu["pstate"]["token_res"]
        vqgan_decoded = self.vqgan_codec.decompress(**vqgan_cu)
        h_vqgan_ctx = self.vqgan.tokens_to_features(vqgan_decoded["tokens"])
        dino_decoded = self.dino_codec.decompress(**dino_cu, ctx=h_vqgan_ctx)

        task_feats = {}
        recon_branch = int(kwargs.get("recon_branch", 2))
        if "rec" in tasks:
            if recon_branch == 1:
                task_feats["rec"] = self.tokenizer.decode(h_vqgan_ctx)
            else:
                h_hat_for_vqgan = self.cond_dec_for_vqgan(
                    torch.cat([dino_decoded["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
                )
                task_feats["rec"] = self.tokenizer.decode(h_hat_for_vqgan)
        if "cls" in tasks:
            task_feats["cls"] = self.dino.decode_cls(dino_decoded["h_hat"])
        if "semseg" in tasks:
            task_feats["semseg"] = self.dino.decode_seg(dino_decoded["h_hat"], token_res)

        return task_feats

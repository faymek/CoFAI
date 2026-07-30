"""
Backbone implementations using timm library.

Available Model Names in timm: https://huggingface.co/timm/collections

1. DINOv2 Models (auto-generated based on model_size and with_registers):
   - 'vit_small_patch14_dinov2.lvd142m'
   - 'vit_small_patch14_reg4_dinov2.lvd142m' (with registers)
   - 'vit_base_patch14_dinov2.lvd142m'
   - 'vit_base_patch14_reg4_dinov2.lvd142m' (with registers)
   - 'vit_large_patch14_dinov2.lvd142m'
   - 'vit_large_patch14_reg4_dinov2.lvd142m' (with registers)
   - 'vit_giant_patch14_dinov2.lvd142m'
   - 'vit_giant_patch14_reg4_dinov2.lvd142m' (with registers)

2. MAE Models:
   - 'vit_base_patch16_224.mae' (default, standard MAE model)
   - 'vit_base_patch16_mae.in1k' (alternative naming)

3. SigLIP2 Models:
   - 'vit_base_patch16_siglip_224' (default, standard SigLIP model)
   - 'vit_base_patch16_siglip.in1k' (alternative naming)
   - 'vit_base_patch16_siglip_256' (alternative size)

Note: All backbones in this module use only timm library implementations.
"""

import os
from functools import partial

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from einops import rearrange
from .base import parse_dtype, BackboneProtocol
import timm


class Dinov2TimmBackbone(nn.Module):
    """
    DINOv2 backbone using timm library.

    This class extends the DINOv2 model to provide flexible feature extraction.
    The DINOv2 backbone implemented with timm supports variable patch sizes and dynamic input image sizes.
    The `slot` parameter determines the splitting point for dividing the ViT blocks into:

    * encode part: blocks[:slot],
    * decode part: blocks[slot:]

    Intermediate feature are extracted after the encode part and before the decode part.

    Args:
        model_size (str): Model variant specification ('small', 'base', 'large', 'giant'). Defaults to 'small'.
        img_size (int): Base input image size. Defaults to 256.
        patch_size (int): Patch embedding size. Defaults to 16.
        dynamic_size (bool): Whether to support dynamically varying input sizes. Defaults to False.
        slot (int or None): Block slicing position for feature extraction. Follows Python list slicing conventions.
                   Defaults to -4.
        n_last_blocks (int): Number of final blocks to utilize for feature aggregation. Defaults to 4.
        ckpt_path (str, optional): Path to pre-trained checkpoint for initialization. Defaults to None.
        cast_dtype (str or torch.dtype): Data type for autocast mixed precision.
                   Supports string format like "torch.float", "torch.float16", "float32", etc.
                   Defaults to "torch.float".
        device (str): Device to run the model on. Defaults to "cuda" if available, else "cpu".
        with_registers (bool): Whether to use register tokens in the model. Defaults to False.

    """

    def __init__(
        self,
        model_size: str = "small",
        img_size: int = 256,
        patch_size: int = 16,
        dynamic_size: bool = False,
        slot: int = -4,  # cut position
        n_last_blocks: int = 4,  # number of last blocks to take
        ckpt_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cast_dtype: str = "float",  # Data type for autocast, supports string configuration
        with_registers: bool = False,
        rae_decode_blocks: int | None = None,
    ):
        super().__init__()
        self.n_last_blocks = n_last_blocks
        assert model_size in ["small", "base", "large", "giant"]
        self.model_size = model_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.dynamic_size = dynamic_size
        # ``slot`` accepts an optional trailing "n" marker (e.g. "40n", "-1n", "n")
        # meaning: after cutting at the integer position, apply the final
        # ``model.norm`` at the *encode* boundary (and skip it again on decode).
        # This lets a feature codec compress the post-norm features that the
        # bespoke "compress the final seg features" models used (see
        # ``Dinov2TimmSegVQFC``), making such setups rate-equivalent.
        self.encode_norm = False
        if isinstance(slot, str):
            s = slot.strip()
            if s.endswith("n"):
                self.encode_norm = True
                s = s[:-1].strip()
            slot = None if s == "" else int(s)
        self.slot = slot
        self.n_last_blocks = n_last_blocks
        self.ckpt_path = ckpt_path
        self.device = device
        self.device_type = self.device.split(":")[0]
        self.cast_dtype = parse_dtype(cast_dtype)
        self.with_registers = with_registers
        self.rae_decode_blocks = rae_decode_blocks
        self.model = self.load_timm_model()
        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        # Set hidden_size and patch_size attributes for Protocol
        self.hidden_size = self.model.embed_dim
        self.patch_size = (
            self.model.patch_embed.proj.kernel_size[0]
            if hasattr(self.model.patch_embed.proj, "kernel_size")
            else patch_size
        )

    def load_timm_model(self):
        model_name = f"vit_{self.model_size}_patch14_dinov2.lvd142m"
        if self.with_registers:
            model_name = f"vit_{self.model_size}_patch14_reg4_dinov2.lvd142m"

        if self.ckpt_path is not None:
            from timm.models._helpers import load_checkpoint
            from timm.models.vision_transformer import checkpoint_filter_fn

            ckpt_path = os.path.expanduser(str(self.ckpt_path))
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Local checkpoint not found: {ckpt_path}")
            feature_model = timm.create_model(
                model_name,
                pretrained=False,
                img_size=self.img_size,
                patch_size=self.patch_size,
                drop_path_rate=0.0,
                dynamic_img_size=self.dynamic_size,
            )
            load_checkpoint(
                feature_model,
                ckpt_path,
                use_ema=False,
                strict=True,
                # Preserve the proposal's released non-antialiased bicubic
                # positional-embedding resampling.
                filter_fn=partial(checkpoint_filter_fn, antialias=False),
            )
        else:
            feature_model = timm.create_model(
                model_name,
                pretrained=True,
                img_size=self.img_size,
                patch_size=self.patch_size,
                drop_path_rate=0.0,
                dynamic_img_size=self.dynamic_size,
            )
        feature_model.eval()
        return feature_model

    def forward(self, x, task="whole"):
        """Forward pass through the backbone.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
            task (str): Task type, one of ["whole", "cls", "seg"]. Defaults to "whole".

        Returns:
            feats (list[torch.Tensor]): Output features, format depends on task.
        """
        assert task in ["whole", "cls", "seg"]
        with torch.inference_mode():
            h = self.encode(x)
            token_res = (x.size(2) // self.patch_size, x.size(3) // self.patch_size)
            h = self.decode(h, token_res=token_res, task=task)
            return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images through the encoder part of the DINOv2 model.

        The encoding process applies input normalization, patch embedding, positional
        embedding, and processes the input through the first `slot` transformer blocks.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            h (torch.Tensor): Encoded features after the encoder blocks, shape (B, N, C).
        """
        dino = self.model
        with torch.autocast(device_type=self.device_type, dtype=self.cast_dtype):
            x = self.input_transform(x)
            x = dino.patch_embed(x)
            x = dino._pos_embed(x)
            x = dino.patch_drop(x)
            x = dino.norm_pre(x)
            for i, blk in enumerate(dino.blocks[: self.slot]):
                x = blk(x)
            if self.encode_norm:
                x = dino.norm(x)
        return x

    def decode(self, h, token_res=None, task="whole"):
        """Decode encoded features through the decoder part of the DINOv2 model.

        Args:
            h (torch.Tensor): Encoded features from the encoder.
            token_res (tuple, optional): Token resolution (H, W) for reshaping patch tokens.
                                        Defaults to None.
            task (str): Decoding task type. Must be one of:

                - "whole": Return full token sequences from multiple layers.
                - "cls": Return class tokens and patch tokens separately.
                - "seg": Return patch tokens reshaped to 2D spatial format.
                Defaults to "whole".

        Returns:
            feats (list[torch.Tensor, ...]): Decoded features, format depends on task.
        """
        if task == "whole":
            return self.decode_whole(h, token_res=token_res)
        elif task == "cls":
            return self.decode_cls(h, token_res=token_res)
        elif task == "seg":
            return self.decode_seg(h, token_res=token_res)

    def _decode(
        self,
        x: torch.Tensor,
        slot: int = -4,
        n: int = 4,
        norm: bool = True,
        return_format: str = "[whole]",
        token_res: tuple = None,
    ) -> list:
        """Internal decode method with flexible return formats."""
        allow_formats = [
            "[whole]",
            "[cls,patch]",
            "[cls]",
            "[patch]",
            "[patch2d]",
            "[patch_rae]",
        ]
        assert return_format in allow_formats, (
            f"return_format must be one of {allow_formats}"
        )
        dino = self.model

        multi_outputs = []

        # If n is an int, take the n last blocks. If it's a list, take them
        total_layers = len(dino.blocks)
        if isinstance(n, int):
            need_layers = range(total_layers - n, total_layers)
        elif isinstance(n, list):
            need_layers = n

        # locate the input feature x is after the layer of curr_layer
        if slot is None:
            curr_layer = total_layers - 1
        elif isinstance(slot, int):
            if slot < 0:
                curr_layer = total_layers + slot - 1
            else:
                curr_layer = slot - 1
        else:
            raise ValueError(f"slot must be an int or None, got {type(slot)}")

        if curr_layer > min(need_layers):
            raise ValueError(
                f"not possible to take required layers, input layer: {curr_layer}, need layers: {need_layers}"
            )
        elif curr_layer == min(need_layers):
            # input feature is just needed
            multi_outputs.append(x)

        # Use autocast for mixed precision computation
        with torch.autocast(device_type=self.device_type, dtype=self.cast_dtype):
            for i in range(curr_layer + 1, total_layers):
                x = dino.blocks[i](x)
                if i in need_layers:
                    multi_outputs.append(x)

            assert len(multi_outputs) == len(need_layers), (
                f"only {len(multi_outputs)} / {len(need_layers)} blocks found"
            )

            if norm and not self.encode_norm:
                if return_format == "[patch_rae]":
                    # For RAE: use LayerNorm without learnable affine parameters
                    # This matches Dinov2TimmwithNorm with normalize=True
                    for i, out in enumerate(multi_outputs):
                        # Manual layer norm without affine parameters
                        mean = out.mean(dim=-1, keepdim=True)
                        var = out.var(dim=-1, keepdim=True, unbiased=False)
                        multi_outputs[i] = (out - mean) / torch.sqrt(
                            var + dino.norm.eps
                        )
                else:
                    multi_outputs = [dino.norm(out) for out in multi_outputs]

        if return_format == "[whole]":
            return multi_outputs

        multi_class_tokens = [out[:, 0] for out in multi_outputs]
        multi_patch_tokens = [out[:, dino.num_prefix_tokens :] for out in multi_outputs]

        if return_format == "[cls,patch]":
            # feature list: [[cls token, patch tokens], ..., [cls token, patch tokens]]
            return tuple(zip(multi_class_tokens, multi_patch_tokens))
        elif return_format == "[cls]":
            return multi_class_tokens
        elif return_format == "[patch]":
            return multi_patch_tokens
        elif return_format == "[patch2d]":
            h, w = token_res
            multi_patch_tokens = [
                rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
                for out in multi_patch_tokens
            ]
            return multi_patch_tokens

        elif return_format == "[patch_rae]":
            # Return patch tokens for RAE (already extracted from normalized outputs)
            return multi_patch_tokens

    def decode_whole(self, h, token_res=None):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[whole]",
            token_res=token_res,
        )

    def decode_cls(self, h, token_res=None):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[cls,patch]",
            token_res=token_res,
        )

    def decode_seg(self, h, token_res):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[patch2d]",
            token_res=token_res,
        )

    def decode_rae(self, h, token_res=None):
        """Decode encoded features for RAE decoder input.

        By default this method follows the original MPCompress behavior: process the
        encoded features through the remaining transformer blocks and return final
        patch tokens. If ``rae_decode_blocks`` is configured, it instead runs exactly
        that many blocks after ``slot``; ``0`` means using the encoder output directly.

        It returns patch tokens without prefix tokens for RAE decoder.
        The input h should be the output from the encode method (after blocks[:slot]).

        This method uses LayerNorm without learnable affine parameters (matching
        Dinov2TimmwithNorm with normalize=True). This equals to simple normalization.
        """
        if self.rae_decode_blocks is None:
            multi_outputs = self._decode(
                h,
                slot=self.slot,
                n=1,  # Take only the last layer
                norm=False,
                return_format="[patch]",
                token_res=token_res,
            )
            out: torch.Tensor = multi_outputs[0]
        else:
            if self.rae_decode_blocks < 0:
                raise ValueError("rae_decode_blocks must be >= 0 when configured.")
            if self.rae_decode_blocks == 0:
                out = h
            else:
                out = self._decode_latter_blocks(
                    h,
                    slot=self.slot,
                    n_blocks=self.rae_decode_blocks,
                    norm=False,
                    token_res=token_res,
                )
            out = out[:, self.model.num_prefix_tokens :]

        mean = out.mean(dim=-1, keepdim=True)
        var = out.var(dim=-1, keepdim=True, unbiased=False)
        out = (out - mean) / torch.sqrt(var + self.model.norm.eps)

        return out

    def _decode_latter_blocks(self, h, slot=-3, n_blocks=1, norm=True, token_res=None):
        # for example, dinov2 base has 12 blocks in total, named as [0, 1, 2, ..., 11]
        # if slot is -3, it means the input feature is after the blocks[8], it the 9th block
        # if n_blocks is 2, it means apply the blocks[9-10] and the begin layernorm of blocks[11]
        # if n_blocks is 3, it means apply the blocks[9-11] and last layernorm of the model
        if slot < 0:
            curr_layer = len(self.model.blocks) + slot - 1
        else:
            curr_layer = slot - 1
        last_layer = len(self.model.blocks) - 1
        assert curr_layer + n_blocks <= last_layer, (
            f"not possible to decode latter {n_blocks} blocks, current layer: {curr_layer}, total layers: {last_layer}"
        )

        # Use autocast for mixed precision computation
        with torch.autocast(device_type=self.device_type, dtype=self.cast_dtype):
            for i in range(curr_layer + 1, curr_layer + n_blocks + 1):
                h = self.model.blocks[i](h)
        if norm:
            # we'll retrieve the layernorm of the next vit block if norm required
            if curr_layer + n_blocks == last_layer:
                h = self.model.norm(h)
            else:
                h = self.model.blocks[curr_layer + n_blocks + 1].norm1(h)
        return h


class MAETimmBackbone(nn.Module):
    """
    MAE (Masked Autoencoder) backbone using timm library.

    This backbone uses ViT-MAE from timm library and provides
    encode/decode functionality for RAE.

    Available model names in timm:
        - 'vit_base_patch16_224.mae' (default, standard MAE model)
        - 'vit_base_patch16_mae.in1k' (alternative naming)

    Args:
        model_name (str): Timm model name, e.g., 'vit_base_patch16_224.mae'.
                         Defaults to 'vit_base_patch16_224.mae'.
        img_size (int): Input image size. Defaults to 256.
        patch_size (int): Patch embedding size. Defaults to 16.
        slot (int): Block slicing position for feature extraction.
                   -1 means use all blocks. Defaults to -1.
        cast_dtype (str or torch.dtype): Data type for autocast mixed precision.
                   Defaults to "float".
        device (str): Device to run the model on. Defaults to "cuda" if available, else "cpu".
    """

    # Available MAE model names in timm (will be tried in order)
    AVAILABLE_MAE_MODELS = [
        "vit_base_patch16_224.mae",  # Standard MAE model
        "vit_base_patch16_mae.in1k",  # Alternative naming
    ]

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224.mae",
        img_size: int = 256,
        patch_size: int = 16,
        slot: int = -1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cast_dtype: str = "float",
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.slot = slot
        self.device = device
        self.device_type = self.device.split(":")[0]
        self.cast_dtype = parse_dtype(cast_dtype)

        # Try to load MAE model from timm
        mae_model_names = [model_name] + self.AVAILABLE_MAE_MODELS
        self.model = None

        for name in mae_model_names:
            try:
                self.model = timm.create_model(
                    name,
                    pretrained=True,
                    img_size=img_size,
                    patch_size=patch_size,
                )
                self.hidden_size = self.model.embed_dim
                # Disable learnable parameters in final norm (for RAE compatibility)
                if hasattr(self.model, "norm") and self.model.norm is not None:
                    self.model.norm.elementwise_affine = False
                    self.model.norm.weight = None
                    self.model.norm.bias = None
                if name != model_name:
                    print(f"Info: Using timm model '{name}' instead of '{model_name}'")
                break
            except Exception:
                continue

        if self.model is None:
            raise RuntimeError(
                f"Failed to load MAE model from timm. Tried: {mae_model_names}"
            )

        # Use ImageNet normalization (standard for timm ViT models)
        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )

        self.model.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images through the MAE encoder.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            h (torch.Tensor): Encoded features, shape (B, N, C).
        """
        with torch.autocast(device_type=self.device_type, dtype=self.cast_dtype):
            x = self.input_transform(x)
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
            x = self.model.norm_pre(x)
            for blk in self.model.blocks:
                x = blk(x)
            h = x

        return h


class SigLIP2TimmBackbone(nn.Module):
    """
    SigLIP2 backbone using timm library.

    This backbone uses SigLIP2 from timm library and provides
    encode/decode functionality for RAE.

    Available model names in timm:
        - 'vit_base_patch16_siglip_224' (default, standard SigLIP model)
        - 'vit_base_patch16_siglip.in1k' (alternative naming)
        - 'vit_base_patch16_siglip_256' (alternative size)

    Args:
        model_name (str): Timm model name for SigLIP2.
                         Defaults to 'vit_base_patch16_siglip_224'.
        img_size (int): Input image size. Defaults to 224.
        patch_size (int): Patch embedding size. Defaults to 16.
        slot (int): Block slicing position for feature extraction.
                   -1 means use all blocks. Defaults to -1.
        cast_dtype (str or torch.dtype): Data type for autocast mixed precision.
                   Defaults to "float".
        device (str): Device to run the model on. Defaults to "cuda" if available, else "cpu".
    """

    # Available SigLIP2 model names in timm (will be tried in order)
    AVAILABLE_SIGLIP_MODELS = [
        "vit_base_patch16_siglip_224",  # Standard SigLIP model
        "vit_base_patch16_siglip.in1k",  # Alternative naming
        "vit_base_patch16_siglip_256",  # Alternative size
    ]

    def __init__(
        self,
        model_name: str = "vit_base_patch16_siglip_224",
        img_size: int = 224,
        patch_size: int = 16,
        slot: int = -1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cast_dtype: str = "float",
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.slot = slot
        self.device = device
        self.device_type = self.device.split(":")[0]
        self.cast_dtype = parse_dtype(cast_dtype)

        # Try to load SigLIP2 model from timm
        siglip_model_names = [model_name] + self.AVAILABLE_SIGLIP_MODELS
        self.model = None

        for name in siglip_model_names:
            try:
                self.model = timm.create_model(
                    name,
                    pretrained=True,
                    img_size=img_size,
                    patch_size=patch_size,
                )
                self.hidden_size = self.model.embed_dim
                # Disable learnable parameters in final norm (for RAE compatibility)
                if hasattr(self.model, "norm") and self.model.norm is not None:
                    self.model.norm.elementwise_affine = False
                    self.model.norm.weight = None
                    self.model.norm.bias = None
                if name != model_name:
                    print(f"Info: Using timm model '{name}' instead of '{model_name}'")
                break
            except Exception:
                continue

        if self.model is None:
            raise RuntimeError(
                f"Failed to load SigLIP2 model from timm. Tried: {siglip_model_names}"
            )

        # Use SigLIP2 normalization (timm SigLIP2 models expect mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # This matches the timm model's default_cfg
        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.model.eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input images through the SigLIP2 encoder.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            h (torch.Tensor): Encoded features, shape (B, N, C).
        """
        with torch.autocast(device_type=self.device_type, dtype=self.cast_dtype):
            x = self.input_transform(x)
            x = self.model.patch_embed(x)
            x = self.model._pos_embed(x)
            x = self.model.norm_pre(x)
            for blk in self.model.blocks:
                x = blk(x)
            # Apply final norm (without affine parameters, matching transformers version)
            # This matches the behavior of transformers SiglipVisionModel which applies post_layernorm
            if hasattr(self.model, "norm") and self.model.norm is not None:
                h = self.model.norm(x)
            else:
                h = x

        return h

class Dinov3TimmBackbone(nn.Module):
    """
    DINOv3 backbone implemented with timm.

    Key differences vs DINOv2:
      - DINOv3 always includes register tokens
      - timm model names are fixed to patch16
      - model sizes include: large / huge_plus / 7b
      - qkvb is an explicit model variant

    This class keeps the same encode/decode/slot abstraction as DINOv2 version.
    """

    def __init__(
        self,
        model_size="large",
        img_size=256,
        patch_size=16,
        dynamic_size=False,
        slot=-4,
        n_last_blocks=4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cast_dtype="float",
        # ---- DINOv3-specific ----
        qkvb=False,
        weights_tag="lvd1689m",
        return_registers=False,
        # ---- NEW: offline/local weights ----
        ckpt_path=None,
        pretrained=True,
    ):
        super().__init__()

        assert patch_size == 16, "timm DINOv3 models are patch16 only."

        self.model_size = model_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.dynamic_size = dynamic_size
        self.slot = slot
        self.n_last_blocks = n_last_blocks

        self.qkvb = qkvb
        self.weights_tag = weights_tag
        self.return_registers = return_registers

        self.device = device
        self.device_type = device.split(":")[0]
        self.cast_dtype = parse_dtype(cast_dtype)

        self.ckpt_path = ckpt_path
        self.pretrained = pretrained

        self.model = self.load_timm_model()

        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060],
                    std=[0.2290, 0.2240, 0.2250],
                )
            ]
        )

        self._rope = None
        self._attn_mask = None

    def load_timm_model(self):
        model_name = f"vit_{self.model_size}_patch16_dinov3.{self.weights_tag}"
        # 1) Prefer local checkpoint when provided.
        if self.ckpt_path is not None:
            ckpt_path = os.path.expanduser(str(self.ckpt_path))
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Local checkpoint not found: {ckpt_path}")

            model = timm.create_model(
                model_name,
                pretrained=False,
                checkpoint_path=self.ckpt_path,
                img_size=self.img_size,
                patch_size=self.patch_size,
                drop_path_rate=0.0,
                dynamic_img_size=self.dynamic_size,
            )
            model.eval()
            return model

        # 2) Otherwise use HF Hub entry in timm.
        _normalize_hf_endpoint_env()
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        if getattr(self, "hf_repo_id", None):
            repo_id = self.hf_repo_id
        else:
            size_map = {
                "small": "vits16",
                "base": "vitb16",
                "large": "vitl16",
                "huge_plus": "vith16plus",
                "7b": "vit7b16",
            }
            if self.model_size not in size_map:
                raise ValueError(f"Unsupported model_size: {self.model_size}")
            repo_id = f"facebook/dinov3-{size_map[self.model_size]}-pretrain-{self.weights_tag}"

        hf_model_name = f"hf_hub:{repo_id}"

        model = timm.create_model(
            hf_model_name,
            pretrained=self.pretrained,
            img_size=self.img_size,
            patch_size=self.patch_size,
            drop_path_rate=0.0,
            dynamic_img_size=self.dynamic_size,
        )
        model.eval()
        return model

    def forward(self, x, task="whole"):
        """
        task:
          - "whole": list of full token tensors
          - "cls":   cls (+ optional reg) + patch tokens
          - "seg":   patch tokens reshaped to 2D
          - "depth": patch tokens reshaped to 2D
        """
        assert task in ["whole", "cls", "seg", "depth"]

        with torch.inference_mode():
            h = self.encode(x)
            token_res = (
                x.size(2) // self.patch_size,
                x.size(3) // self.patch_size,
            )
            return self.decode(h, token_res=token_res, task=task)

    def encode(self, x):
        dino = self.model

        with torch.autocast(device_type=self.device_type, dtype=self.cast_dtype):
            x = self.input_transform(x)
            x = dino.patch_embed(x)

            pos_out = dino._pos_embed(x)
            rope = None
            attn_mask = None
            if isinstance(pos_out, tuple):
                x = pos_out[0]
                if len(pos_out) >= 2:
                    rope = pos_out[1]
                if len(pos_out) >= 3:
                    attn_mask = pos_out[2]
            else:
                x = pos_out

            # cache for decode()
            self._rope = rope
            self._attn_mask = attn_mask

            norm_pre = getattr(dino, "norm_pre", None)
            if norm_pre is not None:
                x = norm_pre(x)

            rope_mixed = bool(getattr(dino, "rope_mixed", False))
            for i, blk in enumerate(dino.blocks[: self.slot]):
                rope_i = None
                if self._rope is not None:
                    rope_i = self._rope[i] if rope_mixed else self._rope

                if rope_i is not None or self._attn_mask is not None:
                    x = blk(x, rope=rope_i, attn_mask=self._attn_mask)
                else:
                    x = blk(x)

        return x

    def build_frozen_tail(
        self,
        layer_idx: int,
        token_hw: tuple[int, int],
        device: torch.device | str,
    ) -> nn.Module:
        """Build the frozen tail after ``layer_idx`` with spatial RoPE primed."""
        from .frozen_tail import Dinov3FrozenTail, FrozenTail

        n_blocks = len(self.model.blocks)
        if not -1 <= layer_idx < n_blocks:
            raise ValueError(f"layer_idx must be in [-1, {n_blocks - 1}], got {layer_idx}")
        if layer_idx == n_blocks - 1:
            return FrozenTail([], self.model.norm).to(device)

        self.to(device).eval()
        height, width = token_hw
        image = torch.zeros(
            1,
            3,
            height * self.patch_size,
            width * self.patch_size,
            device=device,
        )
        with torch.no_grad():
            self.encode(image)
        return Dinov3FrozenTail(
            self.model,
            layer_idx,
            self._rope,
            self._attn_mask,
        ).to(device)

    def decode(self, h, token_res=None, task="whole"):
        if task == "whole":
            return self.decode_whole(h, token_res)
        elif task == "cls":
            return self.decode_cls(h, token_res)
        elif task == "seg":
            return self.decode_seg(h, token_res)
        elif task == "depth":
            return self.decode_depth(h, token_res)

    def _decode(
        self,
        x,
        slot,
        n,
        norm=True,
        return_format="[whole]",
        token_res=None,
    ):
        allow_formats = [
            "[whole]",
            "[cls,patch]",
            "[cls,reg,patch]",
            "[patch2d]",
        ]
        assert return_format in allow_formats

        dino = self.model
        total_layers = len(dino.blocks)

        if isinstance(n, int):
            need_layers = range(total_layers - n, total_layers)
        else:
            need_layers = n

        if slot is None:
            curr_layer = total_layers - 1
        elif slot < 0:
            curr_layer = total_layers + slot - 1
        else:
            curr_layer = slot - 1

        outputs = []

        if curr_layer == min(need_layers):
            outputs.append(x)

        with torch.autocast(
            device_type=self.device_type, dtype=self.cast_dtype
        ):
            rope_mixed = bool(getattr(dino, "rope_mixed", False))
            for i in range(curr_layer + 1, total_layers):
                rope_i = None
                if self._rope is not None:
                    rope_i = self._rope[i] if rope_mixed else self._rope

                if rope_i is not None or self._attn_mask is not None:
                    x = dino.blocks[i](x, rope=rope_i, attn_mask=self._attn_mask)
                else:
                    x = dino.blocks[i](x)

                if i in need_layers:
                    outputs.append(x)

            if norm:
                outputs = [dino.norm(o) for o in outputs]

        if return_format == "[whole]":
            return outputs

        cls_tokens = [o[:, 0] for o in outputs]
        reg_tokens = [o[:, 1 : dino.num_prefix_tokens] for o in outputs]
        patch_tokens = [o[:, dino.num_prefix_tokens :] for o in outputs]

        if return_format == "[cls,patch]":
            return tuple(zip(cls_tokens, patch_tokens))

        if return_format == "[cls,reg,patch]":
            return tuple(zip(cls_tokens, reg_tokens, patch_tokens))

        if return_format == "[patch2d]":
            h, w = token_res
            return [
                rearrange(p, "b (h w) c -> b c h w", h=h, w=w)
                for p in patch_tokens
            ]

    def decode_whole(self, h, token_res=None):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[whole]",
            token_res=token_res,
        )

    def decode_cls(self, h, token_res=None):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format=(
                "[cls,reg,patch]" if self.return_registers else "[cls,patch]"
            ),
            token_res=token_res,
        )

    def decode_seg(self, h, token_res):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[patch2d]",
            token_res=token_res,
        )

    def decode_depth(self, h, token_res):
        return self._decode(
            h,
            slot=self.slot,
            n=self.n_last_blocks,
            norm=True,
            return_format="[patch2d]",
            token_res=token_res,
        )

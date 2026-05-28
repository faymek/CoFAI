"""RAE decoder head, aligned with MPCompress' GeneralDecoder."""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import ModelOutput

from cofai.engine.registry import register


class ViTMAEConfig(PretrainedConfig):
    model_type = "vit_mae"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        mask_ratio=0.75,
        norm_pix_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ViTMAESelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of "
                f"the number of attention heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return (context_layer, attention_probs) if output_attentions else (context_layer,)


class ViTMAESelfOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTMAEAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAESelfAttention(config)
        self.output = ViTMAESelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class ViTMAEIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ViTMAEOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTMAELayer(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)
        return (layer_output,) + outputs


@register("GeneralDecoder")
class GeneralDecoder(nn.Module):
    def __init__(
        self,
        decoder_config: dict,
        patch_size: int = 16,
        image_size: int = 512,
        pretrained_path: Optional[str] = None,
        encoder_img_size: int = 512,
        encoder_patch_size: int = 16,
        encoder_hidden_size: int = 768,
        encoder_mean: Union[list, torch.Tensor] = None,
        encoder_std: Union[list, torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()

        if encoder_mean is None:
            raise ValueError("encoder_mean must be provided in the configuration.")
        if encoder_std is None:
            raise ValueError("encoder_std must be provided in the configuration.")

        if not isinstance(encoder_mean, torch.Tensor):
            encoder_mean = torch.tensor(encoder_mean).view(1, 3, 1, 1)
        elif encoder_mean.dim() == 1:
            encoder_mean = encoder_mean.view(1, 3, 1, 1)
        if not isinstance(encoder_std, torch.Tensor):
            encoder_std = torch.tensor(encoder_std).view(1, 3, 1, 1)
        elif encoder_std.dim() == 1:
            encoder_std = encoder_std.view(1, 3, 1, 1)

        num_patches = (encoder_img_size // encoder_patch_size) ** 2
        cfg_params = decoder_config.copy()
        cfg_params.update(
            {
                "hidden_size": encoder_hidden_size,
                "patch_size": patch_size,
                "image_size": image_size,
            }
        )
        cfg = ViTMAEConfig(**cfg_params)

        self.decoder_embed = nn.Linear(
            cfg.hidden_size, cfg.decoder_hidden_size, bias=True
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, cfg.decoder_hidden_size),
            requires_grad=False,
        )

        layer_cfg = deepcopy(cfg)
        layer_cfg.hidden_size = cfg.decoder_hidden_size
        layer_cfg.num_hidden_layers = cfg.decoder_num_hidden_layers
        layer_cfg.num_attention_heads = cfg.decoder_num_attention_heads
        layer_cfg.intermediate_size = cfg.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(layer_cfg) for _ in range(cfg.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(
            cfg.decoder_hidden_size, eps=cfg.layer_norm_eps
        )
        self.decoder_pred = nn.Linear(
            cfg.decoder_hidden_size,
            cfg.patch_size**2 * cfg.num_channels,
            bias=True,
        )
        self.trainable_cls_token = nn.Parameter(
            torch.zeros(1, 1, cfg.decoder_hidden_size)
        )
        self.gradient_checkpointing = False
        self.config = cfg
        self.num_patches = num_patches
        self.decoder_config = layer_cfg
        self.initialize_weights(num_patches)
        self.register_buffer("encoder_mean", encoder_mean.float())
        self.register_buffer("encoder_std", encoder_std.float())

        if pretrained_path:
            print(f"[GeneralDecoder] loading weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            keys = self.load_state_dict(state_dict, strict=False)
            missing = [key for key in keys.missing_keys if key not in {"encoder_mean", "encoder_std"}]
            if missing:
                print(f"[GeneralDecoder] missing keys: {missing}")
            if keys.unexpected_keys:
                print(f"[GeneralDecoder] unexpected keys: {keys.unexpected_keys}")

        if device is not None:
            self.to(device)
        self.eval()

    def interpolate_pos_encoding(
        self, token_res: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        side = int(self.num_patches**0.5)
        if token_res is None:
            token_res = (side, side)
        class_pe = self.decoder_pos_embed[:, 0:1, :]
        patch_pe = self.decoder_pos_embed[:, 1:, :]
        patch_pe_2d = rearrange(patch_pe, "b (h w) c -> b c h w", h=side, w=side)
        patch_pe_2d = F.interpolate(
            patch_pe_2d,
            size=token_res,
            mode="bicubic",
            align_corners=False,
        )
        patch_pe = rearrange(patch_pe_2d, "b c h w -> b (h w) c")
        return torch.cat((class_pe, patch_pe), dim=1)

    def initialize_weights(self, num_patches):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(num_patches**0.5),
            add_cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

    def unpatchify(self, feature, token_res):
        patch_size = self.config.patch_size
        return rearrange(
            feature,
            "b (h w) (p q c) -> b c (h p) (w q)",
            h=token_res[0],
            w=token_res[1],
            p=patch_size,
            q=patch_size,
        )

    def forward(
        self,
        features: torch.Tensor,
        token_res: Optional[Tuple[int, int]] = None,
        token_format: str = "patch",
        **kwargs,
    ) -> torch.Tensor:
        assert token_format in ["cls,patch", "patch"]
        if token_res is None:
            side = int(features.shape[1] ** 0.5)
            token_res = (side, side)

        x_ = self.decoder_embed(features)
        if token_format == "cls,patch":
            x_ = x_[:, 1:, :]

        H, W = token_res
        if H * W != self.num_patches:
            decoder_pos_embed = self.interpolate_pos_encoding(token_res)
        else:
            decoder_pos_embed = self.decoder_pos_embed

        cls_token = self.trainable_cls_token.expand(x_.shape[0], -1, -1)
        h = torch.cat([cls_token, x_], dim=1)
        h = h + decoder_pos_embed

        for layer in self.decoder_layers:
            h = layer(h, head_mask=None, output_attentions=False)[0]

        h = self.decoder_norm(h)
        logits = self.decoder_pred(h)
        return logits[:, 1:, :]

    def predict(
        self,
        features: torch.Tensor,
        token_res: Optional[Tuple[int, int]] = None,
        token_format: str = "patch",
        clamp: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if token_res is None:
            side = int(features.shape[1] ** 0.5)
            token_res = (side, side)
        logits = self.forward(
            features,
            token_format=token_format,
            token_res=token_res,
        )
        x = self.unpatchify(logits, token_res)
        x = x * self.encoder_std.to(x.device) + self.encoder_mean.to(x.device)
        if clamp:
            x = x.clamp(0, 1)
        return x

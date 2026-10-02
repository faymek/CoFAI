from random import random

import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
)
from compressai.layers import (
    CheckerboardMaskedConv2d,
    sequential_channel_ramp,
)
from compressai.registry import register_model, register_module

from compressai.models.base import CompressionModel
from compressai.models.utils import conv, deconv
from einops import rearrange

    
from cofai.layers.vit import Block
from cofai.utils.tensor_ops import center_pad, border_pad

import numpy as np
import os

from cofai.layers.simple_depthwise_cnn import DepthwiseSeparableConv, DepthwiseSeparableDeconvPixelShuffle

@register_module("ChannelGroupsLatentCodecContiguous")
class ChannelGroupsLatentCodecContiguous(ChannelGroupsLatentCodec):
    def merge_y(self, *args):
        return torch.cat(args, dim=1).contiguous()

    def merge_params(self, *args):
        return torch.cat(args, dim=1).contiguous()


@register_model("SimpleHyperprior")
class SimpleHyperprior(CompressionModel): # [almost same as VTC]
    """Simplified Visual Token Codec for **patch-only** spatial features.

    Compresses spatial feature maps (B, C, H, W) directly without any prefix tokens.
    Designed for variable spatial resolutions with long side = 32.

    Args:
        h_dim (int): Channel dimension of input spatial features. Default 1152.
        y_dim (int): Channel dimension of primary latent representation ``y``.
        z_dim (int): Channel dimension of hyperprior latent representation ``z``.
        groups (int or list[int]): Channel groups for channel-wise context modeling.
        **kwargs (dict): Extra keyword arguments for compatibility (unused).
    """

    def __init__(
        self,
        h_dim=1152,
        y_dim=48,
        z_dim=48,
        groups=16,
        z_down_factor=4, # 2 or 4
        y_down_factor=2, # 1 or 2
        vit_layers = 1,

        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"
        
        
        

        self.h_dim = h_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.z_down_factor = z_down_factor

        # QP-indexed learnable gains (65 QP points for 0-64)
        self.q_scale_pre = nn.Parameter(torch.ones((65, 1, h_dim)))
        self.q_scale_post = nn.Parameter(torch.ones((65, 1, h_dim)))

        self.q_scale_enc = nn.Parameter(torch.ones((65, y_dim, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((65, y_dim, 1, 1)))

        if vit_layers > 0:
            self.pre_vit_blocks = nn.Sequential(
                *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(vit_layers)] 
            )
            self.post_vit_blocks = nn.Sequential(
                *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(vit_layers)] 
            )
        else:
            self.pre_vit_blocks = nn.Identity()
            self.post_vit_blocks = nn.Identity()

       
        if y_down_factor == 2:
            self.f_a = nn.Sequential(
                DepthwiseSeparableConv(h_dim, y_dim, kernel_size=3, stride=1),
                nn.ReLU(inplace=True), 
                DepthwiseSeparableConv(y_dim, y_dim, kernel_size=3, stride=1),
                nn.ReLU(inplace=True), 
                DepthwiseSeparableConv(y_dim, y_dim, kernel_size=5, stride=2),
            )

            self.f_s = nn.Sequential(
                DepthwiseSeparableDeconvPixelShuffle(y_dim, y_dim, kernel_size=5, stride=2),
                nn.ReLU(inplace=True),
                DepthwiseSeparableDeconvPixelShuffle(y_dim, y_dim, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                DepthwiseSeparableDeconvPixelShuffle(y_dim, h_dim, kernel_size=3, stride=1),
            )
        elif y_down_factor == 1:
            self.f_a = nn.Sequential(
                DepthwiseSeparableConv(h_dim, y_dim, kernel_size=3, stride=1),
                nn.ReLU(inplace=True), 
                DepthwiseSeparableConv(y_dim, y_dim, kernel_size=3, stride=1),
            )

            self.f_s = nn.Sequential(
                DepthwiseSeparableDeconvPixelShuffle(y_dim, y_dim, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                DepthwiseSeparableDeconvPixelShuffle(y_dim, h_dim, kernel_size=3, stride=1),
            )   
    

        if z_down_factor == 4:
            h_a = nn.Sequential(
                DepthwiseSeparableConv(y_dim, z_dim, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(z_dim, z_dim, kernel_size=5, stride=2),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(z_dim, z_dim, kernel_size=5, stride=2),
            )

            h_s = nn.Sequential(
                DepthwiseSeparableDeconvPixelShuffle(z_dim, z_dim, kernel_size=5, stride=2),
                nn.ReLU(inplace=True),
                DepthwiseSeparableDeconvPixelShuffle(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
                nn.ReLU(inplace=True),
                DepthwiseSeparableDeconvPixelShuffle(z_dim * 3 // 2 , z_dim * 2, kernel_size=3, stride=1),
            )

        elif z_down_factor == 2:
            h_a = nn.Sequential(
                DepthwiseSeparableConv(y_dim, z_dim, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(z_dim, z_dim, kernel_size=5, stride=2),
            )

            h_s = nn.Sequential(
                DepthwiseSeparableDeconvPixelShuffle(z_dim, z_dim, kernel_size=5, stride=2),
                nn.ReLU(inplace=True),
                DepthwiseSeparableDeconvPixelShuffle(z_dim, z_dim * 2, kernel_size=3, stride=1),
            )

        # Channel context (g_ch)
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(sum(self.groups[:k]), z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(z_dim, z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(z_dim, self.groups[k] * 2, kernel_size=5, stride=1),
            )
            for k in range(1, len(self.groups))
        }

        # Spatial context (g_sp)
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # Parameter aggregation
        param_aggregation = [
            sequential_channel_ramp(
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + z_dim * 2,
                self.groups[k] * 2,
                min_ch=z_dim * 2,
                num_layers=3,
                interp="linear",
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="noise"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        self.y_lc = ChannelGroupsLatentCodecContiguous(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        
        self.hyper_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=h_a,
            h_s=h_s,
            quantizer="noise",
        )

    def _apply_vit_blocks_spatial(self, x, blocks):
        """Apply ViT blocks to spatial features by temporarily flattening to sequence.
        
        Args:
            x: (B, C, H, W) spatial tensor
            blocks: nn.Sequential of ViT Blocks
        Returns:
            (B, C, H, W) spatial tensor
        """
        B, C, H, W = x.shape
        # Flatten to sequence: (B, H*W, C)
        x_seq = rearrange(x, "B C H W -> B (H W) C")
        x_seq = blocks(x_seq)
        # Reshape back to spatial
        x = rearrange(x_seq, "B (H W) C -> B C H W", H=H, W=W)
        return x

    def forward(self, h, qp=0, **kwargs):
        """Forward pass for patch-only spatial feature compression.

        Args:
            h (torch.Tensor): Spatial feature tensor of shape ``(B, h_dim, H, W)``.
            qp (int): Quantization parameter index in ``[0, 64]``.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with keys:
                - ``"h_hat"`` (torch.Tensor): Reconstructed spatial features.
                - ``"likelihoods"`` (dict): Likelihoods for patch latents ``"y"`` 
                  and hyperprior latents ``"z"``.
        """
        enc_gain = self.q_scale_enc[qp : qp + 1, :, :, :]
        dec_gain = self.q_scale_dec[qp : qp + 1, :, :, :]
        

        # Pre-processing
        h = self._apply_vit_blocks_spatial(h, self.pre_vit_blocks)
        h = h * self.q_scale_pre[qp : qp + 1, :, :].transpose(1, 2).unsqueeze(-1)

        # Analysis transform
        y = self.f_a(h) * enc_gain
        
        y_pad = border_pad(y, self.z_down_factor)
        
        hyper_out = self.hyper_lc(y_pad)
        
        _, _, y_H, y_W = y.shape
        params = hyper_out["params"]
        y_out = self.y_lc(y_pad, params)
        
        
        y_hat = y_out["y_hat"] [:, :, :y_H, :y_W]* dec_gain
        
        h_hat = self.f_s(y_hat)
        
        h_hat = h_hat * self.q_scale_post[qp : qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        h_hat = self._apply_vit_blocks_spatial(h_hat, self.post_vit_blocks)

        return {
            "h_hat": h_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, qp=0, **kwargs):
        """Compress spatial features.

        Args:
            h (torch.Tensor): Spatial feature tensor ``(B, h_dim, H, W)``.
            qp (int): Quantization parameter index in ``[0, 64]``.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with keys:
                - ``"strings"`` (dict): Bitstreams for ``"y"`` and ``"z"``.
                - ``"pstate"`` (dict): Side information including shapes, padding, qp.
        """
        enc_gain = self.q_scale_enc[qp : qp + 1, :, :, :]

        # Pre-processing
        h = self._apply_vit_blocks_spatial(h, self.pre_vit_blocks)
        h = h * self.q_scale_pre[qp : qp + 1, :, :].transpose(1, 2).unsqueeze(-1)

        # Analysis
        y = self.f_a(h) * enc_gain
        
        y_pad = border_pad(y, self.z_down_factor)
        hyper_out = self.hyper_lc.compress(y_pad)
        
        _, _, y_H, y_W = y.shape
        #y_out = self.y_lc.compress(y, hyper_out["params"][:, :, :y_H, :y_W])
        y_out = self.y_lc.compress(y_pad, hyper_out["params"])

        return {
            "strings": {
                "y": y_out["strings"],
                "z": hyper_out["strings"],
            },
            "pstate": {
                "y_shape": y_out["shape"], 
                "z_shape": hyper_out["shape"],
                "y_pad": (y_H, y_W),
                "qp": qp,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        """Decompress bitstreams back to spatial features.

        Args:
            strings (dict): Bitstreams for ``"y"`` and ``"z"``.
            pstate (dict): Side information from :meth:`compress`.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with key:
                - ``"h_hat"`` (torch.Tensor): Reconstructed spatial features.
        """
        qp = pstate["qp"]
        dec_gain = self.q_scale_dec[qp : qp + 1, :, :, :]

        # Decompress latents
        y_H, y_W = pstate["y_pad"]
        hyper_out = self.hyper_lc.decompress(strings["z"], pstate["z_shape"])
        y_out = self.y_lc.decompress(
            strings["y"], pstate["y_shape"], hyper_out["params"]
        )
        y_hat = y_out["y_hat"][:, :, :y_H, :y_W] * dec_gain

        # Synthesis + post-processing
        h_hat = self.f_s(y_hat)
        h_hat = h_hat * self.q_scale_post[qp : qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        h_hat = self._apply_vit_blocks_spatial(h_hat, self.post_vit_blocks)

        return {"h_hat": h_hat}
    
                
class SimpleHyperpriorUDualGain(SimpleHyperprior):
    def __init__(self,
                h_dim=1152,
                y_dim=256,
                z_dim=48,
                groups=32,
                z_down_factor=2,
                y_down_factor=2,
                vit_layers=0,
                U_Transform="",      # (h_dim, h_dim) PCA变换基. tensor or path.
                mean_vector="",      # (h_dim, ) PCA均值向量. tensor or path.
                deduct_mean = True,
                pretrained_path = "",
                **kwargs,
    ):
        super().__init__(h_dim=h_dim, y_dim=y_dim, z_dim=z_dim, groups=groups, 
                         z_down_factor=z_down_factor, y_down_factor=y_down_factor, 
                         vit_layers=vit_layers, **kwargs)
        
        # new PCA scalings
        self.u_scale_pre = nn.Parameter(torch.ones((65, 1, h_dim)))
        self.u_scale_post = nn.Parameter(torch.ones((65, 1, h_dim)))
        
        
        
        if isinstance(U_Transform, str):
            U_Transform = np.load(U_Transform)
            U_Transform = torch.from_numpy(U_Transform).float()
        if isinstance(mean_vector, str):
            mean_vector = np.load(mean_vector)
            mean_vector = torch.from_numpy(mean_vector).float()
        
        self.register_buffer("U_Transform", U_Transform)
        self.register_buffer("mean_vector", mean_vector)
    
        self.deduct_mean = deduct_mean  
        
        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu",weights_only=False)["model"]
            self.load_state_dict(state_dict, strict=False)
            self.update(force=True,update_quantiles=True)
            
    def forward(self, h, token_res = None, qp=0,**kwargs):
        enc_gain = self.q_scale_enc[qp: qp + 1, :, :, :]
        dec_gain = self.q_scale_dec[qp: qp + 1, :, :, :]
        
        
        h = rearrange(
            h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1]
        )
        
        h_in = h.clone()
        
        #  == dual path scaling ==
        h1 = h * self.q_scale_pre[qp: qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        
        
        h_residual = h_in - self.mean_vector.view(1, -1, 1, 1)
        u = torch.einsum("dc,bchw->bdhw", self.U_Transform, h_residual)  # (B, C, H, W)
        
        u = u * self.u_scale_pre[qp: qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        
        h_residual = torch.einsum("cd,bdhw->bchw", self.U_Transform.T, u)
        h2 = h_residual + self.mean_vector.view(1, -1, 1, 1) if not self.deduct_mean else h_residual
        
        h = h1+h2
        
        # == == == == == == == ==
        
        y = self.f_a(h) * enc_gain
        
        y_pad = border_pad(y, self.z_down_factor)
        hyper_out = self.hyper_lc(y_pad)
        _, _, y_H, y_W = y.shape
        params = hyper_out["params"]
        y_out = self.y_lc(y_pad, params)
        y_hat = y_out["y_hat"][:, :, :y_H, :y_W] * dec_gain
        
        h_hat = self.f_s(y_hat)
        
        #  == dual path scaling == 
        
        h_hat_residual = h_hat - self.mean_vector.view(1, -1, 1, 1)
        u_hat = torch.einsum("dc,bchw->bdhw", self.U_Transform, h_hat_residual)
        
        u_hat = u_hat * self.u_scale_post[qp: qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        
        h_hat_residual = torch.einsum("cd,bdhw->bchw", self.U_Transform.T, u_hat)
        h_hat2 = h_hat_residual + self.mean_vector.view(1, -1, 1, 1) if not self.deduct_mean else h_hat_residual
        
        h_hat1 = h_hat * self.q_scale_post[qp: qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        
        h_hat = h_hat1+h_hat2
        
        # == == == == == == == == ==
        
        
        h_hat = rearrange(h_hat, "B C H W -> B (H W) C").contiguous()
        
        return {
            "h_hat": h_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, token_res=None, qp=0, **kwargs):
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
            
        #  == dual path scaling ==
        h_in = h.clone()
        h1 = h * self.q_scale_pre[qp: qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        
        h_residual = h_in - self.mean_vector.view(1, -1, 1, 1)
        u = torch.einsum("dc,bchw->bdhw", self.U_Transform, h_residual)
        u = u * self.u_scale_pre[qp: qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        h_residual = torch.einsum("cd,bdhw->bchw", self.U_Transform.T, u)
        h2 = h_residual + self.mean_vector.view(1, -1, 1, 1) if not self.deduct_mean else h_residual
        
        h = h1+h2
        # == == == == == == == == ==
        
        y = self.f_a(h) * self.q_scale_enc[qp: qp + 1, :, :, :]
        y_pad = border_pad(y, self.z_down_factor)
        hyper_out = self.hyper_lc.compress(y_pad)
        
        _, _, y_H, y_W = y.shape
        y_out = self.y_lc.compress(y_pad, hyper_out["params"])
        
        return {
            "strings": {
                "y": y_out["strings"],
                "z": hyper_out["strings"],
            },
            "pstate": {
                "y_shape": y_out["shape"], 
                "z_shape": hyper_out["shape"],
                "y_pad": (y_H, y_W),
                "qp": qp,
                "token_res": token_res,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        qp = pstate["qp"]
        y_H, y_W = pstate["y_pad"]
        
        hyper_out = self.hyper_lc.decompress(strings["z"], pstate["z_shape"])
        y_out = self.y_lc.decompress(strings["y"], pstate["y_shape"], hyper_out["params"])
        y_hat = y_out["y_hat"][:, :, :y_H, :y_W] * self.q_scale_dec[qp: qp + 1, :, :, :]
        
        h_hat = self.f_s(y_hat)
        
        # == dual path scaling ==
        h_hat_residual = h_hat - self.mean_vector.view(1, -1, 1, 1)
        u_hat = torch.einsum("dc,bchw->bdhw", self.U_Transform, h_hat_residual)
        u_hat = u_hat * self.u_scale_post[qp: qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        h_hat_residual = torch.einsum("cd,bdhw->bchw", self.U_Transform.T, u_hat)
        h_hat2 = h_hat_residual + self.mean_vector.view(1, -1, 1, 1) if not self.deduct_mean else h_hat_residual
        
        h_hat1 = h_hat * self.q_scale_post[qp: qp + 1, :, :].transpose(1, 2).unsqueeze(-1)
        h_hat = h_hat1 + h_hat2
        
        # == == == == == == == == ==
        
        h_hat = rearrange(h_hat, "B C H W -> B (H W) C").contiguous()
        
        return {"h_hat": h_hat}
import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from cofai.latent_codecs.hyperprior import (
    HyperLatentCodecWithCtx,
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


@register_module("ChannelGroupsLatentCodecContiguous")
class ChannelGroupsLatentCodecContiguous(ChannelGroupsLatentCodec):
    # monkey patch to make the ch ctx params consistent within compress and decompress
    def merge_y(self, *args):
        return torch.cat(args, dim=1).contiguous()

    def merge_params(self, *args):
        return torch.cat(args, dim=1).contiguous()


@register_model("VitUnionLatentCodec")
class VitUnionLatentCodec(CompressionModel):
    """Vit-based latent codec with joint modeling of cls token and patch tokens.

    This codec takes ViT features as input and compresses the 2D patch tokens
    using a hyperprior + space-channel context model (SCCTX) as in [He2022].
    It reconstructs the ViT feature map and re-injects learned register tokens
    before passing through transformer blocks.

    Args:
        h_dim (int): Channel dimension of ViT features.
        y_dim (int): Channel dimension of primary latent representation ``y``.
        z_dim (int): Channel dimension of hyperprior latent representation ``z``.
        groups (int or list[int]): Channel groups for channel-wise context modeling.
            If int, the channels are evenly split; if list, must sum to ``y_dim``.
        num_prefix_tokens (int): Number of prefix/register tokens in the ViT feature.
        **kwargs (dict): Extra keyword arguments for compatibility (unused).
    """

    def __init__(
        self,
        h_dim=384,
        y_dim=256,
        z_dim=192,
        groups=16,
        num_prefix_tokens=1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.num_prefix_tokens = num_prefix_tokens
        self.post_reg_tokens = nn.Parameter(
            torch.zeros(num_prefix_tokens, h_dim), requires_grad=True
        )
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        h_a = nn.Sequential(
            conv(y_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim * 3 // 2, z_dim * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
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

        # In [He2022], this is labeled "g_sp^(k)".
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

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
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

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # Channel groups with space-channel context model (SCCTX):
        self.y_lc = ChannelGroupsLatentCodecContiguous(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        self.hyper_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=h_a,
            h_s=h_s,
            quantizer="ste",
        )

    def forward(self, h, token_res, **kwargs):
        """Forward pass for end-to-end rate–distortion training.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)`` containing
                prefix tokens and patch tokens.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)`` such that
                ``L = num_prefix_tokens + H * W``.
            **kwargs (dict): Unused keyword arguments for API compatibility.

        Returns:
            out (dict): A dictionary with keys:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features of shape
                  ``(B, L, C)``.
                - ``\"likelihoods\"`` (dict): Per-latent likelihoods with keys
                  ``\"y\"`` and ``\"z\"``.
        """
        B = h.shape[0]
        h = self.pre_vit_blocks(h)[:, self.num_prefix_tokens :].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        y = self.f_a(h)
        hyper_out = self.hyper_lc(y)
        y_out = self.y_lc(y, hyper_out["params"])
        y_hat = y_out["y_hat"]

        _h_hat = self.f_s(y_hat)
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(B, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_hat": h_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, token_res, **kwargs):
        """Compress ViT features into entropy-coded bitstreams.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.
            **kwargs (dict): Unused keyword arguments for API compatibility.

        Returns:
            out (dict): A dictionary with keys:

                - ``\"strings\"`` (dict): Entropy-coded bitstreams for ``\"y\"`` and
                  ``\"z\"``.
                - ``\"pstate\"`` (dict): Side information needed for decoding, including
                  shapes and token resolution.
        """
        h = self.pre_vit_blocks(h)[:, self.num_prefix_tokens :].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        y = self.f_a(h)
        # x --16-> h --2-> y --4-> z
        # if pad 32 for y, y is not compatible with checkerboard codec
        # so we pad 64 for y, then only need to pad 2 for z
        y_pad = border_pad(y, 2)
        hyper_out = self.hyper_lc.compress(y_pad)
        _, _, y_H, y_W = y.shape
        y_out = self.y_lc.compress(y, hyper_out["params"][:, :, :y_H, :y_W])

        return {
            "strings": {"y": y_out["strings"], "z": hyper_out["strings"]},
            "pstate": {
                "y_shape": y_out["shape"],
                "z_shape": hyper_out["shape"],
                "y_pad": (y_H, y_W),
                "token_res": token_res,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        """Decompress entropy-coded bitstreams back to ViT features.

        Args:
            strings (dict): Bitstreams produced by :meth:`compress`, with keys ``\"y\"`` and ``\"z\"``.
            pstate (dict): Side information produced by :meth:`compress`, including shapes and token resolution.
            **kwargs (dict): Unused keyword arguments for API compatibility.

        Returns:
            out (dict): A dictionary with key:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
        """
        y_strings = strings["y"]
        z_strings = strings["z"]
        y_shape = pstate["y_shape"]
        z_shape = pstate["z_shape"]
        hyper_out = self.hyper_lc.decompress(z_strings, z_shape)
        y_H, y_W = pstate["y_pad"]
        y_out = self.y_lc.decompress(
            y_strings, y_shape, hyper_out["params"][:, :, :y_H, :y_W]
        )
        h_hat = self.f_s(y_out["y_hat"])
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(1, -1, -1), _h_hat], dim=1
        ).contiguous()
        h_hat = self.post_vit_blocks(_h_hat)
        return {"h_hat": h_hat}


@register_model("VbrVitUnionLatentCodec")
class VbrVitUnionLatentCodec(CompressionModel):
    """Vit-based union latent codec with variable bit-rate control.

    This variant introduces learnable per-quantization-parameter scaling factors
    to control the bitrate–distortion trade-off, following the strategy in
    DCVC. It scales the latents ``y`` before and after entropy coding.

    Args:
        h_dim (int): Channel dimension of ViT features.
        y_dim (int): Channel dimension of primary latent representation ``y``.
        z_dim (int): Channel dimension of hyperprior latent representation ``z``.
        groups (int or list[int]): Channel groups for channel-wise context modeling.
        num_prefix_tokens (int): Number of prefix/register tokens in the ViT feature.
        **kwargs (dict): Extra keyword arguments for compatibility (unused).
    """

    def __init__(
        self,
        h_dim=384,
        y_dim=256,
        z_dim=192,
        groups=16,
        num_prefix_tokens=1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.q_scale_enc = nn.Parameter(torch.ones((65, y_dim, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((65, y_dim, 1, 1)))
        # https://github.com/microsoft/DCVC/blob/main/src/models/image_model.py

        self.num_prefix_tokens = num_prefix_tokens
        self.post_reg_tokens = nn.Parameter(
            torch.zeros(num_prefix_tokens, h_dim), requires_grad=True
        )
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        h_a = nn.Sequential(
            conv(y_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim * 3 // 2, z_dim * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
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

        # In [He2022], this is labeled "g_sp^(k)".
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

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
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

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # Channel groups with space-channel context model (SCCTX):
        self.y_lc = ChannelGroupsLatentCodecContiguous(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        self.hyper_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=h_a,
            h_s=h_s,
            quantizer="ste",
        )

    def forward(self, h, token_res, qp=0):
        """Forward pass for rate–distortion training with quantization parameter.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.
            qp (int): Quantization parameter index in ``[0, 64]`` controlling the
                bitrate–distortion trade-off.

        Returns:
            out (dict): A dictionary with keys:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
                - ``\"likelihoods\"`` (dict): Likelihoods for ``\"y\"`` and ``\"z\"``.
        """
        enc_gain = self.q_scale_enc[qp : qp + 1, :, :, :]
        dec_gain = self.q_scale_dec[qp : qp + 1, :, :, :]

        B = h.shape[0]
        h = self.pre_vit_blocks(h)[:, self.num_prefix_tokens :].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        y = self.f_a(h) * enc_gain
        hyper_out = self.hyper_lc(y)
        y_out = self.y_lc(y, hyper_out["params"])
        y_hat = y_out["y_hat"] * dec_gain

        _h_hat = self.f_s(y_hat)
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(B, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_hat": h_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, token_res, qp=0, **kwargs):
        """Compress ViT features with a given quantization parameter.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.
            qp (int): Quantization parameter index in ``[0, 64]``.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with keys:

                - ``\"strings\"`` (dict): Bitstreams for ``\"y\"`` and ``\"z\"``.
                - ``\"pstate\"`` (dict): Side information including shapes, padding,
                  token resolution and ``qp``.
        """
        enc_gain = self.q_scale_enc[qp : qp + 1, :, :, :]
        h = self.pre_vit_blocks(h)[:, self.num_prefix_tokens :].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        y = self.f_a(h) * enc_gain
        # x --16-> h --2-> y --4-> z
        # if pad 32 for y, y is not compatible with checkerboard codec
        # so we pad 64 for y, then only need to pad 2 for z
        y_pad = border_pad(y, 2)
        hyper_out = self.hyper_lc.compress(y_pad)
        _, _, y_H, y_W = y.shape
        y_out = self.y_lc.compress(y, hyper_out["params"][:, :, :y_H, :y_W])

        return {
            "strings": {"y": y_out["strings"], "z": hyper_out["strings"]},
            "pstate": {
                "y_shape": y_out["shape"],
                "z_shape": hyper_out["shape"],
                "y_pad": (y_H, y_W),
                "token_res": token_res,
                "qp": qp,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        """Decompress bitstreams produced by :meth:`compress`.

        Args:
            strings (dict): Bitstreams with keys ``\"y\"`` and ``\"z\"``.
            pstate (dict): Side information, including shapes, padding and ``qp``.
            **kwargs (dict): Unused keyword arguments for API compatibility.

        Returns:
            out (dict): A dictionary with key:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
        """
        y_strings_ = strings["y"]
        z_strings_ = strings["z"]
        y_shape = pstate["y_shape"]
        z_shape = pstate["z_shape"]
        qp = pstate["qp"]

        hyper_out = self.hyper_lc.decompress(z_strings_, z_shape)
        dec_gain = self.q_scale_dec[qp : qp + 1, :, :, :]
        y_H, y_W = pstate["y_pad"]
        y_out = self.y_lc.decompress(
            y_strings_, y_shape, hyper_out["params"][:, :, :y_H, :y_W]
        )
        y_hat = y_out["y_hat"] * dec_gain
        h_hat = self.f_s(y_hat)
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(_h_hat.shape[0], -1, -1), _h_hat], dim=1
        ).contiguous()
        h_hat = self.post_vit_blocks(_h_hat)
        return {"h_hat": h_hat}


@register_model("VitSeparateLatentCodec")
class VitSeparateLatentCodec(CompressionModel):
    """Vit latent codec with separate modeling of class and patch tokens.

    This codec encodes class (prefix) tokens and patch tokens with different
    hyperprior models. Class tokens are compressed using a dedicated hyperprior
    codec, while patch tokens are compressed with a hyperprior + SCCTX model.

    Args:
        h_dim (int): Channel dimension of ViT features.
        y_dim (int): Channel dimension of primary latent representation ``y``.
        z_dim (int): Channel dimension of hyperprior latent representation ``z``.
        groups (int or list[int]): Channel groups for channel-wise context modeling.
        num_prefix_tokens (int): Number of prefix/register tokens in the ViT feature.
        **kwargs: Extra keyword arguments for compatibility (unused).
    """

    def __init__(
        self,
        h_dim=384,
        y_dim=256,
        z_dim=192,
        groups=16,
        num_prefix_tokens=5,
        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_prefix_tokens = num_prefix_tokens
        # self.post_reg_tokens = nn.Parameter(torch.zeros(1, h_dim), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        h_a = nn.Sequential(
            conv(y_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim * 3 // 2, z_dim * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
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

        # In [He2022], this is labeled "g_sp^(k)".
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

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
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

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # Channel groups with space-channel context model (SCCTX):
        self.y_lc = ChannelGroupsLatentCodecContiguous(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        self.hyper_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=h_a,
            h_s=h_s,
            quantizer="ste",
        )
        self.cls_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=nn.Conv2d(h_dim, z_dim, kernel_size=1),
            h_s=nn.Conv2d(z_dim, h_dim, kernel_size=1),
            quantizer="ste",
        )

    def forward(self, h, token_res, **kwargs):
        """Forward pass for separate class/patch token compression.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with keys:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
                - ``\"likelihoods\"`` (dict): Likelihoods for class ``\"c\"``, patch
                  latents ``\"y\"`` and hyperprior latents ``\"z\"``.
        """
        h = self.pre_vit_blocks(h)

        h_cls = h[:, 0 : self.num_prefix_tokens]
        h_cls = rearrange(h_cls, "B L C -> B C L 1")
        cls_out = self.cls_lc(h_cls)
        h_cls_hat = cls_out["params"]
        h_cls_hat = rearrange(h_cls_hat, "B C L 1 -> B L C")

        h_patch = h[:, self.num_prefix_tokens :].contiguous()
        h_patch = rearrange(
            h_patch, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1]
        )
        y = self.f_a(h_patch)
        y_pad = border_pad(y, 2)
        hyper_out = self.hyper_lc(y_pad)
        _, _, y_pad_H, y_pad_W = y_pad.shape
        hyper_params = hyper_out["params"][:, :, :y_pad_H, :y_pad_W]
        y_out = self.y_lc(y_pad, hyper_params)
        y_hat = y_out["y_hat"]
        h_patch_hat = self.f_s(y_hat)
        h_patch_hat = h_patch_hat[:, :, : token_res[0], : token_res[1]]
        h_patch_hat = rearrange(h_patch_hat, "B C H W -> B (H W) C")

        h_hat = torch.cat([h_cls_hat, h_patch_hat], dim=1)
        h_hat = self.post_vit_blocks(h_hat)

        return {
            "h_hat": h_hat,
            "likelihoods": {
                "c": cls_out["likelihoods"]["z"],
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, token_res, **kwargs):
        """Compress ViT features with separate class and patch codecs.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with keys:

                - ``\"strings\"`` (dict): Bitstreams for class ``\"cls\"``, patch
                  ``\"y\"`` and hyperprior ``\"z\"``.
                - ``\"pstate\"`` (dict): Side information including shapes and padding.
        """
        h = self.pre_vit_blocks(h)

        h_cls = h[:, 0 : self.num_prefix_tokens]
        h_cls = rearrange(h_cls, "B L C -> B C L 1")
        cls_out = self.cls_lc.compress(h_cls)

        h_patch = h[:, self.num_prefix_tokens :].contiguous()
        h_patch = rearrange(
            h_patch, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1]
        )
        y = self.f_a(h_patch)
        # x --16-> h --2-> y --4-> z
        # if pad 32 for y, y is not compatible with checkerboard codec
        # so we pad 64 for y, then only need to pad 2 for z
        y_pad = border_pad(y, 2)
        hyper_out = self.hyper_lc.compress(y_pad)
        _, _, y_H, y_W = y.shape
        _, _, y_pad_H, y_pad_W = y_pad.shape
        hyper_params = hyper_out["params"][:, :, :y_pad_H, :y_pad_W]
        y_out = self.y_lc.compress(y_pad, hyper_params)

        return {
            "strings": {
                "cls": cls_out["strings"],
                "y": y_out["strings"],
                "z": hyper_out["strings"],
            },
            "pstate": {
                "cls_shape": cls_out["shape"],
                "y_shape": y_out["shape"],
                "z_shape": hyper_out["shape"],
                "y_pad": (y_H, y_W),
                "y_padded_shape": (y_pad_H, y_pad_W),
                "token_res": token_res,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        """Decompress bitstreams back to ViT features.

        Args:
            strings (dict): Bitstreams for ``\"cls\"``, ``\"y\"`` and ``\"z\"``.
            pstate (dict): Side information produced by :meth:`compress`.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with key:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
        """
        cls_out = self.cls_lc.decompress(strings["cls"], pstate["cls_shape"])
        h_cls_hat = cls_out["params"]
        h_cls_hat = rearrange(h_cls_hat, "B C L 1 -> B L C")

        y_H, y_W = pstate["y_pad"]
        hyper_out = self.hyper_lc.decompress(strings["z"], pstate["z_shape"])
        y_pad_H, y_pad_W = pstate.get("y_padded_shape", (y_H, y_W))
        hyper_params = hyper_out["params"][:, :, : int(y_pad_H), : int(y_pad_W)]
        y_out = self.y_lc.decompress(strings["y"], pstate["y_shape"], hyper_params)
        h_patch_hat = self.f_s(y_out["y_hat"])
        token_res = pstate.get("token_res")
        if token_res is not None:
            h_patch_hat = h_patch_hat[:, :, : int(token_res[0]), : int(token_res[1])]
        h_patch_hat = rearrange(h_patch_hat, "B C H W -> B (H W) C")

        h_hat = torch.cat([h_cls_hat, h_patch_hat], dim=1)
        h_hat = self.post_vit_blocks(h_hat)
        return {"h_hat": h_hat}



@register_model("VbrVitSeparateLatentCodec")
class VbrVitSeparateLatentCodec(CompressionModel):
    """Vit latent codec with separate cls/patch modeling and variable bit-rate control.

    This variant extends ``VitSeparateLatentCodec`` by introducing learnable
    per-QP scaling factors on patch latents ``y`` (pre/post entropy coding),
    following the same VBR strategy used in ``VbrVitUnionLatentCodec``.
    """

    def __init__(
        self,
        h_dim=384,
        y_dim=256,
        z_dim=192,
        groups=16,
        num_prefix_tokens=1,
        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.num_prefix_tokens = num_prefix_tokens

        # VBR learnable scales (QP in [0, 64])
        #cls+reg token vbr
        self.q_scale_cls_enc = nn.Parameter(torch.ones((65, h_dim, 1, 1)))
        self.q_scale_cls_dec = nn.Parameter(torch.ones((65, h_dim, 1, 1)))
        #patch token vbr
        self.q_scale_enc = nn.Parameter(torch.ones((65, y_dim, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((65, y_dim, 1, 1)))

        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        h_a = nn.Sequential(
            conv(y_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim * 3 // 2, z_dim * 2, kernel_size=3, stride=1),
        )

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
                latent_codec={"y": GaussianConditionalLatentCodec(quantizer="ste")},
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
            quantizer="ste",
        )
        self.cls_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=nn.Conv2d(h_dim, z_dim, kernel_size=1),
            h_s=nn.Conv2d(z_dim, h_dim, kernel_size=1),
            quantizer="ste",
        )

    def forward(self, h, token_res, qp=0, **kwargs):
        h = self.pre_vit_blocks(h)

        # cls branch with VBR scaling
        cls_enc_gain = self.q_scale_cls_enc[qp : qp + 1, :, :, :]
        cls_dec_gain = self.q_scale_cls_dec[qp : qp + 1, :, :, :]
        
        h_cls = h[:, 0 : self.num_prefix_tokens]
        h_cls = rearrange(h_cls, "B L C -> B C L 1")
        h_cls = h_cls * cls_enc_gain
        cls_out = self.cls_lc(h_cls)
        h_cls_hat = rearrange(cls_out["params"] * cls_dec_gain, "B C L 1 -> B L C")

        # patch branch with VBR scaling
        enc_gain = self.q_scale_enc[qp : qp + 1, :, :, :]
        dec_gain = self.q_scale_dec[qp : qp + 1, :, :, :]

        h_patch = h[:, self.num_prefix_tokens :].contiguous()
        h_patch = rearrange(
            h_patch, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1]
        )
        y = self.f_a(h_patch) * enc_gain
        hyper_out = self.hyper_lc(y)
        y_out = self.y_lc(y, hyper_out["params"])
        y_hat = y_out["y_hat"] * dec_gain

        h_patch_hat = self.f_s(y_hat)
        h_patch_hat = rearrange(h_patch_hat, "B C H W -> B (H W) C")

        h_hat = torch.cat([h_cls_hat, h_patch_hat], dim=1)
        h_hat = self.post_vit_blocks(h_hat)

        return {
            "h_hat": h_hat,
            "likelihoods": {
                "c": cls_out["likelihoods"]["z"],
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, token_res, qp=0, **kwargs):
        h = self.pre_vit_blocks(h)

        # cls branch with VBR scaling
        cls_enc_gain = self.q_scale_cls_enc[qp : qp + 1, :, :, :]
        
        h_cls = h[:, 0 : self.num_prefix_tokens]
        h_cls = rearrange(h_cls, "B L C -> B C L 1")
        h_cls = h_cls * cls_enc_gain
        cls_out = self.cls_lc.compress(h_cls)

        # patch branch with VBR scaling
        enc_gain = self.q_scale_enc[qp : qp + 1, :, :, :]

        h_patch = h[:, self.num_prefix_tokens :].contiguous()
        h_patch = rearrange(
            h_patch, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1]
        )
        y = self.f_a(h_patch) * enc_gain

        y_pad = border_pad(y, 2)
        hyper_out = self.hyper_lc.compress(y_pad)
        _, _, y_H, y_W = y.shape
        y_out = self.y_lc.compress(y, hyper_out["params"][:, :, :y_H, :y_W])

        return {
            "strings": {
                "cls": cls_out["strings"],
                "y": y_out["strings"],
                "z": hyper_out["strings"],
            },
            "pstate": {
                "cls_shape": cls_out["shape"],
                "y_shape": y_out["shape"],
                "z_shape": hyper_out["shape"],
                "y_pad": (y_H, y_W),
                "qp": qp,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        # cls branch
        qp = pstate["qp"]
        cls_dec_gain = self.q_scale_cls_dec[qp : qp + 1, :, :, :]
        cls_out = self.cls_lc.decompress(strings["cls"], pstate["cls_shape"])
        h_cls_hat = rearrange(cls_out["params"] * cls_dec_gain, "B C L 1 -> B L C")

        # patch branch with VBR scaling
        qp = pstate["qp"]
        dec_gain = self.q_scale_dec[qp : qp + 1, :, :, :]

        y_H, y_W = pstate["y_pad"]
        hyper_out = self.hyper_lc.decompress(strings["z"], pstate["z_shape"])
        y_out = self.y_lc.decompress(
            strings["y"], pstate["y_shape"], hyper_out["params"][:, :, :y_H, :y_W]
        )
        y_hat = y_out["y_hat"] * dec_gain

        h_patch_hat = self.f_s(y_hat)
        h_patch_hat = rearrange(h_patch_hat, "B C H W -> B (H W) C")

        h_hat = torch.cat([h_cls_hat, h_patch_hat], dim=1)
        h_hat = self.post_vit_blocks(h_hat)
        return {"h_hat": h_hat}

class HyperEncoderWithCtx(nn.Module):
    def __init__(self, z_dim, y_dim, ctx_dim):
        super().__init__()
        self.h_a = nn.Sequential(
            conv(y_dim + ctx_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
        )

    def forward(self, x, ctx):
        return self.h_a(torch.cat([x, ctx], dim=1))


class HyperDecoderWithCtx(nn.Module):
    def __init__(self, z_dim, y_dim, ctx_dim):
        super().__init__()
        self.h_s = nn.Sequential(
            deconv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim * 3 // 2, z_dim * 2, kernel_size=3, stride=1),
        )
        self.fusion = nn.Sequential(
            conv(z_dim * 2 + ctx_dim, z_dim * 2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim * 2, z_dim * 2, kernel_size=3, stride=1),
        )

    def forward(self, x, ctx):
        h = self.h_s(x)
        h = self.fusion(torch.cat([h, ctx], dim=1))
        return h


@register_model("VitUnionLatentCodecWithCtx")
class VitUnionLatentCodecWithCtx(CompressionModel):
    """Vit union latent codec conditioned on an external context feature map.

    This codec jointly compresses ViT patch tokens and an additional context
    feature map. The context is injected into both the analysis and synthesis
    transforms as well as the hyperprior pathway.

    Args:
        h_dim (int): Channel dimension of ViT features.
        y_dim (int): Channel dimension of primary latent representation ``y``.
        z_dim (int): Channel dimension of hyperprior latent representation ``z``.
        ctx_dim (int): Channel dimension of the external context feature map.
        groups (int or list[int]): Channel groups for channel-wise context modeling.
        **kwargs (dict): Extra keyword arguments for compatibility (unused).
    """

    def __init__(
        self,
        h_dim=384,
        y_dim=256,
        z_dim=192,
        ctx_dim=256,
        groups=16,
        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.post_reg_tokens = nn.Parameter(torch.zeros(1, h_dim), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.cond_enc = nn.Sequential(
            conv(h_dim + ctx_dim, h_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(h_dim, h_dim, kernel_size=3, stride=1),
        )
        # self.cond_dec1 = nn.Sequential(
        #     deconv(dim, dim, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     deconv(dim, dim + ctx_dim, kernel_size=3, stride=1),
        # )

        self.cond_dec = nn.Sequential(
            conv(h_dim + ctx_dim, h_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(h_dim, h_dim, kernel_size=3, stride=1),
        )
        self.f_ctx_down = conv(ctx_dim, ctx_dim, kernel_size=3, stride=2)

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        h_a = HyperEncoderWithCtx(z_dim, y_dim, ctx_dim)
        h_s = HyperDecoderWithCtx(z_dim, y_dim, ctx_dim)

        # In [He2022], this is labeled "g_ch^(k)".
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

        # In [He2022], this is labeled "g_sp^(k)".
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

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
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

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # Channel groups with space-channel context model (SCCTX):
        self.y_lc = ChannelGroupsLatentCodecContiguous(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        self.hyper_lc = HyperLatentCodecWithCtx(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=h_a,
            h_s=h_s,
            quantizer="ste",
        )

    def forward(self, h, ctx, token_res):
        """Forward pass with context-conditioned hyperprior.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            ctx (torch.Tensor): Context feature map of shape ``(B, ctx_dim, H, W)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.

        Returns:
            out (dict): A dictionary with keys:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
                - ``\"h_hat_share\"`` (torch.Tensor): Shared feature map before
                  context decoding.
                - ``\"likelihoods\"`` (dict): Likelihoods for ``\"y\"`` and ``\"z\"``.
        """
        B = h.shape[0]
        h = self.pre_vit_blocks(h)[:, 1:].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        h_share = self.cond_enc(torch.cat([h, ctx], dim=1))
        y = self.f_a(h_share)

        ctx_down = self.f_ctx_down(ctx)
        hyper_out = self.hyper_lc(y, ctx_down)
        y_out = self.y_lc(y, hyper_out["params"])
        y_hat = y_out["y_hat"]

        h_hat_share = self.f_s(y_hat)
        _h_hat = self.cond_dec(torch.cat([h_hat_share, ctx], dim=1))
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(B, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_hat": h_hat,
            "h_hat_share": h_hat_share,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, ctx, token_res):
        """Compress ViT features conditioned on a context feature map.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            ctx (torch.Tensor): Context feature map of shape ``(B, ctx_dim, H, W)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.

        Returns:
            out (dict): A dictionary with keys:

                - ``\"strings\"`` (dict): Bitstreams for ``\"y\"`` and ``\"z\"``.
                - ``\"pstate\"`` (dict): Side information with shapes and token
                  resolution.
        """
        h = self.pre_vit_blocks(h)[:, 1:].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        h_share = self.cond_enc(torch.cat([h, ctx], dim=1))
        y = self.f_a(h_share)

        ctx_down = self.f_ctx_down(ctx)

        hyper_out = self.hyper_lc.compress(y, ctx_down)
        y_out = self.y_lc.compress(y, hyper_out["params"])

        return {
            "strings": {"y": y_out["strings"], "z": hyper_out["strings"]},
            "pstate": {
                "y_shape": y_out["shape"],
                "z_shape": hyper_out["shape"],
                "token_res": token_res,
            },
            # "y_hat": y_out["y_hat"],
        }

    def decompress(self, strings, pstate, ctx, **kwargs):
        """Decompress context-conditioned bitstreams back to ViT features.

        Args:
            strings (dict): Bitstreams with keys ``\"y\"`` and ``\"z\"``.
            pstate (dict): Side information produced by :meth:`compress`.
            ctx (torch.Tensor): Context feature map used also at decoding time.
            **kwargs (dict): Unused keyword arguments for API compatibility.

        Returns:
            out (dict): A dictionary with keys:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
                - ``\"h_hat_share\"`` (torch.Tensor): Shared feature map before
                  context decoding.
        """
        y_strings_ = strings["y"]
        z_strings_ = strings["z"]
        # assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        ctx_down = self.f_ctx_down(ctx)
        hyper_out = self.hyper_lc.decompress(z_strings_, pstate["z_shape"], ctx_down)
        y_out = self.y_lc.decompress(y_strings_, pstate["y_shape"], hyper_out["params"])
        y_hat = y_out["y_hat"]

        h_hat_share = self.f_s(y_hat)
        _h_hat = self.cond_dec(torch.cat([h_hat_share, ctx], dim=1))
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(1, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)
        return {"h_hat": h_hat, "h_hat_share": h_hat_share}


@register_model("VitUnionLatentCodecCtxAsHyper")
class VitUnionLatentCodecCtxAsHyper(CompressionModel):
    """Vit union latent codec using context as hyperprior parameters.

    Instead of learning hyperprior latents ``z``, this variant derives the
    entropy model parameters directly from the external context feature map.

    Args:
        h_dim (int): Channel dimension of ViT features.
        y_dim (int): Channel dimension of primary latent representation ``y``.
        z_dim (int): Channel dimension used in context-to-parameter mapping.
        ctx_dim (int): Channel dimension of the external context feature map.
        groups (int or list[int]): Channel groups for channel-wise context modeling.
        **kwargs (dict): Extra keyword arguments for compatibility (unused).
    """

    def __init__(
        self,
        h_dim=384,
        y_dim=256,
        z_dim=192,
        ctx_dim=256,
        groups=16,
        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.post_reg_tokens = nn.Parameter(torch.zeros(1, h_dim), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.cond_enc = nn.Sequential(
            conv(h_dim + ctx_dim, h_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(h_dim, h_dim, kernel_size=3, stride=1),
        )
        # self.cond_dec1 = nn.Sequential(
        #     deconv(dim, dim, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     deconv(dim, dim + ctx_dim, kernel_size=3, stride=1),
        # )

        self.cond_dec = nn.Sequential(
            conv(h_dim + ctx_dim, h_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(h_dim, h_dim, kernel_size=3, stride=1),
        )
        self.f_ctx_params = nn.Sequential(
            conv(ctx_dim, z_dim * 2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim * 2, z_dim * 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim * 2, z_dim * 2, kernel_size=3, stride=1),
        )

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
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

        # In [He2022], this is labeled "g_sp^(k)".
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

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
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

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # Channel groups with space-channel context model (SCCTX):
        self.y_lc = ChannelGroupsLatentCodecContiguous(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        self.useless_lc = EntropyBottleneck(z_dim)

    def forward(self, h, ctx, token_res):
        """Forward pass using context-derived entropy model parameters.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            ctx (torch.Tensor): Context feature map of shape ``(B, ctx_dim, H, W)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.

        Returns:
            dict: A dictionary with keys:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
                - ``\"h_hat_share\"`` (torch.Tensor): Shared feature map before
                  context decoding.
                - ``\"likelihoods\"`` (dict): Likelihoods for ``\"y\"``.
        """
        B = h.shape[0]
        h = self.pre_vit_blocks(h)[:, 1:].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        h_share = self.cond_enc(torch.cat([h, ctx], dim=1))
        y = self.f_a(h_share)

        ctx_params = self.f_ctx_params(ctx)
        y_out = self.y_lc(y, ctx_params)
        y_hat = y_out["y_hat"]

        h_hat_share = self.f_s(y_hat)
        _h_hat = self.cond_dec(torch.cat([h_hat_share, ctx], dim=1))
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(B, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_hat": h_hat,
            "h_hat_share": h_hat_share,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
            },
        }

    def compress(self, h, ctx, token_res):
        """Compress ViT features using context-derived entropy parameters.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            ctx (torch.Tensor): Context feature map of shape ``(B, ctx_dim, H, W)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.

        Returns:
            dict: A dictionary with keys:

                - ``\"strings\"`` (dict): Bitstreams for ``\"y\"``.
                - ``\"pstate\"`` (dict): Side information including shapes.
        """
        h = self.pre_vit_blocks(h)[:, 1:].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        h_share = self.cond_enc(torch.cat([h, ctx], dim=1))
        y = self.f_a(h_share)

        ctx_params = self.f_ctx_params(ctx)

        hyper_out = self.hyper_lc.compress(y, ctx_params)
        y_out = self.y_lc.compress(y, hyper_out["params"])

        return {
            "strings": {"y": y_out["strings"]},
            "pstate": {"y_shape": y_out["shape"]},
            # "y_hat": y_out["y_hat"],
        }

    def decompress(self, strings, pstate, ctx, **kwargs):
        """Decompress bitstreams back to ViT features using context as hyperprior.

        Args:
            strings (dict): Bitstreams with key ``\"y\"``.
            pstate (dict): Side information produced by :meth:`compress`.
            ctx (torch.Tensor): Context feature map used to reconstruct entropy
                model parameters.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with keys:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
                - ``\"h_hat_share\"`` (torch.Tensor): Shared feature map before
                  context decoding.
        """
        y_strings_ = strings["y"]
        ctx_params = self.f_ctx_params(ctx)
        y_out = self.y_lc.decompress(y_strings_, pstate["y_shape"], ctx_params)
        y_hat = y_out["y_hat"]

        h_hat_share = self.f_s(y_hat)
        _h_hat = self.cond_dec(torch.cat([h_hat_share, ctx], dim=1))
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(1, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)
        return {"h_hat": h_hat, "h_hat_share": h_hat_share}

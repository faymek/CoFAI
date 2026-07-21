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


@register_module("ChannelGroupsLatentCodecContiguous")
class ChannelGroupsLatentCodecContiguous(ChannelGroupsLatentCodec):
    # monkey patch to make the ch ctx params consistent within compress and decompress
    def merge_y(self, *args):
        return torch.cat(args, dim=1).contiguous()

    def merge_params(self, *args):
        return torch.cat(args, dim=1).contiguous()


@register_model("VisualTokenCodec")
class VisualTokenCodec(CompressionModel):
    """Latent codec for **ViT token features** (Visual Token Codec).

    Splits the sequence into **prefix tokens** (class/registers, length
    ``num_prefix_tokens``) and **patch tokens**. Prefix tokens are compressed with a
    dedicated hyperprior on a tight ``B×C×L×1`` layout; patch tokens are reshaped to a
    spatial map ``B×C×H×W`` and compressed through an analysis transform ``f_a``, a
    standard hyperprior on the latent ``y``, and a **channel-grouped** entropy coder.
    Each patch group combines **channel-wise context**, **spatial context**
    (checkerboard-masked convolutions on the latent grid), and hyperprior side info so
    entropy modeling respects **2D patch geometry** instead of treating tokens as an
    unstructured vector.

    Optional **QP-indexed learnable gains** scale patch latents around entropy coding
    (and scale features before/after shallow ViT blocks) to sweep rate–distortion
    points.

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
        self.num_prefix_tokens = num_prefix_tokens

        self.q_scale_pre = nn.Parameter(torch.ones((65, 1, h_dim)))
        self.q_scale_post = nn.Parameter(torch.ones((65, 1, h_dim)))

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

    def forward(self, h, token_res, qp=0, **kwargs):
        """Forward pass for separate class/patch token compression with quantization parameter.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.
            qp (int): Quantization parameter index in ``[0, 64]`` controlling the
                bitrate–distortion trade-off for patch tokens.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with keys:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
                - ``\"likelihoods\"`` (dict): Likelihoods for class ``\"prefix\"``, patch
                  latents ``\"y\"`` and hyperprior latents ``\"z\"``.
        """
        enc_gain = self.q_scale_enc[qp : qp + 1, :, :, :]
        dec_gain = self.q_scale_dec[qp : qp + 1, :, :, :]

        h = self.pre_vit_blocks(h) * self.q_scale_pre[qp : qp + 1, :, :]

        h_prefix = h[:, 0 : self.num_prefix_tokens]
        h_prefix = rearrange(h_prefix, "B L C -> B C L 1")
        prefix_out = self.cls_lc(h_prefix)
        h_prefix_hat = prefix_out["params"]
        h_prefix_hat = rearrange(h_prefix_hat, "B C L 1 -> B L C")

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

        h_hat = torch.cat([h_prefix_hat, h_patch_hat], dim=1)
        h_hat = h_hat * self.q_scale_post[qp : qp + 1, :, :]
        h_hat = self.post_vit_blocks(h_hat)

        return {
            "h_hat": h_hat,
            "likelihoods": {
                "prefix": prefix_out["likelihoods"]["z"],
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, token_res, qp=0, **kwargs):
        """Compress ViT features with separate class and patch codecs and quantization parameter.

        Args:
            h (torch.Tensor): ViT output tensor of shape ``(B, L, C)``.
            token_res (tuple[int, int]): Spatial token resolution ``(H, W)``.
            qp (int): Quantization parameter index in ``[0, 64]``.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with keys:

                - ``\"strings\"`` (dict): Bitstreams for class ``\"prefix\"``, patch
                  ``\"y\"`` and hyperprior ``\"z\"``.
                - ``\"pstate\"`` (dict): Side information including shapes, padding,
                  token resolution and ``qp``.
        """
        enc_gain = self.q_scale_enc[qp : qp + 1, :, :, :]

        h = self.pre_vit_blocks(h) * self.q_scale_pre[qp : qp + 1, :, :]

        h_prefix = h[:, 0 : self.num_prefix_tokens]
        h_prefix = rearrange(h_prefix, "B L C -> B C L 1")
        prefix_out = self.cls_lc.compress(h_prefix)

        h_patch = h[:, self.num_prefix_tokens :].contiguous()
        h_patch = rearrange(
            h_patch, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1]
        )
        y = self.f_a(h_patch) * enc_gain
        # x --16-> h --2-> y --4-> z
        # if pad 32 for y, y is not compatible with checkerboard codec
        # so we pad 64 for y, then only need to pad 2 for z
        y_pad = border_pad(y, 2)
        hyper_out = self.hyper_lc.compress(y_pad)
        _, _, y_H, y_W = y.shape
        y_out = self.y_lc.compress(y, hyper_out["params"][:, :, :y_H, :y_W])

        return {
            "strings": {
                "prefix": prefix_out["strings"],
                "y": y_out["strings"],
                "z": hyper_out["strings"],
            },
            "pstate": {
                "prefix_shape": prefix_out["shape"],
                "y_shape": y_out["shape"],
                "z_shape": hyper_out["shape"],
                "y_pad": (y_H, y_W),
                "token_res": token_res,
                "qp": qp,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        """Decompress bitstreams back to ViT features.

        Args:
            strings (dict): Bitstreams for ``\"prefix\"``, ``\"y\"`` and ``\"z\"``.
            pstate (dict): Side information produced by :meth:`compress`, including
                shapes, padding and ``qp``.
            **kwargs: Unused keyword arguments for API compatibility.

        Returns:
            dict: A dictionary with key:

                - ``\"h_hat\"`` (torch.Tensor): Reconstructed ViT features.
        """
        qp = pstate["qp"]
        dec_gain = self.q_scale_dec[qp : qp + 1, :, :, :]

        prefix_out = self.cls_lc.decompress(strings["prefix"], pstate["prefix_shape"])
        h_prefix_hat = prefix_out["params"]
        h_prefix_hat = rearrange(h_prefix_hat, "B C L 1 -> B L C")

        y_H, y_W = pstate["y_pad"]
        hyper_out = self.hyper_lc.decompress(strings["z"], pstate["z_shape"])
        y_out = self.y_lc.decompress(
            strings["y"], pstate["y_shape"], hyper_out["params"][:, :, :y_H, :y_W]
        )
        y_hat = y_out["y_hat"] * dec_gain
        h_patch_hat = self.f_s(y_hat)
        h_patch_hat = rearrange(h_patch_hat, "B C H W -> B (H W) C")

        h_hat = torch.cat([h_prefix_hat, h_patch_hat], dim=1)
        h_hat = h_hat * self.q_scale_post[qp : qp + 1, :, :]
        h_hat = self.post_vit_blocks(h_hat)
        return {"h_hat": h_hat}


@register_model("VisualTokenCodecVariant")
class VisualTokenCodecVariant(CompressionModel):
    """Variant of :class:`VisualTokenCodec` for different VBR schemas.
    
    VBR gains: ``q_scale_cls_enc`` / ``q_scale_cls_dec`` on the prefix
      branch replace ``q_scale_pre`` / ``q_scale_post`` on the full sequence.

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
        self.num_prefix_tokens = num_prefix_tokens

        # VBR learnable scales (QP in [0, 64])
        # cls+reg token vbr
        self.q_scale_cls_enc = nn.Parameter(torch.ones((65, h_dim, 1, 1)))
        self.q_scale_cls_dec = nn.Parameter(torch.ones((65, h_dim, 1, 1)))
        # patch token vbr
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
                "prefix": cls_out["likelihoods"]["z"],
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
                "prefix": cls_out["strings"],
                "y": y_out["strings"],
                "z": hyper_out["strings"],
            },
            "pstate": {
                "prefix_shape": cls_out["shape"],
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
        cls_out = self.cls_lc.decompress(strings["prefix"], pstate["prefix_shape"])
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

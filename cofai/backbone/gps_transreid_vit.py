import math
from collections.abc import Iterable
from functools import partial

import torch
import torch.nn as nn

from cofai.token_grouping import GPSTokenGrouper


def to_2tuple(value):
    return tuple(value) if isinstance(value, Iterable) else (value, value)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch, sequence_length, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch,
                sequence_length,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        attention_probs = ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attention = self.attn_drop(attention_probs)
        x = (
            (attention @ value)
            .transpose(1, 2)
            .reshape(
                batch,
                sequence_length,
                channels,
            )
        )
        return self.proj_drop(self.proj(x)), attention_probs


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # Unused by GPS inference but retained for released checkpoint keys.
        self.norm_cross = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, need_attn=False):
        shortcut = x
        x, attn = self.attn(self.norm1(x))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if need_attn:
            return x, attn
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping patches"""

    def __init__(
        self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print(
            "using stride: {}, and patch number is num_y{} * num_x{}".format(
                stride_size, self.num_y, self.num_x
            )
        )
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.stride_size = stride_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride_size
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # Keep the original timm-style size check for patch embedding.
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        )
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class GPSTransReIDVisionTransformer(nn.Module):
    """
    Transformer-based Object Re-Identification
    separate modeling
    first 8 layers to model single-view, and then model cross-view in the last 4 layers
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        camera=0,
        view=0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        sie_xishu=1.0,
        pruning_layers=(),
        pruning_ratios=(),
        propagation_max_iter=10,
        beta=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_embed = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride_size=stride_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Retained for compatibility with the released GPS checkpoints.
        self.cls_token2 = nn.Parameter(torch.zeros(1, 0, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            print(
                "camera number is : {} and viewpoint number is : {}".format(
                    camera, view
                )
            )
            print("using SIE_Lambda is : {}".format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            print("camera number is : {}".format(camera))
            print("using SIE_Lambda is : {}".format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=0.02)
            print("viewpoint number is : {}".format(view))
            print("using SIE_Lambda is : {}".format(sie_xishu))

        print("using drop_out rate is : {}".format(drop_rate))
        print("using attn_drop_out rate is : {}".format(attn_drop_rate))
        print("using drop_path rate is : {}".format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.background_pruning_layers = tuple(int(layer) for layer in pruning_layers)
        self.background_pruning_ratios = tuple(float(ratio) for ratio in pruning_ratios)
        if len(self.background_pruning_layers) != len(self.background_pruning_ratios):
            raise ValueError("GPS pruning layers and ratios must have equal lengths")
        if tuple(
            sorted(set(self.background_pruning_layers))
        ) != self.background_pruning_layers or any(
            layer < 0 or layer >= depth - 1 for layer in self.background_pruning_layers
        ):
            raise ValueError(
                "GPS pruning layers must be unique, increasing block indices"
            )
        if any(not 0.0 <= ratio < 1.0 for ratio in self.background_pruning_ratios):
            raise ValueError("GPS pruning ratios must be in [0, 1)")
        self.propagation_max_iter = int(propagation_max_iter)
        self.beta = float(beta)
        self.token_grouper = GPSTokenGrouper(
            max_iter=self.propagation_max_iter,
            beta=self.beta,
        )

        self.blocks = nn.ModuleList()

        self.depth = depth

        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
            )

        self.norm = norm_layer(embed_dim)

        self.fc = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _group_tokens(
        self,
        x,  # shape [B, N + 1, d]
        attn,  # shape [B, num_heads, N, N]
        keep_ratio,
        orig_indices,
    ):
        grouped = self.token_grouper(
            tokens=x,
            attention=attn,
            keep_ratio=keep_ratio,
            original_indices=orig_indices,
        )
        return grouped.tokens, grouped.kept_indices

    def forward_features(
        self,
        x,
        camera_id,
        view_id,
        label=None,
        dataset_name="query",
    ):
        B = x.shape[0]
        x = self.patch_embed(x)  # B N C

        pos_embed = self.pos_embed
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat([cls_tokens, x], dim=1)

        if self.cam_num > 0 and self.view_num > 0:
            x = (
                x
                + pos_embed
                + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
            )
        elif self.cam_num > 0:
            x = x + pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + pos_embed

        x = self.pos_drop(x)

        if dataset_name == "query":
            viewpoint_num = 3
            lsort = torch.arange(x.size(0)).reshape((-1, viewpoint_num))
            if not (
                torch.all(label[lsort[:, 0]] == label[lsort[:, 1]])
                and torch.all(label[lsort[:, 1]] == label[lsort[:, 2]])
            ):
                raise ValueError("each GPS query group must contain one identity")
            pruning_ratios = self.background_pruning_ratios
        elif dataset_name == "gallery":
            viewpoint_num = 1
            lsort = torch.arange(x.size(0)).reshape((-1, viewpoint_num))
            pruning_ratios = (0.0,) * len(self.background_pruning_ratios)
        else:
            raise ValueError(f"unsupported GPS dataset split: {dataset_name!r}")

        split = 8
        attn_views = [None for _ in range(viewpoint_num)]
        x_views = [x[lsort[:, view]] for view in range(viewpoint_num)]
        num_patches = self.patch_embed.num_patches
        batch_size_per_group = x_views[0].size(0)
        orig_indices_per_view = [
            torch.arange(
                view_index * num_patches,
                (view_index + 1) * num_patches,
                device=x.device,
            )
            .unsqueeze(0)
            .repeat(batch_size_per_group, 1)
            for view_index in range(viewpoint_num)
        ]

        pruning_index = 0
        for i, blk in enumerate(self.blocks[:-1]):
            if i == split:
                cls_token_fused = (
                    sum(view_tokens[:, 0:1] for view_tokens in x_views) / viewpoint_num
                )
                patch_tokens_all = torch.cat(
                    [view_tokens[:, 1:] for view_tokens in x_views],
                    dim=1,
                )
                x_fused = torch.cat([cls_token_fused, patch_tokens_all], dim=1)
                orig_indices_fused = torch.cat(orig_indices_per_view, dim=1)

            if i < split:
                for view in range(viewpoint_num):
                    x_views[view], attn_views[view] = blk(
                        x_views[view],
                        need_attn=True,
                    )
            else:
                x_fused, attn_fused = blk(x_fused, need_attn=True)

            if i in self.background_pruning_layers:
                keep_ratio = 1.0 - pruning_ratios[pruning_index]
                pruning_index += 1
                if i < split:
                    for view in range(viewpoint_num):
                        x_views[view], orig_indices_per_view[view] = self._group_tokens(
                            x=x_views[view],
                            attn=attn_views[view],
                            keep_ratio=keep_ratio,
                            orig_indices=orig_indices_per_view[view],
                        )
                else:
                    x_fused, orig_indices_fused = self._group_tokens(
                        x=x_fused,
                        attn=attn_fused,
                        keep_ratio=keep_ratio,
                        orig_indices=orig_indices_fused,
                    )

        return (
            x_fused,
            lsort[:, 0],
            0.0,
            {
                "indices": orig_indices_fused,
                "token_count": viewpoint_num * num_patches,
            },
        )

    def forward(
        self,
        x,
        cam_label=None,
        view_label=None,
        label=None,
        dataset_name="query",
    ):
        x = self.forward_features(
            x,
            cam_label,
            view_label,
            label=label,
            dataset_name=dataset_name,
        )
        return x


def build_gps_transreid_vit(
    img_size=(256, 128),
    stride_size=16,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    camera=0,
    view=0,
    sie_xishu=1.5,
    **kwargs,
):

    model = GPSTransReIDVisionTransformer(
        img_size=img_size,
        patch_size=16,
        stride_size=stride_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        camera=camera,
        view=view,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sie_xishu=sie_xishu,
        **kwargs,
    )

    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [lower, upper], then translate
        # to [2 * lower - 1, 2 * upper - 1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

import math
from functools import partial
from itertools import repeat

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from cofai.token_grouping import GraphTokenGrouper
# The public GPS evaluation path uses the transformer backbone only.
#from torch._six import container_abcs
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, x_kv = None):
        if x_kv is None:
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            attn_origin = (q @ k.transpose(-2, -1)) * self.scale
            attn_probs = attn_origin.softmax(dim=-1)


            attn = self.attn_drop(attn_probs)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn_probs
            # return x

        else:
            assert 1==0
            pass


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_cross = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)



    def forward(self, x, need_attn = False):

        shortcut = x
        x, attn = self.attn(self.norm1(x))
        x = shortcut + self.drop_path(x)
        # attn = None
        # x = x + self.drop_path(self.attn(self.norm1(x)))

        x = x + self.drop_path(self.mlp(self.norm2(x)))


        if need_attn:
            return x, attn
        else:
            return x


    # def forward(self, x):
    #     x = x + self.drop_path(self.attn(self.norm1(x)))
    #     x = x + self.drop_path(self.mlp(self.norm2(x)))
    #     return x







class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.stride_size = stride_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # Keep the original timm-style size check for patch embedding.
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x



class TransReID_sep(nn.Module):
    """
        Transformer-based Object Re-Identification
        separate modeling
        first 8 layers to model single-view, and then model cross-view in the last 4 layers
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu =1.0,cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = local_feature
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)
        self.cls_token_num = cfg.cls_token_num

        self.cfg = cfg

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token2 = nn.Parameter(torch.zeros(1, cfg.cls_token_num-1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.token_pruning_ratios = cfg.MODEL.TOKEN_PRUNING_RATIO
        self.background_pruning_layers = cfg.MODEL.BACKGROUND_PRUNING_LAYERS
        self.background_pruning_ratios = cfg.MODEL.BACKGROUND_PRUNING_RATIOS
        self.propagation_max_iter = cfg.MODEL.PROPAGATION_MAX_ITER
        self.beta = cfg.MODEL.BETA
        self.token_grouper = GraphTokenGrouper(
            max_iter=self.propagation_max_iter,
            beta=self.beta,
        )

        self.use_bg_pruning = True

        self.blocks = nn.ModuleList()


        self.depth = depth

        for i in range(depth):
            self.blocks.append(Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer))

        self.norm = norm_layer(embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def token_pruning_graph_partitioning(
        self,
        x,            # shape [B, N + 1, d]
        attn,         # shape [B, num_heads, N, N]
        keep_ratio,
        orig_indices,
        max_iter=10,
        beta=0.1,
        eps=1e-3,
    ):
        grouped = self.token_grouper(
            tokens=x,
            attention=attn,
            keep_ratio=keep_ratio,
            original_indices=orig_indices,
        )
        return (
            grouped.tokens,
            grouped.kept_indices,
            grouped.deleted_indices,
            grouped.deleted_sequence_indices,
        )

        # Legacy implementation retained below only as source context.
        cls_token = x[:, :1]  # [B, 1, d]
        tokens    = x[:, 1:]  # [B, N, d]
        B, N, d = x.shape
        num_tokens_current = N - 1
        attn = attn.mean(dim=1, keepdim=True)  # [B, 1, N, N]
        attn = (attn + attn.transpose(-1, -2)) / 2.0  # [B, 1, N, N]

        _, num_heads, N_, N2_ = attn.shape

        assert N_ == N and N2_ == N, "N_ == N and N2_ == N ?"

        keep_num = max(int((N - 1) * keep_ratio), 1)

        if keep_num == (N - 1):
            empty = torch.zeros(B, 0, dtype=torch.long, device=x.device)
            return x, orig_indices, empty, empty

        Praw = attn.reshape(B * num_heads, N, N)

        row_sum = torch.clamp(Praw.sum(dim=-1, keepdim=True), min=1e-8)
        P = Praw / row_sum  # shape [B_*num_heads, N, N]

        B_ = B * num_heads
        device = P.device
        dtype  = P.dtype

        e_cls = torch.zeros(B_, N, device=device, dtype=dtype)
        e_cls[:, 0] = 1.0

        r = e_cls.clone()  # shape [B_, N]
        P_t = P.transpose(1,2)  # [B_, N, N]
        r_unsq = r.unsqueeze(-1) # [B_, N, 1]

        for _iter in range(max_iter):
            r_next = beta * e_cls + (1.0 - beta) * torch.bmm(P_t, r_unsq).squeeze(-1)
            diff = torch.norm(r_next - r, p=1, dim=1)  # shape [B_]
            r = r_next
            r_unsq = r.unsqueeze(-1)
            if torch.all(diff < eps):
                break
        r = r.view(B, num_heads, N)


        patch_scores_all = r[:,:,1:]  # [B, num_heads, N-1]
        patch_scores = patch_scores_all.max(dim=1)[0]  # [B, N-1]

        _, topk_indices = torch.topk(patch_scores, k=keep_num, dim=1, largest=True)  # [B, keep_num]
        topk_indices, _ = torch.sort(topk_indices, dim=1)
        pruned_patches = torch.gather(tokens, 1, topk_indices.unsqueeze(-1).expand(B, keep_num, d))  # [B, keep_num, d]
        x_pruned = torch.cat([cls_token, pruned_patches], dim=1)  # [B, 1+keep_num, d]

        keep_mask = torch.zeros(B, num_tokens_current, dtype=torch.bool, device=patch_scores.device)
        keep_mask.scatter_(1, topk_indices, True)
        delete_mask = ~keep_mask
        deleted_seq_indices = torch.stack([torch.nonzero(delete_mask[i], as_tuple=False).squeeze(-1) for i in range(B)], dim=0)

        kept_orig_idx = torch.gather(orig_indices, 1, topk_indices)
        deleted_orig_idx = torch.gather(orig_indices, 1, deleted_seq_indices)

        return x_pruned, kept_orig_idx, deleted_orig_idx, deleted_seq_indices





    def forward_features(self, x, camera_id, view_id, label=None,
        multi_view=False, flip_view=False, extra_token=False, dataset_name = 'train'):

        B = x.shape[0]

        x = self.patch_embed(x)#B N C

        pos_embed = self.pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat([cls_tokens, x], dim=1)

        if self.cam_num > 0 and self.view_num > 0:
            x = x + pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + pos_embed

        x = self.pos_drop(x)


        if not multi_view:
            split = 8
            if self.local_feature:
                for blk in self.blocks[:-1]:
                    x = blk(x)
                return x
            else:
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]
        else: #multi-view
            if dataset_name == 'train':
                # for Train and Gallery. not for Query. Because Query(Test) is already paired.
                viewpoint_num = 3
                lsort = torch.argsort(label)
                lsort = lsort.reshape((-1,viewpoint_num))
                assert torch.all(label[lsort[:, 0]] == label[lsort[:, 1]])
                assert torch.all(label[lsort[:, 1]] == label[lsort[:, 2]])
                self.token_pruning_ratio = self.cfg.MODEL.TOKEN_PRUNING_RATIO
                self.background_pruning_ratios = self.cfg.MODEL.BACKGROUND_PRUNING_RATIOS

            elif dataset_name == 'query' or dataset_name == 'test':
                viewpoint_num = 3
                lsort = torch.arange(x.size(0)).reshape((-1,viewpoint_num))
                assert torch.all(label[lsort[:, 0]] == label[lsort[:, 1]])
                assert torch.all(label[lsort[:, 1]] == label[lsort[:, 2]])
                self.token_pruning_ratio = self.cfg.MODEL.TOKEN_PRUNING_RATIO
                self.background_pruning_ratios = self.cfg.MODEL.BACKGROUND_PRUNING_RATIOS

            elif dataset_name == 'gallery':
                viewpoint_num = 1
                lsort = torch.arange(x.size(0)).reshape((-1,viewpoint_num))

                # Single-view does not need pruning
                self.background_pruning_ratios = self.cfg.MODEL.BACKGROUND_PRUNING_RATIOS
                self.background_pruning_ratios = [0.0 for _ in self.background_pruning_ratios] # No background pruning for single-view
                self.token_pruning_ratio = 0.0

            split = 8
            if self.local_feature:
                attn_views = [None for _ in range(viewpoint_num)]
                x_views = []
                for v in range(viewpoint_num):
                    x_views.append(x[lsort[:, v]])

                if 'transreid_baseline' in self.cfg.descrip: # Directly Concat
                    patch_tokens_list = []
                    for v in range(viewpoint_num):
                        patch_tokens_list.append(x_views[v][:, 1:])
                    patch_tokens_all = torch.cat(patch_tokens_list, dim=1)

                    x_fused = torch.cat([x_views[0][:, 0:1], patch_tokens_all], dim=1)

                    for i, blk in enumerate(self.blocks[:-1]):
                        x_fused = blk(x_fused)

                elif 'gps_multi_view' in self.cfg.descrip:
                    num_patches = self.patch_embed.num_patches
                    batch_size_per_group = x_views[0].size(0)
                    orig_indices_per_view = [
                        torch.arange(
                            view_index * num_patches,
                            (view_index + 1) * num_patches,
                            device=x.device,
                        ).unsqueeze(0).repeat(batch_size_per_group, 1)
                        for view_index in range(viewpoint_num)
                    ]

                    bg_index = 0
                    for i, blk in enumerate(self.blocks[:-1]):
                        if i == split:
                            # cat patch_tokens
                            cls_token_fused = 0
                            for v in range(viewpoint_num):
                                cls_token_fused += x_views[v][:, 0:1]
                            cls_token_fused = cls_token_fused / viewpoint_num

                            patch_tokens_list = []
                            for v in range(viewpoint_num):
                                patch_tokens_list.append(x_views[v][:, 1:])

                            patch_tokens_all = torch.cat(patch_tokens_list, dim=1)
                            x_fused = torch.cat([cls_token_fused, patch_tokens_all], dim=1)


                        if i < split:
                            for v in range(viewpoint_num):
                                assert x_views[v] is not None
                                x_views[v], attn_views[v] = blk(x_views[v], need_attn = True)
                        else:
                            x_fused, attn_fused = blk(x_fused, need_attn = True)

                        if i in self.background_pruning_layers:

                            bg_pruning_ratio = self.background_pruning_ratios[bg_index]
                            bg_index += 1
                            if self.use_bg_pruning == False:
                                keep_ratio = 1.
                            else:
                                keep_ratio = 1. - bg_pruning_ratio

                            if i < split:
                                for v in range(viewpoint_num):

                                    x_views[v], orig_indices_per_view[v], _, _ = self.token_pruning_graph_partitioning(
                                        x = x_views[v],
                                        attn = attn_views[v],
                                        keep_ratio = keep_ratio,
                                        orig_indices = orig_indices_per_view[v],
                                        max_iter = self.propagation_max_iter,
                                        beta = self.beta
                                    )

                            else:
                                orig_indices_fused = torch.cat(orig_indices_per_view, dim=1)
                                x_fused, _, _, _ = self.token_pruning_graph_partitioning(
                                    x = x_fused,
                                    attn = attn_fused,
                                    keep_ratio = keep_ratio,
                                    orig_indices = orig_indices_fused,
                                    max_iter = self.propagation_max_iter,
                                    beta = self.beta
                                )


                return x_fused, lsort[:, 0], 0.0

            else:

                raise NotImplementedError("Multi-view inference requires MODEL.JPM=True/local_feature=True.")

    def forward(self, x, cam_label=None, view_label=None,label=None, multi_view=False,flip_view=False, extra_token=False, dataset_name='train'):
        x = self.forward_features(x, cam_label, view_label, label = label,
        multi_view = multi_view, flip_view = flip_view, extra_token = extra_token, dataset_name = dataset_name)
        return x

    def set_pruning_mode(self, epoch, warmup_end = 5):
        if epoch <= warmup_end:
            self.use_bg_pruning = False
        else:
            self.use_bg_pruning = True

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def vit_base_patch16_224_TransReID_seq(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):

    model = TransReID_sep(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,\
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8,  mlp_ratio=3., qkv_bias=False, drop_path_rate = drop_path_rate,\
        camera=camera, view=view,  drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def deit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
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

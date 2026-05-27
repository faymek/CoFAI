"""
Backbone wrappers for DINOv2 and CLIP ViT models.

Provides frozen inference from intermediate block tokens to final predictions
(classification logits or segmentation maps).
"""

import os
import sys
import math
import itertools
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_BACKBONE_DIR = os.path.dirname(os.path.abspath(__file__))
_RELEASE_ROOT = os.path.dirname(_BACKBONE_DIR)

from dotenv import load_dotenv
load_dotenv()
_PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(_RELEASE_ROOT))))

_DINOV2_PKG_DIR = os.path.join(_PROJECT_ROOT, "cofai", "backbone")
if _DINOV2_PKG_DIR not in sys.path:
    sys.path.insert(0, _DINOV2_PKG_DIR)

try:
    from dinov2.hub.classifiers import dinov2_vitl14_lc, dinov2_vitg14_lc
    _HAS_DINOV2 = True
except Exception as e:
    _HAS_DINOV2 = False
    print(f"Failed to import DINOv2: {e}")

UTILS_DIR = os.path.join(_RELEASE_ROOT, "cfg")

if _HAS_DINOV2:
    _DINOV2_REGISTRY = {
        "dinov2_vitl14": {
            "lc_fn": dinov2_vitl14_lc,
            "pretrain": "dinov2_vitl14_pretrain.pth",
            "linear_head": "dinov2_vitl14_linear_head.pth",
            "seg_head": "dinov2_vitl14_voc2012_linear_head.pth",
            "embed_dim": 1024,
            "config": os.path.join(UTILS_DIR, "dinov2_vitl14_voc2012_linear_config.py"),
            "vit_fn": "vit_large",
            "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0, block_chunks=0),
        },
        "dinov2_vitg14": {
            "lc_fn": dinov2_vitg14_lc,
            "pretrain": "dinov2_vitg14_pretrain.pth",
            "linear_head": "dinov2_vitg14_linear_head.pth",
            "seg_head": "dinov2_vitg14_voc2012_linear_head.pth",
            "embed_dim": 1536,
            "config": os.path.join(UTILS_DIR, "dinov2_vitg14_voc2012_linear_config.py"),
            "vit_fn": "vit_giant2",
            "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0,
                               block_chunks=0, ffn_layer="swiglufused"),
        },
    }
else:
    _DINOV2_REGISTRY = {}

try:
    import clip as _clip_module
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False


# ========================= Segmentation Head =========================

class SegmentationHead(nn.Module):
    """BN + 1x1 Conv segmentation head (matches official VOC2012 linear head)."""
    def __init__(self, in_channels=1024, num_classes=21):
        super().__init__()
        self.bn = nn.SyncBatchNorm(in_channels)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv_seg(x)
        return x


def load_seg_head(weights_path, in_channels=1024, num_classes=21, device='cuda'):
    """Load segmentation head weights."""
    head = SegmentationHead(in_channels, num_classes)
    ckpt = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    head_state = {}
    for k, v in ckpt.items():
        if k.startswith('decode_head.'):
            new_k = k.replace('decode_head.', '')
            head_state[new_k] = v
    head.load_state_dict(head_state, strict=True)
    return head.to(device).eval()


# ========================= CLIP Wrapper =========================

class ClipWrapper:
    """
    CLIP ViT-L/14 wrapper.

    Classification: zero-shot via cosine similarity with text embeddings.
    """

    def __init__(self, classnames_path, device="cuda",
                 template="a photo of a {}"):
        assert _HAS_CLIP, "pip install git+https://github.com/openai/CLIP.git"
        model, _ = _clip_module.load("ViT-L/14", device=device)
        model.eval().float()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.visual = model.visual
        self.device = device
        self.weights_root = None
        self.head = None

        self._resblocks = list(self.visual.transformer.resblocks)
        self._ln_post = self.visual.ln_post
        self._proj = self.visual.proj

        self.text_emb = self._build_text_emb(classnames_path, template)

    class _BackboneView:
        """Thin proxy so that wrapper.backbone.blocks / .norm work."""
        def __init__(self, resblocks, ln_post):
            self.blocks = resblocks
            self.norm = ln_post
        def to(self, device):
            for b in self.blocks:
                b.to(device)
            self.norm.to(device)
            return self
        def cpu(self):
            return self.to('cpu')
        def parameters(self):
            for b in self.blocks:
                yield from b.parameters()
            yield from self.norm.parameters()

    @property
    def backbone(self):
        return self._BackboneView(self._resblocks, self._ln_post)

    @torch.no_grad()
    def _build_text_emb(self, classnames_path, template):
        names = []
        with open(classnames_path, 'r') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                names.append(" ".join(parts[1:]))
        prompts = [template.format(n) for n in names]
        tokens = _clip_module.tokenize(prompts).to(self.device)
        text_feat = self.model.encode_text(tokens).float()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return text_feat

    @torch.no_grad()
    def forward_from_tokens(self, tokens, start_block_idx):
        """tokens: [B, T, D] -> logits [B, C]  (zero-shot)"""
        x = tokens.permute(1, 0, 2).contiguous()
        for i in range(start_block_idx + 1, len(self._resblocks)):
            x = self._resblocks[i](x)
        x = x.permute(1, 0, 2)
        cls = x[:, 0, :]
        cls = self._ln_post(cls)
        if isinstance(self._proj, torch.Tensor):
            img_feat = cls @ self._proj
        else:
            img_feat = self._proj(cls)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * img_feat @ self.text_emb.t()


# ========================= DINOv2 Wrapper =========================

class Dinov2Wrapper:
    """
    DINOv2 ViT wrapper for classification and segmentation.

    Supports dinov2_vitl14 (24 blocks, dim=1024) and dinov2_vitg14 (40 blocks, dim=1536).
    """
    def __init__(self, head_layers=1, model_name="dinov2_vitl14",
                 weights_root=os.path.join(_RELEASE_ROOT, "pretrained"),
                 device="cuda"):
        reg = _DINOV2_REGISTRY[model_name]
        self.model_name = model_name
        self.embed_dim = reg["embed_dim"]

        back = os.path.join(weights_root, reg["pretrain"])
        head = os.path.join(weights_root, reg["linear_head"])
        lc_fn = reg["lc_fn"]
        clf = lc_fn(layers=head_layers, pretrained=True, weights=[back, head])
        clf.to(device).eval()
        for p in clf.parameters():
            p.requires_grad_(False)
        self.backbone = getattr(clf, "backbone", clf)
        self.head = getattr(clf, "linear_head", None)
        self.seg_head = None
        self.device = device
        self.weights_root = weights_root

    def load_segmentation_head(self, seg_head_path=None, num_classes=21):
        reg = _DINOV2_REGISTRY[self.model_name]
        if seg_head_path is None:
            seg_head_path = os.path.join(self.weights_root, reg["seg_head"])
        self.seg_head = load_seg_head(
            seg_head_path, in_channels=self.embed_dim,
            num_classes=num_classes, device=self.device)
        print(f"  Segmentation head loaded: {seg_head_path}")

    @torch.no_grad()
    def forward_from_tokens(self, tokens, start_block_idx):
        """
        Forward from intermediate tokens to classification logits.

        tokens: [B, N, D] (including CLS token)
        Returns: logits [B, num_classes]
        """
        x = tokens
        for i in range(start_block_idx + 1, len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        mean_patch = patch_tokens.mean(dim=1)
        linear_input = torch.cat([cls_token, mean_patch], dim=1)
        return self.head(linear_input)


# ========================= Segmentation Evaluator =========================

class SegmentationEvaluator:
    """
    Segmentation evaluator with slide inference for VOC2012.

    Subclasses must implement quantize_tokens(tokens_np) -> tensor on device.
    """

    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    NUM_CLASSES = 21
    IGNORE_INDEX = 255
    CROP_SIZE = (512, 512)
    STRIDE = (341, 341)
    PATCH_SIZE = 14

    def normalize(self, x, mode='per_image'):
        """Per-image normalization."""
        eps = 1e-5
        mu_full = x.mean()
        var = ((x - mu_full) ** 2).mean()
        std = np.sqrt(var + eps)
        return ((x - mu_full) / std).astype(np.float32), \
               np.array([[mu_full]], dtype=np.float32), \
               np.array([[std]], dtype=np.float32)

    def quantize_tokens(self, tokens_np):
        """Override in subclass: quantize [1+N, D] -> tensor [1+N, D] on device."""
        raise NotImplementedError

    @staticmethod
    def get_slide_crops(h_img, w_img, crop_size, stride):
        """Compute slide window crop regions."""
        h_crop, w_crop = crop_size
        h_stride, w_stride = stride
        crops = []
        for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
            for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crops.append((y1, x1, y2, x2))
        return crops

    def slide_inference_decode(self, backbone, head, feature_list, crops, img_shape):
        """Slide-window decode and fusion."""
        h_img, w_img = img_shape
        h_crop, w_crop = self.CROP_SIZE
        patch_size = self.PATCH_SIZE

        preds = torch.zeros((1, self.NUM_CLASSES, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, 1, h_img, w_img), device=self.device)

        for i, (y1, x1, y2, x2) in enumerate(crops):
            feat = feature_list[i]
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat).to(self.device)
            else:
                feat = feat.to(self.device)

            x = feat
            for blk_idx in range(self.layer_idx + 1, len(backbone.blocks)):
                x = backbone.blocks[blk_idx](x)
            x = backbone.norm(x)

            patch_tokens = x[:, 1:, :]

            actual_h = y2 - y1
            actual_w = x2 - x1
            padded_h = math.ceil(actual_h / patch_size) * patch_size
            padded_w = math.ceil(actual_w / patch_size) * patch_size
            feat_h = padded_h // patch_size
            feat_w = padded_w // patch_size

            patch_tokens = patch_tokens.reshape(1, feat_h, feat_w, -1).permute(0, 3, 1, 2)
            logits = head(patch_tokens)

            logits_up = F.interpolate(logits, size=(h_crop, w_crop), mode='bilinear', align_corners=False)
            logits_crop = logits_up[:, :, :actual_h, :actual_w]

            preds[:, :, y1:y2, x1:x2] += logits_crop
            count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "count_mat has zero values"
        preds = preds / count_mat
        return preds

    @staticmethod
    def _resize_dims(orig_h, orig_w, short_side=512):
        """Compute resized dimensions (shortest side = short_side, keep ratio)."""
        scale = short_side / min(orig_h, orig_w)
        return int(round(orig_h * scale)), int(round(orig_w * scale))

    @torch.no_grad()
    def evaluate(self, seg_feat_dir, image_list=None, verbose=True):
        """
        Run segmentation evaluation on VOC2012.

        Args:
            seg_feat_dir: directory with pre-extracted features (name.npy, shape [num_slides, 1+N, D])
            image_list: text file with image names (one per line, no extension)
            verbose: print progress

        Returns:
            dict with 'miou', 'acc', 'class_iou'
        """
        from PIL import Image
        from tqdm import tqdm

        reg = _DINOV2_REGISTRY[self.model_name]

        if verbose:
            print(f"  Loading backbone + seg head ({self.model_name})...")

        from dinov2.models import vision_transformer as vits
        vit_builder = getattr(vits, reg["vit_fn"])
        backbone = vit_builder(**reg["vit_kwargs"])
        backbone_ckpt = os.path.join(self.weights_root, reg["pretrain"])
        backbone.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"), strict=True)
        backbone = backbone.to(self.device).eval()

        head_ckpt = os.path.join(self.weights_root, reg["seg_head"])
        head = load_seg_head(head_ckpt, in_channels=reg["embed_dim"],
                             num_classes=self.NUM_CLASSES, device=self.device)

        if image_list and os.path.exists(image_list):
            with open(image_list, 'r') as f:
                val_list = [ln.strip() for ln in f if ln.strip()]
        else:
            val_txt = os.path.join(self.voc_root, 'ImageSets/Segmentation/val.txt')
            with open(val_txt, 'r') as f:
                val_list = [ln.strip() for ln in f if ln.strip()]

        if verbose:
            print(f"  Images: {len(val_list)}, feature dir: {seg_feat_dir}")
            print("  Starting slide inference...")

        hist = np.zeros((self.NUM_CLASSES, self.NUM_CLASSES), dtype=np.int64)
        missing = 0

        for name in tqdm(val_list, desc="Seg eval", disable=not verbose):
            feat_path = os.path.join(seg_feat_dir, f"{name}.npy")
            if not os.path.exists(feat_path):
                missing += 1
                continue

            img_path = os.path.join(self.voc_root, 'JPEGImages', f'{name}.jpg')
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            h_img, w_img = self._resize_dims(orig_h, orig_w, short_side=512)

            crops = self.get_slide_crops(h_img, w_img, self.CROP_SIZE, self.STRIDE)

            features = np.load(feat_path)
            assert features.shape[0] == len(crops), \
                f"{name}: slide count mismatch feat={features.shape[0]} vs crops={len(crops)}"

            quantized_list = []
            for s in range(features.shape[0]):
                tokens_np = features[s].astype(np.float32)
                x_cal = self.quantize_tokens(tokens_np)
                quantized_list.append(x_cal.unsqueeze(0))

            preds_logits = self.slide_inference_decode(
                backbone, head, quantized_list, crops, (h_img, w_img)
            )

            gt_path = os.path.join(self.voc_root, 'SegmentationClass', f'{name}.png')
            gt = np.array(Image.open(gt_path))
            ori_h, ori_w = gt.shape[:2]

            if (h_img, w_img) != (ori_h, ori_w):
                preds_logits = F.interpolate(
                    preds_logits, size=(ori_h, ori_w),
                    mode='bilinear', align_corners=False
                )
            seg_pred = preds_logits.argmax(dim=1).squeeze(0).cpu().numpy()

            mask = gt != self.IGNORE_INDEX
            hist += np.bincount(
                self.NUM_CLASSES * gt[mask].astype(int) + seg_pred[mask].astype(int),
                minlength=self.NUM_CLASSES ** 2
            ).reshape(self.NUM_CLASSES, self.NUM_CLASSES)

        if missing > 0 and verbose:
            print(f"  [warn] Missing features: {missing} images")

        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        miou = np.nanmean(iou)
        acc = np.diag(hist).sum() / hist.sum()

        return {
            'miou': miou,
            'acc': acc,
            'class_iou': iou,
        }

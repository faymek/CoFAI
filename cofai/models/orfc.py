"""
ORFC (Optimized Rotation for Feature Compression) models for CoFAI engine.

Provides two model classes:
- Dinov2ClsORFC: DINOv2 backbone + ORFC/SoftPQ for classification
- Dinov2SlideSegORFC: DINOv2 backbone + ORFC/SoftPQ with sliding window for segmentation

Supports two codec types:
- "orfc": Standard OPQ with fixed rotation + codebooks (.npz weights)
- "soft_pq": Differentiable Soft-PQ with learned rotation + codebooks (.pt checkpoint)
"""

import itertools
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from compressai.models.base import CompressionModel
from compressai.registry import register_model

from cofai.backbone import Dinov2OrgBackbone
from cofai.engine.registry import instantiate_class
from cofai.entropy_models.orfc_model import (
    batch_normalize_gpu,
    batch_inv_normalize_gpu,
    batched_assign,
)


class _ORFCMixin:
    """Shared ORFC encode/decode logic for all ORFC model variants.

    Handles:
    - Loading .npz weights (R, codebooks, optional PMF)
    - ORFC quantization: normalize → rotate → PQ assign → inverse rotate → denormalize
    - Rate computation (fixed bits and rANS)
    """

    def _init_orfc(self, orfc_weights_path, K, embedding_dim, device="cuda"):
        """Load ORFC weights and set up quantization parameters."""
        self._orfc_K = K
        self._orfc_embedding_dim = embedding_dim

        data = np.load(orfc_weights_path, allow_pickle=True)
        R = data["R"]  # (D, D)
        codebooks = data["codebooks"]  # (G, K, d)

        D = R.shape[0]
        self._orfc_feat_dim = D
        self._orfc_num_groups = D // embedding_dim

        self.register_buffer("_orfc_R", torch.from_numpy(R).float())
        self.register_buffer("_orfc_codebooks", torch.from_numpy(codebooks).float())

        if "pmf" in data:
            self.register_buffer(
                "_orfc_pmf", torch.from_numpy(data["pmf"].astype(np.float32))
            )
        else:
            self._orfc_pmf = None

    def _orfc_actual_bits(self, labels):
        """Compute actual bits via rANS encoding, fallback to theoretical max."""
        byte_strings = self._orfc_rans_encode(labels)
        if byte_strings is not None:
            return sum(len(s) for s in byte_strings) * 8.0
        n_tokens = labels.shape[1]
        return float(n_tokens * self._orfc_num_groups * math.log2(self._orfc_K))

    def _orfc_encode_decode(self, tokens):
        """Apply ORFC quantization and reconstruction on token features.

        Args:
            tokens: (B, N, D) tensor of feature tokens

        Returns:
            tokens_hat: (B, N, D) reconstructed tokens
            labels: (G, B*N) PQ index labels for rate computation
        """
        B, N, D = tokens.shape
        device = tokens.device

        Y, mu, std = batch_normalize_gpu(tokens, mode="per_image")

        flat = Y.reshape(B * N, D)
        Z = flat @ self._orfc_R  # (B*N, D)

        z_3d = Z.reshape(B * N, self._orfc_num_groups, self._orfc_embedding_dim)
        z_3d = z_3d.permute(1, 0, 2).contiguous()  # (G, B*N, d)

        z_hat_3d, labels = batched_assign(
            z_3d, self._orfc_codebooks, device=device
        )

        flat_hat = z_hat_3d.permute(1, 0, 2).reshape(B * N, D)
        Y_hat = (flat_hat @ self._orfc_R.T).reshape(B, N, D)
        tokens_hat = batch_inv_normalize_gpu(Y_hat, mu, std)

        return tokens_hat, labels

    def _orfc_rans_encode(self, labels):
        """Encode PQ labels to bytes using rANS with stored PMF.

        Args:
            labels: (G, N_total) int64 tensor of PQ indices

        Returns:
            byte_strings: list of bytes (one per group), or None if PMF unavailable
        """
        if self._orfc_pmf is None:
            return None

        try:
            from compressai.ans import RansEncoder
        except ImportError:
            return None

        G, N = labels.shape
        K = self._orfc_K
        pmf = self._orfc_pmf  # (G, K)
        byte_strings = []

        for g in range(G):
            pmf_g = pmf[g].cpu().numpy()
            pmf_int = (pmf_g * (1 << 16)).astype(np.int32)
            pmf_int = np.maximum(pmf_int, 1)
            pmf_int[-1] = (1 << 16) - 1 - pmf_int[:-1].sum()

            indices = labels[g].cpu().numpy().astype(np.int32)
            cdf = np.zeros(K + 2, dtype=np.int32)
            cdf[1:K + 1] = np.cumsum(pmf_int)
            cdf[K + 1] = 1 << 16
            cdf_list = cdf.tolist()

            encoder = RansEncoder()
            bs = encoder.encode_with_indexes(
                indices.tolist(),
                [0] * N,
                [cdf_list],
                [K + 2],
                [0],
            )
            byte_strings.append(bs)

        return byte_strings

    def _orfc_rans_decode(self, byte_strings, n_tokens):
        """Decode rANS byte strings back to PQ labels.

        Args:
            byte_strings: list of bytes (one per group)
            n_tokens: number of tokens to decode

        Returns:
            labels: (G, n_tokens) int64 tensor
        """
        from compressai.ans import RansDecoder

        G = self._orfc_num_groups
        K = self._orfc_K
        pmf = self._orfc_pmf  # (G, K)
        device = self._orfc_R.device

        all_labels = []

        for g in range(G):
            pmf_g = pmf[g].cpu().numpy()
            pmf_int = (pmf_g * (1 << 16)).astype(np.int32)
            pmf_int = np.maximum(pmf_int, 1)
            pmf_int[-1] = (1 << 16) - 1 - pmf_int[:-1].sum()

            cdf = np.zeros(K + 2, dtype=np.int32)
            cdf[1:K + 1] = np.cumsum(pmf_int)
            cdf[K + 1] = 1 << 16
            cdf_list = cdf.tolist()

            decoder = RansDecoder()
            decoder.set_stream(byte_strings[g])
            indices = decoder.decode_stream(
                [0] * n_tokens,
                [cdf_list] * n_tokens,
                [K + 2] * n_tokens,
                [0] * n_tokens,
            )
            all_labels.append(torch.tensor(indices, dtype=torch.int64, device=device))

        return torch.stack(all_labels)  # (G, n_tokens)

    def _orfc_decode_from_labels(self, labels, mu, std, B, N):
        """Reconstruct tokens from PQ labels + normalization stats.

        Args:
            labels: (G, B*N) int64 tensor
            mu, std: normalization stats from encode, shape (B, 1, 1)
            B, N: batch size and token count

        Returns:
            tokens_hat: (B, N, D) reconstructed tokens
        """
        D = self._orfc_feat_dim
        device = labels.device
        G = self._orfc_num_groups
        d = self._orfc_embedding_dim

        # (G, B*N) → gather from codebooks (G, K, d)
        labels_exp = labels.unsqueeze(-1).expand(G, B * N, d)  # (G, B*N, d)
        z_hat_3d = torch.gather(self._orfc_codebooks, 1,
                                labels_exp)  # (G, B*N, d)

        flat_hat = z_hat_3d.permute(1, 0, 2).reshape(B * N, D)
        Y_hat = (flat_hat @ self._orfc_R.T).reshape(B, N, D)
        tokens_hat = batch_inv_normalize_gpu(Y_hat, mu, std)
        return tokens_hat


class _SoftPQMixin:
    """Shared Soft-PQ encode/decode logic using FeatureCodec (.pt checkpoint).

    Handles:
    - Loading .pt checkpoint (FeatureCodec = OrthogonalTransform + SoftPQ)
    - Codec quantization: normalize → rotate → PQ → inverse rotate → denormalize
    - Rate computation via learned prior or fixed ceiling
    """

    def _init_soft_pq(self, codec_path, device="cuda"):
        """Load trained FeatureCodec from .pt checkpoint."""
        from cofai.entropy_models.soft_pq import load_codec
        self._codec = load_codec(codec_path, device=device)
        self._codec.eval()
        pq = self._codec.pq
        self._spq_G = pq.G
        self._spq_K = pq.K
        self._spq_d = pq.d

    def _soft_pq_encode_decode(self, tokens):
        """Apply FeatureCodec: normalize → codec → denormalize.

        Args:
            tokens: (B, N, D) tensor of feature tokens

        Returns:
            tokens_hat: (B, N, D) reconstructed tokens
            usage: (G, K) usage counts
        """
        Y, mu, std = batch_normalize_gpu(tokens, mode="per_image")
        Y_hat, usage = self._codec(Y)
        tokens_hat = batch_inv_normalize_gpu(Y_hat, mu, std)
        return tokens_hat, usage

    def _soft_pq_actual_bits(self, tokens):
        """Compute rate: rANS bits if available, else cross-entropy estimate."""
        B, N, D = tokens.shape
        pq = self._codec.pq
        if pq.use_rate and pq._last_rate is not None:
            return pq._last_rate.item() * N
        return float(N * self._spq_G * math.log2(self._spq_K))


@register_model("Dinov2ClsORFC")
class Dinov2ClsORFC(CompressionModel, _ORFCMixin, _SoftPQMixin):
    """DINOv2 backbone + ORFC/SoftPQ for classification.

    Compresses intermediate ViT features using standard OPQ (.npz) or
    trained Soft-PQ FeatureCodec (.pt), then continues through remaining
    backbone blocks and classification head.

    Args:
        dino_backbone (dict): Configuration for Dinov2OrgBackbone.
        codec_type (str): "orfc" for standard OPQ or "soft_pq" for FeatureCodec.
        orfc_weights_path (str): Path to .npz file (codec_type="orfc").
        codec_path (str): Path to .pt FeatureCodec checkpoint (codec_type="soft_pq").
        K (int): Codebook size per group (codec_type="orfc" only).
        embedding_dim (int): Sub-vector dimension (codec_type="orfc" only).
        heads (dict, optional): Task head configurations.
    """

    def __init__(
        self,
        dino_backbone={},
        codec_type="orfc",
        orfc_weights_path="",
        codec_path="",
        K=256,
        embedding_dim=32,
        heads: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2OrgBackbone(**dino_backbone)
        self.patch_size = self.dino.patch_size
        self.codec_type = codec_type

        if codec_type == "orfc":
            self._init_orfc(orfc_weights_path, K, embedding_dim)
        elif codec_type == "soft_pq":
            device = dino_backbone.get("device", "cuda")
            self._init_soft_pq(codec_path, device=device)
        else:
            raise ValueError(f"Unknown codec_type: {codec_type}")

        self.heads = nn.ModuleDict()
        if heads:
            for task, hcfg in heads.items():
                if not isinstance(task, str) or hcfg is None:
                    continue
                if not isinstance(hcfg, dict) or "type" not in hcfg:
                    raise ValueError(f"heads.{task} must be a dict with 'type' field")
                self.heads[task] = instantiate_class(hcfg)

    def forward(self, x):
        raise NotImplementedError("Inference-only model.")

    def forward_test(self, x, qp=None, tasks=[], **kwargs):
        """Forward pass: encode → ORFC/SoftPQ → decode → head."""
        with torch.inference_mode():
            h = self.dino.encode(x)  # (B, 1+HW, D)

            if self.codec_type == "soft_pq":
                h_hat, usage = self._soft_pq_encode_decode(h)
                bits = self._soft_pq_actual_bits(h)
            else:
                h_hat, labels = self._orfc_encode_decode(h)
                bits = self._orfc_actual_bits(labels)

            task_feats = {}
            if "cls" in tasks:
                cls_features = self.dino.decode_cls(h_hat)
                if "cls" in self.heads:
                    task_feats["cls"] = self.heads["cls"](cls_features)

            coded_data = {
                "bits": {"orfc": bits},
            }
            return coded_data, task_feats

    def compress(self, x, qp=None, tasks=[], **kwargs):
        """Real compression: encode → codec → encode bitstream."""
        with torch.inference_mode():
            h = self.dino.encode(x)  # (B, 1+HW, D)
            B, N, D = h.shape

            if self.codec_type == "soft_pq":
                h_hat, usage = self._soft_pq_encode_decode(h)
                bits = self._soft_pq_actual_bits(h)
                coded_data = {
                    "bits": {"orfc": bits},
                    "pstate": {"h_hat": h_hat},
                }
            else:
                Y, mu, std = batch_normalize_gpu(h, mode="per_image")
                flat = Y.reshape(B * N, D)
                Z = flat @ self._orfc_R

                z_3d = Z.reshape(B * N, self._orfc_num_groups, self._orfc_embedding_dim)
                z_3d = z_3d.permute(1, 0, 2).contiguous()

                _, labels = batched_assign(z_3d, self._orfc_codebooks, device=h.device)

                byte_strings = self._orfc_rans_encode(labels)
                if byte_strings is not None:
                    total_bits = sum(len(s) for s in byte_strings) * 8.0
                    coded_data = {
                        "bits": {"orfc": total_bits},
                        "pstate": {
                            "shape": (B, N, D),
                            "mu": mu.cpu(),
                            "std": std.cpu(),
                            "byte_strings": byte_strings,
                        },
                    }
                else:
                    total_bits = self._orfc_actual_bits(labels)
                    coded_data = {
                        "bits": {"orfc": total_bits},
                        "pstate": {
                            "shape": (B, N, D),
                            "mu": mu.cpu(),
                            "std": std.cpu(),
                            "labels": labels.cpu(),
                        },
                    }
            return coded_data

    def decompress(self, coded_unit, tasks=[], **kwargs):
        """Decompress: decode bitstream → reconstruct → head."""
        pstate = coded_unit["pstate"]

        if self.codec_type == "soft_pq":
            tokens_hat = pstate["h_hat"]
        else:
            B, N, D = pstate["shape"]
            device = self._orfc_R.device
            mu = pstate["mu"].to(device)
            std = pstate["std"].to(device)

            if "byte_strings" in pstate:
                byte_strings = pstate["byte_strings"]
                labels = self._orfc_rans_decode(byte_strings, B * N)
            else:
                labels = pstate["labels"].to(device)

            tokens_hat = self._orfc_decode_from_labels(labels, mu, std, B, N)

        task_feats = {}
        if "cls" in tasks:
            cls_features = self.dino.decode_cls(tokens_hat)
            if "cls" in self.heads:
                task_feats["cls"] = self.heads["cls"](cls_features)
        return task_feats

    def get_feature_numel(self, x):
        """Number of compressible feature elements (for bpfp).

        Computed analytically: (num_patches + 1_cls) * embed_dim * batch.
        """
        B, C, H, W = x.shape
        ps = self.patch_size
        n_patches = (math.ceil(H / ps)) * (math.ceil(W / ps))
        n_tokens = n_patches + 1  # CLS token
        embed_dim = self.dino.model.embed_dim
        return B * n_tokens * embed_dim


@register_model("Dinov2SlideSegORFC")
class Dinov2SlideSegORFC(CompressionModel, _ORFCMixin, _SoftPQMixin):
    """DINOv2 backbone + ORFC/SoftPQ with sliding window for segmentation.

    Processes large images via sliding window crops. Each crop is independently:
    1. Encoded through backbone blocks[:slot]
    2. ORFC/SoftPQ quantized (with per-crop normalization)
    3. Continued through blocks[slot:] + norm
    4. Passed to segmentation head

    Overlapping crop predictions are averaged for final output.

    Args:
        dino_backbone (dict): Configuration for Dinov2OrgBackbone.
        codec_type (str): "orfc" for standard OPQ or "soft_pq" for FeatureCodec.
        orfc_weights_path (str): Path to .npz file (codec_type="orfc").
        codec_path (str): Path to .pt FeatureCodec checkpoint (codec_type="soft_pq").
        K (int): Codebook size per group (codec_type="orfc" only).
        embedding_dim (int): Sub-vector dimension (codec_type="orfc" only).
        slide_size (list): Sliding window crop size [H, W].
        slide_stride (list): Sliding window stride [H, W].
        heads (dict, optional): Task head configurations.
    """

    def __init__(
        self,
        dino_backbone={},
        codec_type="orfc",
        orfc_weights_path="",
        codec_path="",
        K=256,
        embedding_dim=32,
        slide_size=(512, 512),
        slide_stride=(341, 341),
        heads: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2OrgBackbone(**dino_backbone)
        self.patch_size = self.dino.patch_size
        self.slide_size = tuple(slide_size)
        self.slide_stride = tuple(slide_stride)
        self.codec_type = codec_type

        if codec_type == "orfc":
            self._init_orfc(orfc_weights_path, K, embedding_dim)
        elif codec_type == "soft_pq":
            device = dino_backbone.get("device", "cuda")
            self._init_soft_pq(codec_path, device=device)
        else:
            raise ValueError(f"Unknown codec_type: {codec_type}")

        self.heads = nn.ModuleDict()
        if heads:
            for task, hcfg in heads.items():
                if not isinstance(task, str) or hcfg is None:
                    continue
                if not isinstance(hcfg, dict) or "type" not in hcfg:
                    raise ValueError(f"heads.{task} must be a dict with 'type' field")
                self.heads[task] = instantiate_class(hcfg)

    def forward(self, x):
        raise NotImplementedError("Inference-only model.")

    def _center_pad(self, x):
        """Pad spatial dims to the next multiple of patch_size (center-aligned).

        Matches opq_release's CenterPadding: 512→518 for patch_size=14.
        """
        pads = list(itertools.chain.from_iterable(
            self._get_pad(m) for m in x.shape[:1:-1]
        ))
        return F.pad(x, pads)

    def _get_pad(self, size):
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left
        return pad_left, pad_right

    def _get_slide_crops(self, h_img, w_img):
        """Compute sliding window crop coordinates."""
        h_crop, w_crop = self.slide_size
        h_stride, w_stride = self.slide_stride
        crops = []
        for h_idx in range(
            0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        ):
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

    def _slide_seg_inference(self, x, real=False):
        """Full sliding-window seg pipeline (aligned with opq_release).

        Each crop is CenterPadded to next multiple of patch_size before
        backbone encoding (e.g., 512→518 for patch_size=14). Logits are
        upsampled to the slide_size (crop window size), not the padded size.

        Returns:
            seg_logits: (B, num_classes, H_img, W_img) or None
            total_bits: float (total bits for all crops)
            total_tokens: int (for bpfp)
        """
        B, C, h_img, w_img = x.shape
        assert B == 1, "Sliding window only supports batch size 1"

        crops = self._get_slide_crops(h_img, w_img)
        device = x.device

        seg_head = self.heads["semseg"] if "semseg" in self.heads else None
        if seg_head is None:
            return None, 0.0, 0

        num_classes = seg_head.num_classes
        preds = torch.zeros(1, num_classes, h_img, w_img, device=device)
        count_mat = torch.zeros(1, 1, h_img, w_img, device=device)

        total_bits = 0.0
        total_tokens = 0
        all_labels = []

        for y1, x1, y2, x2 in crops:
            crop_img = x[:, :, y1:y2, x1:x2]
            padded_crop = self._center_pad(crop_img)

            h = self.dino.encode(padded_crop)  # (1, 1+N, D)

            if self.codec_type == "soft_pq":
                h_hat, usage = self._soft_pq_encode_decode(h)
                total_bits += self._soft_pq_actual_bits(h)
            else:
                h_hat, labels = self._orfc_encode_decode(h)
                total_bits += self._orfc_actual_bits(labels)
                all_labels.append(labels)

            n_tokens = h.shape[1]
            total_tokens += n_tokens

            dino = self.dino.model
            x_dec = h_hat
            for blk in dino.blocks[self.dino.slot :]:
                x_dec = blk(x_dec)
            x_dec = dino.norm(x_dec)

            patch_tokens = x_dec[:, 1:, :]  # remove CLS

            actual_h = y2 - y1
            actual_w = x2 - x1
            padded_h = math.ceil(actual_h / self.patch_size) * self.patch_size
            padded_w = math.ceil(actual_w / self.patch_size) * self.patch_size
            feat_h = padded_h // self.patch_size
            feat_w = padded_w // self.patch_size

            patch_2d = rearrange(
                patch_tokens[:, : feat_h * feat_w, :],
                "b (h w) c -> b c h w",
                h=feat_h,
                w=feat_w,
            )

            logits = seg_head.forward([patch_2d])
            logits_up = F.interpolate(
                logits,
                size=(self.slide_size[0], self.slide_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            logits_crop = logits_up[:, :, : actual_h, : actual_w]

            preds[:, :, y1:y2, x1:x2] += logits_crop
            count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0, "count_mat has zeros"
        preds = preds / count_mat

        return preds, total_bits, total_tokens

    def forward_test(self, x, qp=None, tasks=[], **kwargs):
        """Forward pass with sliding window ORFC segmentation."""
        with torch.inference_mode():
            task_feats = {}
            total_bits = 0.0
            total_tokens = 0

            if "semseg" in tasks:
                preds, total_bits, total_tokens = self._slide_seg_inference(x)
                if preds is not None:
                    task_feats["semseg"] = preds

            coded_data = {"bits": {"orfc": total_bits}}
            return coded_data, task_feats

    def compress(self, x, qp=None, tasks=[], **kwargs):
        """Real compression with encoding per crop."""
        with torch.inference_mode():
            if self.codec_type == "soft_pq":
                return self._compress_soft_pq(x, tasks=tasks, **kwargs)
            return self._compress_orfc(x, tasks=tasks, **kwargs)

    def _compress_soft_pq(self, x, tasks=[], **kwargs):
        """SoftPQ compress: slide → codec → forward_test equivalent."""
        preds, total_bits, _ = self._slide_seg_inference(x)
        coded_data = {
            "bits": {"orfc": total_bits},
            "pstate": {"preds": preds},
        }
        return coded_data

    def _compress_orfc(self, x, tasks=[], **kwargs):
        """Standard ORFC compress with rANS encoding per crop."""
        with torch.inference_mode():
            B, C, h_img, w_img = x.shape
            crops = self._get_slide_crops(h_img, w_img)
            device = x.device

            all_byte_strings = []
            all_mu = []
            all_std = []
            all_shapes = []
            total_tokens = 0

            for y1, x1, y2, x2 in crops:
                crop_img = x[:, :, y1:y2, x1:x2]
                padded_crop = self._center_pad(crop_img)
                h = self.dino.encode(padded_crop)
                _B, N, D = h.shape
                total_tokens += N

                Y, mu, std = batch_normalize_gpu(h, mode="per_image")
                flat = Y.reshape(N, D)
                Z = flat @ self._orfc_R
                z_3d = Z.reshape(N, self._orfc_num_groups, self._orfc_embedding_dim)
                z_3d = z_3d.permute(1, 0, 2).contiguous()
                _, labels = batched_assign(z_3d, self._orfc_codebooks, device=device)

                byte_strings = self._orfc_rans_encode(labels)
                all_byte_strings.append(byte_strings)
                all_mu.append(mu.cpu())
                all_std.append(std.cpu())
                all_shapes.append((1, N, D))

            if all(bs is not None for bs in all_byte_strings):
                total_bits = sum(
                    sum(len(s) for s in bs) for bs in all_byte_strings
                ) * 8.0
            else:
                total_bits = float(
                    total_tokens * self._orfc_num_groups * math.log2(self._orfc_K)
                )

            coded_data = {
                "bits": {"orfc": total_bits},
                "pstate": {
                    "crops": crops,
                    "shapes": all_shapes,
                    "mu_list": all_mu,
                    "std_list": all_std,
                    "img_shape": (B, C, h_img, w_img),
                    "byte_strings": all_byte_strings,
                },
            }
            return coded_data

    def decompress(self, coded_unit, tasks=[], **kwargs):
        """Decompress: decode → reconstruct → seg head → fuse."""
        pstate = coded_unit["pstate"]

        if self.codec_type == "soft_pq":
            task_feats = {}
            if "semseg" in tasks and pstate.get("preds") is not None:
                task_feats["semseg"] = pstate["preds"]
            return task_feats

        crops = pstate["crops"]
        shapes = pstate["shapes"]
        mu_list = pstate["mu_list"]
        std_list = pstate["std_list"]
        B, C, h_img, w_img = pstate["img_shape"]
        device = self._orfc_R.device

        all_byte_strings = pstate["byte_strings"]

        seg_head = self.heads["semseg"] if "semseg" in self.heads else None
        task_feats = {}
        if "semseg" not in tasks or seg_head is None:
            return task_feats

        num_classes = seg_head.num_classes
        preds = torch.zeros(1, num_classes, h_img, w_img, device=device)
        count_mat = torch.zeros(1, 1, h_img, w_img, device=device)

        dino = self.dino.model

        for i, (y1, x1, y2, x2) in enumerate(crops):
            _B, N, D = shapes[i]
            mu = mu_list[i].to(device)
            std = std_list[i].to(device)

            byte_strings = all_byte_strings[i]
            labels = self._orfc_rans_decode(byte_strings, N)
            tokens_hat = self._orfc_decode_from_labels(labels, mu, std, 1, N)

            x_dec = tokens_hat
            for blk in dino.blocks[self.dino.slot :]:
                x_dec = blk(x_dec)
            x_dec = dino.norm(x_dec)

            patch_tokens = x_dec[:, 1:, :]

            actual_h = y2 - y1
            actual_w = x2 - x1
            padded_h = math.ceil(actual_h / self.patch_size) * self.patch_size
            padded_w = math.ceil(actual_w / self.patch_size) * self.patch_size
            feat_h = padded_h // self.patch_size
            feat_w = padded_w // self.patch_size

            patch_2d = rearrange(
                patch_tokens[:, : feat_h * feat_w, :],
                "b (h w) c -> b c h w",
                h=feat_h,
                w=feat_w,
            )

            logits = seg_head.forward([patch_2d])
            logits_up = F.interpolate(
                logits,
                size=(self.slide_size[0], self.slide_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            logits_crop = logits_up[:, :, : actual_h, : actual_w]

            preds[:, :, y1:y2, x1:x2] += logits_crop
            count_mat[:, :, y1:y2, x1:x2] += 1

        preds = preds / count_mat
        task_feats["semseg"] = preds
        return task_feats

    def get_feature_numel(self, x):
        """Total compressible feature elements across all crops.

        Computed analytically without running the backbone.
        """
        B, C, h_img, w_img = x.shape
        crops = self._get_slide_crops(h_img, w_img)
        ps = self.patch_size
        embed_dim = self.dino.model.embed_dim
        total = 0
        for y1, x1, y2, x2 in crops:
            actual_h, actual_w = y2 - y1, x2 - x1
            padded_h = math.ceil(actual_h / ps) * ps
            padded_w = math.ceil(actual_w / ps) * ps
            n_tokens = (padded_h // ps) * (padded_w // ps) + 1  # +1 for CLS
            total += B * n_tokens * embed_dim
        return total

"""Soft-PQ (Differentiable Product Quantization) as a latent codec.

Wraps the trained :class:`~cofai.entropy_models.soft_pq.FeatureCodec` as a
latent-codec API component usable by
:class:`~cofai.models.base.DinoFeatureCodecModel` /
:class:`~cofai.models.base.DinoSlideFeatureCodecModel`.

When a sidecar ``.npz`` (same stem as ``codec_path``) provides ``pmf``, labels
are entropy-coded with rANS for real compression and bit accounting (ORFC-style).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cofai.entropy_models.orfc_model import (
    batch_normalize_gpu,
    batch_inv_normalize_gpu,
)
from cofai.entropy_models.soft_pq import load_codec
from cofai.entropy_models.soft_pq_export import try_load_sidecar_pmf


class SoftPQFeatureCodec(nn.Module):
    """Soft-PQ Feature Codec (latent codec API).

    Args:
        codec_path: Path to the ``.pt`` checkpoint saved by ``save_codec``.
        **kwargs: Ignored (config convenience).
    """

    def __init__(self, codec_path: str = "", **kwargs):
        super().__init__()
        self.codec_path = str(codec_path)
        self._codec = load_codec(codec_path, device="cpu")
        self._codec.eval()
        pq = self._codec.pq
        self.G = pq.G
        self.K = pq.K
        self.d = pq.d
        self.feat_dim = int(pq.D)

        pmf = try_load_sidecar_pmf(codec_path)
        if pmf is not None:
            self.register_buffer("pmf", torch.from_numpy(pmf))
        else:
            self.pmf = None

    def _assign_labels(self, h):
        """(B, N, D) tokens -> mu, std, labels (G, B*N)."""
        B, N, D = h.shape
        Y, mu, std = batch_normalize_gpu(h, mode="per_image")
        with torch.no_grad():
            _ = self._codec(Y)
        labels = self._codec.pq._last_labels
        return mu, std, labels

    def _tokens_from_labels(self, labels, mu, std, B, N):
        """Reconstruct (B, N, D) from PQ labels."""
        pq = self._codec.pq
        transform = self._codec.transform
        G, d = self.G, self.d
        device = labels.device

        labels_exp = labels.unsqueeze(-1).expand(G, B * N, d)
        z_hat = torch.gather(pq.codebooks.to(device), 1, labels_exp)
        flat_hat = z_hat.permute(1, 0, 2).reshape(B * N, G * d)

        if transform is not None and hasattr(transform, "get_rotation"):
            R = transform.get_rotation()
            Y_hat = (flat_hat @ R.T).reshape(B, N, -1)
        else:
            Y_hat = transform.decode(flat_hat).reshape(B, N, -1) if transform else flat_hat.reshape(B, N, -1)

        return batch_inv_normalize_gpu(Y_hat, mu, std)

    def _theoretical_bits(self, n_tokens: int) -> float:
        return float(n_tokens * self.G * math.log2(self.K))

    def _actual_bits(self, labels) -> float:
        byte_strings = self._rans_encode(labels)
        if byte_strings is not None:
            return float(sum(len(s) for s in byte_strings) * 8.0)
        return self._theoretical_bits(labels.shape[1])

    def _group_cdf(self, g: int):
        pmf_g = self.pmf[g].cpu().numpy()
        pmf_int = (pmf_g * (1 << 16)).astype(np.int32)
        pmf_int = np.maximum(pmf_int, 1)
        pmf_int[-1] = (1 << 16) - 1 - pmf_int[:-1].sum()
        cdf = np.zeros(self.K + 2, dtype=np.int32)
        cdf[1 : self.K + 1] = np.cumsum(pmf_int)
        cdf[self.K + 1] = 1 << 16
        return cdf.tolist()

    def _rans_encode(self, labels):
        if self.pmf is None:
            return None
        try:
            from compressai.ans import RansEncoder
        except ImportError:
            return None

        G, N = labels.shape
        byte_strings = []
        for g in range(G):
            cdf_list = self._group_cdf(g)
            indices = labels[g].cpu().numpy().astype(np.int32)
            encoder = RansEncoder()
            bs = encoder.encode_with_indexes(
                indices.tolist(), [0] * N, [cdf_list], [self.K + 2], [0]
            )
            byte_strings.append(bs)
        return byte_strings

    def _rans_decode(self, byte_strings, n_tokens: int):
        from compressai.ans import RansDecoder

        G = self.G
        device = next(self.parameters()).device
        all_labels = []
        for g in range(G):
            cdf_list = self._group_cdf(g)
            decoder = RansDecoder()
            decoder.set_stream(byte_strings[g])
            indices = decoder.decode_stream(
                [0] * n_tokens,
                [cdf_list] * n_tokens,
                [self.K + 2] * n_tokens,
                [0] * n_tokens,
            )
            all_labels.append(torch.tensor(indices, dtype=torch.int64, device=device))
        return torch.stack(all_labels)

    def _encode_decode(self, h):
        B, N, D = h.shape
        mu, std, labels = self._assign_labels(h)
        h_hat = self._tokens_from_labels(labels, mu, std, B, N)
        bits = self._actual_bits(labels)
        return h_hat, bits

    def forward(self, h, token_res=None, qp=0, **kwargs):
        h_hat, bits = self._encode_decode(h)
        return {"h_hat": h_hat, "bits": {"soft_pq": bits}}

    def compress(self, h, token_res=None, qp=0, **kwargs):
        B, N, D = h.shape
        mu, std, labels = self._assign_labels(h)
        byte_strings = self._rans_encode(labels)
        if byte_strings is not None:
            return {
                "strings": {"soft_pq": [[bs] for bs in byte_strings]},
                "pstate": {
                    "shape": (int(B), int(N), int(D)),
                    "mu": mu.cpu(),
                    "std": std.cpu(),
                },
            }
        return {
            "bits": {"soft_pq": self._theoretical_bits(B * N)},
            "pstate": {
                "shape": (int(B), int(N), int(D)),
                "mu": mu.cpu(),
                "std": std.cpu(),
                "labels": labels.cpu(),
            },
        }

    def decompress(self, strings=None, pstate=None, **kwargs):
        B, N, D = pstate["shape"]
        device = next(self.parameters()).device
        mu = pstate["mu"].to(device)
        std = pstate["std"].to(device)

        if strings is not None and "soft_pq" in strings:
            byte_strings = [s[0] for s in strings["soft_pq"]]
            labels = self._rans_decode(byte_strings, B * N)
        else:
            labels = pstate["labels"].to(device)

        h_hat = self._tokens_from_labels(labels, mu, std, B, N)
        return {"h_hat": h_hat}

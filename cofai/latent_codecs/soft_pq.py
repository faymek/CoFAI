"""Soft-PQ (Differentiable Product Quantization) as a latent codec.

Wraps the trained :class:`~cofai.entropy_models.soft_pq.FeatureCodec` as a
latent-codec API component usable by
:class:`~cofai.models.base.DinoFeatureCodecModel` /
:class:`~cofai.models.base.DinoSlideFeatureCodecModel`:

* ``forward(h, token_res, qp) -> {"h_hat", "bits"}``
* ``compress(h, token_res, qp) -> {"bits", "pstate"}``
* ``decompress(strings, pstate) -> {"h_hat"}``

The pipeline mirrors ``_SoftPQMixin``: per-image normalize → FeatureCodec
(OrthogonalTransform + SoftPQ hard assignment) → denormalize.
"""

import math

import torch
import torch.nn as nn

from cofai.entropy_models.orfc_model import (
    batch_normalize_gpu,
    batch_inv_normalize_gpu,
)
from cofai.entropy_models.soft_pq import load_codec


class SoftPQFeatureCodec(nn.Module):
    """Soft-PQ Feature Codec (latent codec API).

    Loads a trained FeatureCodec checkpoint (.pt) and exposes the
    standard latent-codec interface.

    Args:
        codec_path (str): Path to the ``.pt`` checkpoint saved by
            :func:`~cofai.entropy_models.soft_pq.save_codec`.
        **kwargs: Ignored (accepted for config convenience).
    """

    def __init__(self, codec_path: str = "", **kwargs):
        super().__init__()
        self._codec = load_codec(codec_path, device="cpu")
        self._codec.eval()
        pq = self._codec.pq
        self.G = pq.G
        self.K = pq.K
        self.d = pq.d

    def _encode_decode(self, h):
        """Normalize → FeatureCodec → denormalize.

        Returns:
            h_hat: (B, N, D) reconstructed tokens
            bits: float, estimated bits for the batch
        """
        B, N, D = h.shape
        Y, mu, std = batch_normalize_gpu(h, mode="per_image")
        Y_hat, usage = self._codec(Y)
        h_hat = batch_inv_normalize_gpu(Y_hat, mu, std)

        pq = self._codec.pq
        if pq.use_rate and pq._last_rate is not None:
            bits = pq._last_rate.item() * N
        else:
            bits = float(N * self.G * math.log2(self.K))

        return h_hat, bits

    # ------------------------------------------------------------------ #
    # Latent-codec API
    # ------------------------------------------------------------------ #
    def forward(self, h, token_res=None, qp=0, **kwargs):
        h_hat, bits = self._encode_decode(h)
        return {"h_hat": h_hat, "bits": {"soft_pq": bits}}

    def compress(self, h, token_res=None, qp=0, **kwargs):
        B, N, D = h.shape
        Y, mu, std = batch_normalize_gpu(h, mode="per_image")

        pq = self._codec.pq
        transform = self._codec.transform

        flat = Y.reshape(B * N, D)
        R = transform.get_rotation()
        Z_flat = flat @ R  # (B*N, D)

        # Use the same quantisation logic as forward (handles use_rate RD cost)
        with torch.no_grad():
            pq._quantise(Z_flat)
        labels = pq._last_labels  # (G, B*N)

        if pq.use_rate and pq._last_rate is not None:
            bits = pq._last_rate.item() * N
        else:
            bits = float(B * N * self.G * math.log2(self.K))

        return {
            "bits": {"soft_pq": bits},
            "pstate": {
                "shape": (int(B), int(N), int(D)),
                "mu": mu.cpu(),
                "std": std.cpu(),
                "labels": labels.cpu(),  # (G, B*N)
            },
        }

    def decompress(self, strings=None, pstate=None, **kwargs):
        B, N, D = pstate["shape"]
        device = next(self.parameters()).device
        mu = pstate["mu"].to(device)
        std = pstate["std"].to(device)
        labels = pstate["labels"].to(device)  # (G, B*N)

        pq = self._codec.pq
        transform = self._codec.transform
        cb = pq.codebooks  # (G, K, d)

        labels_3d = labels.unsqueeze(-1).expand(self.G, B * N, self.d)
        z_hat = torch.gather(cb, 1, labels_3d)  # (G, B*N, d)
        flat_hat = z_hat.permute(1, 0, 2).reshape(B * N, D)

        R = transform.get_rotation()
        Y_hat = (flat_hat @ R.T).reshape(B, N, D)
        h_hat = batch_inv_normalize_gpu(Y_hat, mu, std)

        return {"h_hat": h_hat}

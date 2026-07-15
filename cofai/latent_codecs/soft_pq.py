"""Soft-PQ as a latent codec loaded from a single ORFC-compatible ``.npz``.

Canonical weight file fields: ``R`` (D, D), ``codebooks`` (G, K, d), optional
``pmf`` (G, K), optional ``norm_mode`` / ``n_prefix``.

Quantization: normalize -> rotate by ``R`` -> product-quantize -> inverse rotate
-> denormalize. When ``pmf`` is present, labels are entropy-coded with rANS.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cofai.entropy_models.orfc_model import (
    batch_normalize_gpu,
    batch_inv_normalize_gpu,
    batched_assign,
)
from cofai.entropy_models.soft_pq_export import load_softpq_npz


class SoftPQFeatureCodec(nn.Module):
    """Soft-PQ Feature Codec (latent codec API).

    Args:
        codec_path: Path to SoftPQ ``.npz`` (``R``, ``codebooks``, optional ``pmf``).
        norm_mode: Override npz meta (``per_image`` | ``split_cls_patch`` |
            ``split_reg_cls_patch``). Empty / None -> use npz or ``per_image``.
        n_prefix: Override npz meta prefix length for split_* norms.
        **kwargs: Also accepts ``orfc_weights_path`` as alias of ``codec_path``.
    """

    def __init__(
        self,
        codec_path: str = "",
        norm_mode: Optional[str] = None,
        n_prefix: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        path = codec_path or kwargs.get("orfc_weights_path", "")
        if not path:
            raise ValueError("SoftPQFeatureCodec requires codec_path (.npz)")
        if not str(path).endswith(".npz"):
            # Allow passing a .pt path only if sibling .npz exists (migration aid).
            from pathlib import Path
            from cofai.entropy_models.soft_pq_export import npz_path_for_codec

            npz = npz_path_for_codec(path)
            if not npz.is_file():
                raise ValueError(
                    f"SoftPQFeatureCodec expects a .npz weight file; got {path}. "
                    f"Export with export_softpq_npz / train pipeline first."
                )
            path = str(npz)

        self.codec_path = str(path)
        payload = load_softpq_npz(path)

        self.K = int(payload["K"])
        self.embedding_dim = int(payload["embedding_dim"])
        self.G = int(payload["G"])
        self.feat_dim = int(payload["feat_dim"])
        self.num_groups = self.G

        meta_norm = payload["norm_mode"] or "per_image"
        meta_prefix = int(payload["n_prefix"])
        if norm_mode is not None and str(norm_mode).strip():
            self.norm_mode = str(norm_mode)
        else:
            self.norm_mode = meta_norm
        if n_prefix is not None:
            self.n_prefix = int(n_prefix)
        else:
            self.n_prefix = meta_prefix

        self.register_buffer("R", torch.from_numpy(payload["R"]).float())
        self.register_buffer("codebooks", torch.from_numpy(payload["codebooks"]).float())
        if payload["pmf"] is not None:
            self.register_buffer("pmf", torch.from_numpy(payload["pmf"]))
        else:
            self.pmf = None

    def _assign_labels(self, h):
        """(B, N, D) tokens -> mu, std, labels (G, B*N)."""
        B, N, D = h.shape
        Y, mu, std = batch_normalize_gpu(
            h, mode=self.norm_mode, n_prefix=self.n_prefix
        )
        flat = Y.reshape(B * N, D)
        Z = flat @ self.R
        z_3d = Z.reshape(B * N, self.num_groups, self.embedding_dim)
        z_3d = z_3d.permute(1, 0, 2).contiguous()
        _, labels = batched_assign(z_3d, self.codebooks, device=h.device)
        return mu, std, labels

    def _tokens_from_labels(self, labels, mu, std, B, N):
        """Reconstruct (B, N, D) from PQ labels."""
        D = self.feat_dim
        G = self.num_groups
        d = self.embedding_dim
        labels_exp = labels.unsqueeze(-1).expand(G, B * N, d)
        z_hat_3d = torch.gather(self.codebooks, 1, labels_exp)
        flat_hat = z_hat_3d.permute(1, 0, 2).reshape(B * N, D)
        Y_hat = (flat_hat @ self.R.T).reshape(B, N, D)
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
        device = self.R.device
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
        device = self.R.device
        mu = pstate["mu"].to(device)
        std = pstate["std"].to(device)

        if strings is not None and "soft_pq" in strings:
            byte_strings = [s[0] for s in strings["soft_pq"]]
            labels = self._rans_decode(byte_strings, B * N)
        else:
            labels = pstate["labels"].to(device)

        h_hat = self._tokens_from_labels(labels, mu, std, B, N)
        return {"h_hat": h_hat}

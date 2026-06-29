"""ORFC (Orthogonal Rotation Feature Codec) as a latent codec.

This adapts the ORFC quantizer used by the bespoke ``Dinov2ClsORFC`` /
``Dinov2SlideSegORFC`` models to the latent-codec API expected by
:class:`~cofai.models.base.DinoFeatureCodecModel` /
:class:`~cofai.models.base.DinoSlideFeatureCodecModel`:

* ``forward(h, token_res, qp) -> {"h_hat", "bits"}``
* ``compress(h, token_res, qp) -> {"strings", "pstate"}``
* ``decompress(strings, pstate) -> {"h_hat"}``

operating on encoder token tensors of shape ``(B, N, C)`` (cls / register
prefix included -- ORFC compresses the whole token sequence, mirroring the
bespoke models which call ``_orfc_encode_decode`` on the full ``h``).

The quantization pipeline is identical to ``_ORFCMixin`` (per-image normalize ->
rotate by ``R`` -> product-quantize against ``codebooks`` -> inverse rotate ->
denormalize). When the loaded ``.npz`` provides a ``pmf``, labels are entropy
coded with rANS for real compression; otherwise the fixed-rate
``num_groups * log2(K)`` bits/token is reported.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from cofai.entropy_models.orfc_model import (
    batch_normalize_gpu,
    batch_inv_normalize_gpu,
    batched_assign,
)


class OrthoRotationFeatureCodec(nn.Module):
    """ORFC (Orthogonal Rotation Feature Codec) .

    ORFC quantizer wrapped as a feature latent codec.

    Args:
        orfc_weights_path (str): Path to the ``.npz`` file with ``R`` (D, D),
            ``codebooks`` (G, K, d) and optional ``pmf`` (G, K).
        K (int): Codebook size per group.
        embedding_dim (int): Sub-vector dimension ``d`` (``G = D // d``).
        **kwargs: Ignored (accepted for config convenience).
    """

    def __init__(
        self,
        orfc_weights_path: str = "",
        K: int = 256,
        embedding_dim: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.K = int(K)
        self.embedding_dim = int(embedding_dim)

        data = np.load(orfc_weights_path, allow_pickle=True)
        R = data["R"]  # (D, D)
        codebooks = data["codebooks"]  # (G, K, d)

        D = R.shape[0]
        self.feat_dim = int(D)
        self.num_groups = int(D) // self.embedding_dim

        self.register_buffer("R", torch.from_numpy(R).float())
        self.register_buffer("codebooks", torch.from_numpy(codebooks).float())
        if "pmf" in data:
            self.register_buffer(
                "pmf", torch.from_numpy(data["pmf"].astype(np.float32))
            )
        else:
            self.pmf = None

    # ------------------------------------------------------------------ #
    # Core ORFC quantization
    # ------------------------------------------------------------------ #
    def _assign_labels(self, tokens):
        """(B, N, D) tokens -> (per-image mu/std, PQ labels (G, B*N))."""
        B, N, D = tokens.shape
        Y, mu, std = batch_normalize_gpu(tokens, mode="per_image")
        flat = Y.reshape(B * N, D)
        Z = flat @ self.R
        z_3d = Z.reshape(B * N, self.num_groups, self.embedding_dim)
        z_3d = z_3d.permute(1, 0, 2).contiguous()  # (G, B*N, d)
        _, labels = batched_assign(z_3d, self.codebooks, device=tokens.device)
        return mu, std, labels

    def _tokens_from_labels(self, labels, mu, std, B, N):
        """PQ labels + normalization stats -> reconstructed (B, N, D) tokens."""
        D = self.feat_dim
        G = self.num_groups
        d = self.embedding_dim
        labels_exp = labels.unsqueeze(-1).expand(G, B * N, d)
        z_hat_3d = torch.gather(self.codebooks, 1, labels_exp)  # (G, B*N, d)
        flat_hat = z_hat_3d.permute(1, 0, 2).reshape(B * N, D)
        Y_hat = (flat_hat @ self.R.T).reshape(B, N, D)
        return batch_inv_normalize_gpu(Y_hat, mu, std)

    def _theoretical_bits(self, n_tokens: int) -> float:
        return float(n_tokens * self.num_groups * math.log2(self.K))

    def _actual_bits(self, labels) -> float:
        byte_strings = self._rans_encode(labels)
        if byte_strings is not None:
            return float(sum(len(s) for s in byte_strings) * 8.0)
        return self._theoretical_bits(labels.shape[1])

    # ------------------------------------------------------------------ #
    # rANS entropy coding (optional, requires stored pmf + compressai)
    # ------------------------------------------------------------------ #
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

        G = self.num_groups
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
        return torch.stack(all_labels)  # (G, n_tokens)

    # ------------------------------------------------------------------ #
    # Latent-codec API
    # ------------------------------------------------------------------ #
    def forward(self, h, token_res=None, qp=0, **kwargs):
        B, N, _ = h.shape
        mu, std, labels = self._assign_labels(h)
        h_hat = self._tokens_from_labels(labels, mu, std, B, N)
        return {"h_hat": h_hat, "bits": {"orfc": self._actual_bits(labels)}}

    def compress(self, h, token_res=None, qp=0, **kwargs):
        B, N, D = h.shape
        mu, std, labels = self._assign_labels(h)
        byte_strings = self._rans_encode(labels)
        if byte_strings is not None:
            return {
                "strings": {"orfc": [[bs] for bs in byte_strings]},
                "pstate": {
                    "shape": (int(B), int(N), int(D)),
                    "mu": mu.cpu(),
                    "std": std.cpu(),
                },
            }
        # No entropy model available: fall back to fixed-rate bits and keep the
        # raw labels so decompression can still reconstruct exactly.
        return {
            "bits": {"orfc": self._theoretical_bits(N)},
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

        if strings is not None and "orfc" in strings:
            byte_strings = [s[0] for s in strings["orfc"]]
            labels = self._rans_decode(byte_strings, B * N)
        else:
            labels = pstate["labels"].to(device)

        h_hat = self._tokens_from_labels(labels, mu, std, B, N)
        return {"h_hat": h_hat}

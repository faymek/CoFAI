"""CTC runtime for the saved PQFCv2 OPQ initialization.

The rotation and codebooks come from the OPQ artifact. RMS is the fixed
training-set statistic, so the bitstream contains only the PQ indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cofai.entropy_models.static_categorical import StaticCategoricalEntropyModel
from cofai.ops.orfc import batched_assign

from examples.pqfcv2.train_stage1 import apply_rms, invert_rms


def _load_pmf(usage: np.ndarray) -> torch.Tensor:
    counts = np.asarray(usage, dtype=np.float64)
    if counts.ndim != 2 or np.any(counts < 0) or np.any(counts.sum(axis=1) <= 0):
        raise ValueError(f"OPQ usage must be non-negative with positive group mass, got {counts.shape}")
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    pmf = torch.from_numpy(probabilities.astype(np.float32))
    return pmf / pmf.sum(dim=1, keepdim=True)


class PqfCv2OpqCodec(nn.Module):
    """Hard PQ / ECVQ with the OPQ rotation, fixed RMS, and a static rANS prior."""

    def __init__(self, opq_path: str, rms_path: str, lmbda: float | None = None):
        super().__init__()
        opq_file = Path(opq_path)
        rms_file = Path(rms_path)
        npz_lmbda: float | None = None
        with np.load(opq_file, allow_pickle=False) as data:
            rotation = np.asarray(data["R"], dtype=np.float32)
            codebooks = np.asarray(data["codebooks"], dtype=np.float32)
            if "pmf" in data.files:
                probabilities = np.asarray(data["pmf"], dtype=np.float64)
            elif "usage" in data.files:
                probabilities = np.asarray(data["usage"], dtype=np.float64)
            else:
                raise ValueError(f"PQFCv2 artifact {opq_file} has neither pmf nor usage")
            if "lmbda" in data.files:
                npz_lmbda = float(np.asarray(data["lmbda"]).item())
        with np.load(rms_file, allow_pickle=False) as data:
            patch_rms = np.asarray(data["patch_channel_rms"], dtype=np.float32)
            if "prefix_position_channel_rms" in data.files:
                prefix_rms = np.asarray(data["prefix_position_channel_rms"], dtype=np.float32)
            else:
                prefix_rms = np.zeros((0, patch_rms.size), dtype=np.float32)

        if codebooks.ndim != 3:
            raise ValueError(f"OPQ codebooks must be [groups, entries, dim], got {codebooks.shape}")
        groups, _entries, embedding_dim = codebooks.shape
        feature_dim = groups * embedding_dim
        if rotation.shape != (feature_dim, feature_dim):
            raise ValueError(f"OPQ rotation shape {rotation.shape} does not match D={feature_dim}")
        if probabilities.shape[:2] != codebooks.shape[:2]:
            raise ValueError(
                f"OPQ probability shape {probabilities.shape} does not match codebooks {codebooks.shape[:2]}"
            )
        probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        if prefix_rms.ndim != 2 or patch_rms.shape != (feature_dim,):
            raise ValueError(f"RMS shapes prefix={prefix_rms.shape} patch={patch_rms.shape} are incompatible with D={feature_dim}")
        if not np.isfinite(rotation).all() or not np.isfinite(codebooks).all():
            raise ValueError(f"OPQ artifact contains non-finite values: {opq_file}")

        self.n_prefix = int(prefix_rms.shape[0])
        self.num_groups = int(groups)
        self.embedding_dim = int(embedding_dim)
        self.feat_dim = int(feature_dim)
        resolved_lmbda = npz_lmbda if lmbda is None else float(lmbda)
        self.lmbda = None if resolved_lmbda is None or resolved_lmbda <= 0 else float(resolved_lmbda)
        self.register_buffer("R", torch.from_numpy(rotation))
        self.register_buffer("codebooks", torch.from_numpy(codebooks))
        self.register_buffer("prefix_rms", torch.from_numpy(prefix_rms))
        self.register_buffer("patch_rms", torch.from_numpy(patch_rms))
        self.entropy_model = StaticCategoricalEntropyModel(_load_pmf(probabilities))
        label_total = float((1 << self.entropy_model.range_coder_precision) - 1)
        rans_bits = np.stack(
            [
                -np.log2(np.diff(np.asarray(cdf[:-1], dtype=np.float64)) / label_total)
                for cdf in self.entropy_model._cdfs
            ],
            axis=0,
        ).astype(np.float32)
        self.register_buffer("rans_bits", torch.from_numpy(rans_bits))

    def _labels(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, feature_dim = tokens.shape
        if feature_dim != self.feat_dim:
            raise ValueError(f"PQFCv2 expected feature dimension {self.feat_dim}, got {feature_dim}")
        if n_tokens <= self.n_prefix:
            raise ValueError(f"PQFCv2 expected more than {self.n_prefix} tokens, got {n_tokens}")
        normalized = apply_rms(tokens.float(), self.prefix_rms, self.patch_rms, self.n_prefix)
        flat = normalized.reshape(batch_size * n_tokens, feature_dim)
        rotated = flat @ self.R
        sub_vectors = rotated.reshape(batch_size * n_tokens, self.num_groups, self.embedding_dim)
        sub_vectors = sub_vectors.permute(1, 0, 2).contiguous()
        if self.lmbda is None:
            _reconstructed, labels = batched_assign(sub_vectors, self.codebooks, device=tokens.device)
            return labels
        # ECVQ assignment matching SoftPQ: argmin_k (||z - c_k||^2 + R_rANS(k) / lmbda).
        penalty = (self.rans_bits / self.lmbda).unsqueeze(1)
        total_tokens = sub_vectors.shape[1]
        chunk_size = 4096
        if total_tokens <= chunk_size:
            return (torch.cdist(sub_vectors, self.codebooks).pow(2) + penalty).argmin(dim=-1)
        chunks = [
            (torch.cdist(sub_vectors[:, start : start + chunk_size, :], self.codebooks).pow(2) + penalty).argmin(dim=-1)
            for start in range(0, total_tokens, chunk_size)
        ]
        return torch.cat(chunks, dim=1)

    def _tokens_from_labels(self, labels: torch.Tensor, batch_size: int, n_tokens: int) -> torch.Tensor:
        label_index = labels.unsqueeze(-1).expand(self.num_groups, batch_size * n_tokens, self.embedding_dim)
        quantized = torch.gather(self.codebooks, 1, label_index)
        rotated_hat = quantized.permute(1, 0, 2).reshape(batch_size * n_tokens, self.feat_dim)
        normalized_hat = (rotated_hat @ self.R.T).reshape(batch_size, n_tokens, self.feat_dim)
        return invert_rms(normalized_hat, self.prefix_rms, self.patch_rms, self.n_prefix, 0)

    def forward(self, h, token_res=None, qp=0, **kwargs):
        batch_size, n_tokens, _feature_dim = h.shape
        labels = self._labels(h)
        return {
            "h_hat": self._tokens_from_labels(labels, batch_size, n_tokens),
            "likelihoods": {"pq": self.entropy_model(labels)},
        }

    def compress(self, h, token_res=None, qp=0, **kwargs):
        batch_size, n_tokens, feature_dim = h.shape
        labels = self._labels(h)
        streams = self.entropy_model.compress(labels)
        return {
            "strings": {"pq": [[stream] for stream in streams]},
            "pstate": {"shape": (int(batch_size), int(n_tokens), int(feature_dim))},
        }

    def decompress(self, strings, pstate, **kwargs):
        batch_size, n_tokens, feature_dim = map(int, pstate["shape"])
        if feature_dim != self.feat_dim:
            raise ValueError(f"PQFCv2 stream feature dimension is {feature_dim}; expected {self.feat_dim}")
        try:
            label_streams = [stream[0] for stream in strings["pq"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("PQFCv2 coded unit is missing PQ index streams") from exc
        labels = self.entropy_model.decompress(label_streams, num_samples=batch_size * n_tokens)
        return {"h_hat": self._tokens_from_labels(labels, batch_size, n_tokens)}

"""Orthogonal-rotation product quantization for DINO feature tokens.

The runtime codec consumes a released ``.npz`` artifact containing ``R``,
``codebooks`` and ``pmf``. Training and artifact export live with the proposal
under ``examples/orfc_2446/offline``; this module only implements the stable
LatentCodec contract used by the evaluation engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cofai.entropy_models.static_categorical import StaticCategoricalEntropyModel
from cofai.ops.orfc import batched_assign

from .orfc_normalization import (
    ORFC_NORM_MODES,
    denormalize_orfc_features,
    normalize_orfc_features,
)

_STATS_DTYPE = np.dtype("<f4")


def _scalar(data: Any, key: str, default: Any) -> Any:
    if key not in data:
        return default
    value = data[key]
    return value.item() if hasattr(value, "item") else value


def load_orfc_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate one released ORFC ``.npz`` artifact."""
    artifact_path = Path(path)
    if artifact_path.suffix.lower() != ".npz":
        raise ValueError(f"ORFC requires a .npz artifact, got: {artifact_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"ORFC artifact not found: {artifact_path}")

    with np.load(artifact_path, allow_pickle=False) as data:
        required = {"R", "codebooks", "pmf"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(
                f"ORFC artifact is missing required arrays {missing}: {artifact_path}"
            )

        rotation = np.asarray(data["R"], dtype=np.float32)
        codebooks = np.asarray(data["codebooks"], dtype=np.float32)
        pmf = np.asarray(data["pmf"], dtype=np.float32)
        norm_mode = str(_scalar(data, "norm_mode", "per_image"))
        n_prefix = int(_scalar(data, "n_prefix", 0))

    if codebooks.ndim != 3:
        raise ValueError(
            f"ORFC codebooks must have shape (groups, entries, dim), got {codebooks.shape}"
        )
    groups, entries, embedding_dim = codebooks.shape
    feature_dim = groups * embedding_dim
    if rotation.shape != (feature_dim, feature_dim):
        raise ValueError(
            f"ORFC rotation has shape {rotation.shape}; expected {(feature_dim, feature_dim)}"
        )
    if pmf.shape != (groups, entries):
        raise ValueError(
            f"ORFC PMF has shape {pmf.shape}; expected {(groups, entries)}"
        )
    if norm_mode not in ORFC_NORM_MODES:
        raise ValueError(f"Unsupported ORFC normalization mode: {norm_mode!r}")
    if n_prefix < 0:
        raise ValueError(f"ORFC n_prefix must be non-negative, got {n_prefix}")
    if not all(np.all(np.isfinite(array)) for array in (rotation, codebooks, pmf)):
        raise ValueError(f"ORFC artifact contains non-finite values: {artifact_path}")
    if np.any(pmf < 0) or np.any(pmf.sum(axis=1) <= 0):
        raise ValueError(
            f"ORFC PMF must be non-negative with positive group mass: {artifact_path}"
        )

    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    return {
        "R": rotation,
        "codebooks": codebooks,
        "pmf": pmf.astype(np.float32),
        "norm_mode": norm_mode,
        "n_prefix": n_prefix,
        "groups": int(groups),
        "entries": int(entries),
        "embedding_dim": int(embedding_dim),
        "feature_dim": int(feature_dim),
        "path": str(artifact_path),
    }


class OrthoRotationFeatureCodec(nn.Module):
    """Fixed ORFC/Soft-PQ artifact exposed through the LatentCodec API.

    ``K`` and ``embedding_dim`` are optional artifact assertions retained by
    existing ORFC plans. New plans only need ``orfc_weights_path``.
    """

    def __init__(
        self,
        orfc_weights_path: str,
        K: int | None = None,
        embedding_dim: int | None = None,
        norm_mode: str | None = None,
        n_prefix: int | None = None,
    ):
        super().__init__()
        artifact = load_orfc_artifact(orfc_weights_path)

        self.orfc_weights_path = str(orfc_weights_path)
        self.K = artifact["entries"]
        self.embedding_dim = artifact["embedding_dim"]
        self.num_groups = artifact["groups"]
        self.G = self.num_groups
        self.feat_dim = artifact["feature_dim"]
        self.norm_mode = str(norm_mode or artifact["norm_mode"])
        self.n_prefix = artifact["n_prefix"] if n_prefix is None else int(n_prefix)

        if K is not None and int(K) != self.K:
            raise ValueError(f"Configured K={K} does not match artifact K={self.K}")
        if embedding_dim is not None and int(embedding_dim) != self.embedding_dim:
            raise ValueError(
                "Configured embedding_dim="
                f"{embedding_dim} does not match artifact embedding_dim={self.embedding_dim}"
            )
        if self.norm_mode not in ORFC_NORM_MODES:
            raise ValueError(f"Unsupported ORFC normalization mode: {self.norm_mode!r}")
        if self.n_prefix < 0:
            raise ValueError(f"ORFC n_prefix must be non-negative, got {self.n_prefix}")

        self.register_buffer("R", torch.from_numpy(artifact["R"]))
        self.register_buffer("codebooks", torch.from_numpy(artifact["codebooks"]))
        self.entropy_model = StaticCategoricalEntropyModel(
            torch.from_numpy(artifact["pmf"])
        )

    def _assign_labels(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_tokens, feature_dim = tokens.shape
        if feature_dim != self.feat_dim:
            raise ValueError(
                f"ORFC expected feature dimension {self.feat_dim}, got {feature_dim}"
            )

        normalized, mean, std = normalize_orfc_features(
            tokens,
            mode=self.norm_mode,
            n_prefix=self.n_prefix,
        )
        rotated = normalized.reshape(batch_size * n_tokens, feature_dim) @ self.R
        sub_vectors = rotated.reshape(
            batch_size * n_tokens,
            self.num_groups,
            self.embedding_dim,
        )
        sub_vectors = sub_vectors.permute(1, 0, 2).contiguous()
        _, labels = batched_assign(sub_vectors, self.codebooks, device=tokens.device)
        return mean, std, labels

    def _tokens_from_labels(
        self,
        labels: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        batch_size: int,
        n_tokens: int,
    ) -> torch.Tensor:
        label_index = labels.unsqueeze(-1).expand(
            self.num_groups,
            batch_size * n_tokens,
            self.embedding_dim,
        )
        quantized = torch.gather(self.codebooks, 1, label_index)
        rotated_hat = quantized.permute(1, 0, 2).reshape(
            batch_size * n_tokens,
            self.feat_dim,
        )
        normalized_hat = (rotated_hat @ self.R.T).reshape(
            batch_size,
            n_tokens,
            self.feat_dim,
        )
        return denormalize_orfc_features(
            normalized_hat,
            mean,
            std,
            mode=self.norm_mode,
            n_prefix=self.n_prefix,
        )

    @staticmethod
    def _serialize_stats(mean: torch.Tensor, std: torch.Tensor) -> bytes:
        stats = torch.stack((mean, std), dim=0).detach().cpu().float().numpy()
        return np.asarray(stats, dtype=_STATS_DTYPE).tobytes(order="C")

    def _deserialize_stats(
        self,
        payload: bytes,
        stats_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_values = 2 * int(np.prod(stats_shape))
        values = np.frombuffer(payload, dtype=_STATS_DTYPE)
        if values.size != expected_values:
            raise ValueError(
                f"ORFC normalization stream has {values.size} floats; expected {expected_values}"
            )
        stats = (
            torch.from_numpy(values.copy()).reshape(2, *stats_shape).to(self.R.device)
        )
        return stats[0], stats[1]

    def forward(self, h, token_res=None, qp=0, **kwargs):
        batch_size, n_tokens, _ = h.shape
        mean, std, labels = self._assign_labels(h)
        h_hat = self._tokens_from_labels(labels, mean, std, batch_size, n_tokens)
        stats_likelihoods = torch.full(
            (mean.numel() + std.numel(),),
            2.0**-32,
            dtype=torch.float32,
            device=h.device,
        )
        return {
            "h_hat": h_hat,
            "likelihoods": {
                "orfc": self.entropy_model(labels),
                "orfc_stats": stats_likelihoods,
            },
        }

    def compress(self, h, token_res=None, qp=0, **kwargs):
        batch_size, n_tokens, feature_dim = h.shape
        mean, std, labels = self._assign_labels(h)
        streams = self.entropy_model.compress(labels)
        return {
            "strings": {
                "orfc": [[stream] for stream in streams],
                "orfc_stats": [[self._serialize_stats(mean, std)]],
            },
            "pstate": {
                "shape": (int(batch_size), int(n_tokens), int(feature_dim)),
                "stats_shape": tuple(int(value) for value in mean.shape),
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        batch_size, n_tokens, feature_dim = map(int, pstate["shape"])
        if feature_dim != self.feat_dim:
            raise ValueError(
                f"ORFC stream feature dimension is {feature_dim}; expected {self.feat_dim}"
            )
        try:
            label_streams = [stream[0] for stream in strings["orfc"]]
            stats_stream = strings["orfc_stats"][0][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "ORFC coded unit is missing label or normalization streams"
            ) from exc

        labels = self.entropy_model.decompress(
            label_streams,
            num_samples=batch_size * n_tokens,
        )
        mean, std = self._deserialize_stats(
            stats_stream,
            tuple(int(value) for value in pstate["stats_shape"]),
        )
        h_hat = self._tokens_from_labels(labels, mean, std, batch_size, n_tokens)
        return {"h_hat": h_hat}

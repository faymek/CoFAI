"""Static grouped categorical entropy model backed by rANS."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


__all__ = ["StaticCategoricalEntropyModel"]


def _pmf_to_rans_cdf(
    pmf: np.ndarray,
    *,
    precision: int,
) -> list[int]:
    """Convert one PMF row to a strictly increasing rANS CDF."""
    probabilities = np.asarray(pmf, dtype=np.float64)
    alphabet_size = int(probabilities.size)
    total = 1 << precision
    if not 0 < alphabet_size < total:
        raise ValueError(
            f"Expected between 1 and {total - 1} symbols, got {alphabet_size}"
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Categorical PMF must be finite and non-negative")

    probability_sum = float(probabilities.sum())
    if probability_sum <= 0:
        raise ValueError("Categorical PMF must have positive mass")

    label_total = total - 1
    remaining = label_total - alphabet_size
    scaled = probabilities / probability_sum * remaining
    extra_counts = np.floor(scaled).astype(np.int64)
    frequencies = extra_counts + 1
    missing = remaining - int(extra_counts.sum())
    if missing:
        residuals = scaled - extra_counts
        frequencies[np.argsort(-residuals, kind="stable")[:missing]] += 1

    cdf = np.empty(alphabet_size + 2, dtype=np.int64)
    cdf[0] = 0
    cdf[1 : alphabet_size + 1] = np.cumsum(frequencies)
    cdf[-1] = total
    if cdf[-2] != label_total or np.any(np.diff(cdf) <= 0):
        raise RuntimeError("PMF quantization produced an invalid rANS CDF")
    return cdf.astype(np.int32).tolist()


class StaticCategoricalEntropyModel(nn.Module):
    """Encode grouped categorical labels using one fixed PMF per group."""

    def __init__(
        self,
        pmf: torch.Tensor,
        *,
        range_coder_precision: int = 16,
    ) -> None:
        super().__init__()
        probabilities = torch.as_tensor(pmf, dtype=torch.float32).detach().clone()
        if probabilities.ndim != 2:
            raise ValueError(
                "Static categorical PMF must have shape (groups, symbols), "
                f"got {tuple(probabilities.shape)}"
            )
        if not torch.all(torch.isfinite(probabilities)) or torch.any(probabilities < 0):
            raise ValueError("Static categorical PMF must be finite and non-negative")
        row_mass = probabilities.sum(dim=1)
        if torch.any(row_mass <= 0):
            raise ValueError("Every static categorical PMF row must have positive mass")
        if not torch.allclose(
            row_mass,
            torch.ones_like(row_mass),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("Every static categorical PMF row must sum to one")
        if range_coder_precision <= 0:
            raise ValueError("range_coder_precision must be positive")

        self.range_coder_precision = int(range_coder_precision)
        self.num_groups, self.num_symbols = map(int, probabilities.shape)
        self.register_buffer("pmf", probabilities)
        self._cdfs = [
            _pmf_to_rans_cdf(
                probabilities[group].cpu().numpy(),
                precision=self.range_coder_precision,
            )
            for group in range(self.num_groups)
        ]

    def _validate_labels(self, labels: torch.Tensor) -> None:
        if labels.ndim != 2 or labels.shape[0] != self.num_groups:
            raise ValueError(
                "Static categorical labels must have shape (groups, samples); "
                f"expected {self.num_groups} groups, got {tuple(labels.shape)}"
            )

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Return one probability for each grouped categorical label."""
        self._validate_labels(labels)
        return torch.gather(self.pmf, 1, labels).clamp_min(
            torch.finfo(self.pmf.dtype).tiny
        )

    def compress(self, labels: torch.Tensor) -> list[bytes]:
        """Encode ``[groups, samples]`` labels into one rANS stream per group."""
        self._validate_labels(labels)
        try:
            from compressai.ans import RansEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Categorical compression requires compressai.ans"
            ) from exc

        streams: list[bytes] = []
        num_samples = int(labels.shape[1])
        for group, cdf in enumerate(self._cdfs):
            symbols = labels[group].detach().cpu().numpy().astype(np.int32).tolist()
            streams.append(
                RansEncoder().encode_with_indexes(
                    symbols,
                    [0] * num_samples,
                    [cdf],
                    [self.num_symbols + 2],
                    [0],
                )
            )
        return streams

    def decompress(
        self,
        streams: list[bytes],
        *,
        num_samples: int,
    ) -> torch.Tensor:
        """Decode one rANS stream per group into ``[groups, samples]`` labels."""
        try:
            from compressai.ans import RansDecoder
        except ImportError as exc:
            raise RuntimeError(
                "Categorical decompression requires compressai.ans"
            ) from exc
        if len(streams) != self.num_groups:
            raise ValueError(
                f"Expected {self.num_groups} categorical streams, got {len(streams)}"
            )
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")

        labels = []
        for stream, cdf in zip(streams, self._cdfs):
            decoder = RansDecoder()
            decoder.set_stream(stream)
            symbols = decoder.decode_stream(
                [0] * num_samples,
                [cdf] * num_samples,
                [self.num_symbols + 2] * num_samples,
                [0] * num_samples,
            )
            labels.append(
                torch.tensor(symbols, dtype=torch.int64, device=self.pmf.device)
            )
        return torch.stack(labels)

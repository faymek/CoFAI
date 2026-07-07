"""Soft-PQ sidecar ``.npz`` export (ORFC-compatible PMF for rANS).

Exports ``R``, ``codebooks``, and train-set histogram ``pmf`` from a trained
``.pt`` FeatureCodec checkpoint.  Online (:class:`~cofai.latent_codecs.soft_pq.SoftPQFeatureCodec`)
and offline tests load the sidecar for real rANS bit counting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cofai.entropy_models.orfc_model import batch_normalize_gpu

_LAYER_RE = re.compile(r"^(blk\d+)_")


def npz_path_for_codec(codec_path: Union[str, Path]) -> Path:
    """Sidecar ``.npz`` path for a ``.pt`` codec checkpoint."""
    path = Path(codec_path)
    return path.with_suffix(".npz")


def parse_layer_from_stem(stem: str) -> Optional[str]:
    match = _LAYER_RE.match(stem)
    return match.group(1) if match else None


def extract_npz_arrays(codec) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``R`` (D,D) and ``codebooks`` (G,K,d) from a FeatureCodec."""
    transform = codec.transform
    if transform is None or not hasattr(transform, "get_rotation"):
        raise ValueError(
            "NPZ export requires OrthogonalTransform; "
            f"got {type(transform).__name__ if transform else None}"
        )
    R = transform.get_rotation().detach().cpu().numpy()
    codebooks = codec.pq.codebooks.detach().cpu().numpy()
    return R, codebooks


@torch.no_grad()
def compute_histogram_pmf(
    codec,
    features: Sequence[np.ndarray],
    *,
    norm_mode: str = "per_image",
    device: Union[str, torch.device] = "cuda",
    batch_size: int = 200,
) -> np.ndarray:
    """Count label histogram on train features; return ``pmf`` (G, K)."""
    codec.eval()
    pq = codec.pq
    G, K = pq.G, pq.K
    label_counts = np.zeros((G, K), dtype=np.int64)

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        _ = codec(Y)
        labels = pq._last_labels.cpu().numpy()
        for g in range(G):
            for lbl in labels[g]:
                label_counts[g, int(lbl)] += 1
        del X, Y

    pmf = label_counts.astype(np.float64)
    pmf = np.maximum(pmf, 1e-7)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    return pmf.astype(np.float32)


def save_codec_npz(
    codec,
    npz_path: Union[str, Path],
    pmf: np.ndarray,
    *,
    source_pt: Optional[Union[str, Path]] = None,
) -> Path:
    """Write ORFC-compatible ``.npz`` with ``R``, ``codebooks``, ``pmf``."""
    R, codebooks = extract_npz_arrays(codec)
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {}
    if source_pt is not None:
        meta["source_pt"] = str(source_pt)
    np.savez(npz_path, R=R, codebooks=codebooks, pmf=pmf, **meta)
    return npz_path


def load_pmf_npz(npz_path: Union[str, Path]) -> np.ndarray:
    """Load ``pmf`` array (G, K) from sidecar ``.npz``."""
    data = np.load(npz_path, allow_pickle=True)
    if "pmf" not in data:
        raise KeyError(f"No 'pmf' in {npz_path}")
    return data["pmf"].astype(np.float32)


def pmf_as_list(pmf: np.ndarray) -> List[np.ndarray]:
    """Convert (G, K) PMF to list of G arrays for rANS helpers."""
    G = pmf.shape[0]
    return [pmf[g] for g in range(G)]


def try_load_sidecar_pmf(codec_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Return PMF if sidecar ``.npz`` exists next to ``codec_path``."""
    npz_path = npz_path_for_codec(codec_path)
    if not npz_path.is_file():
        return None
    return load_pmf_npz(npz_path)

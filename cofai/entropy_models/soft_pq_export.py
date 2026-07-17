"""Soft-PQ ``.npz`` I/O (ORFC-compatible: R + codebooks + pmf + norm meta).

Training may still produce an intermediate ``.pt`` FeatureCodec checkpoint; the
canonical artifact for online/offline evaluation and release is a single
``.npz`` written by :func:`save_codec_npz`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cofai.entropy_models.orfc_model import batch_normalize_gpu

_LAYER_RE = re.compile(r"^(blk\d+|slot\d+)_")

NORM_MODE_CHOICES = (
    "per_image",
    "split_cls_patch",
    "split_reg_cls_patch",
    "per_token_ln",
)


def npz_path_for_codec(codec_path: Union[str, Path]) -> Path:
    """``.npz`` path for a codec path (``*.pt`` -> ``*.npz``, else same stem)."""
    path = Path(codec_path)
    if path.suffix.lower() == ".npz":
        return path
    return path.with_suffix(".npz")


def parse_layer_from_stem(stem: str) -> Optional[str]:
    match = _LAYER_RE.match(stem)
    return match.group(1) if match else None


def extract_npz_arrays(codec) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Extract ``codebooks`` (G,K,d) and optional ``R`` (D,D) from a FeatureCodec."""
    codebooks = codec.pq.codebooks.detach().cpu().numpy()
    transform = codec.transform
    if transform is None:
        return None, codebooks
    if not hasattr(transform, "get_rotation"):
        raise ValueError(
            "NPZ export requires OrthogonalTransform for R; "
            f"got {type(transform).__name__}"
        )
    R = transform.get_rotation().detach().cpu().numpy()
    return R, codebooks


@torch.no_grad()
def compute_histogram_pmf(
    codec,
    features: Sequence[np.ndarray],
    *,
    norm_mode: str = "per_image",
    n_prefix: int = 0,
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
        if isinstance(features, np.ndarray):
            batch = features[start:end]
        else:
            items = []
            for i in range(start, end):
                item = features[i]
                if isinstance(item, (str, Path)):
                    items.append(np.load(item).astype(np.float32))
                else:
                    items.append(np.asarray(item, dtype=np.float32))
            batch = np.stack(items)
        X = torch.from_numpy(batch).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
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
    norm_mode: str = "per_image",
    n_prefix: int = 0,
) -> Path:
    """Write evaluation ``.npz`` with ``codebooks``, ``pmf``, optional ``R``, norm meta."""
    R, codebooks = extract_npz_arrays(codec)
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "codebooks": codebooks,
        "pmf": pmf,
        "norm_mode": np.asarray(norm_mode),
        "n_prefix": np.asarray(int(n_prefix), dtype=np.int32),
    }
    if R is not None:
        payload["R"] = R
    else:
        D = int(codebooks.shape[0] * codebooks.shape[2])
        payload["R"] = np.eye(D, dtype=np.float32)
        payload["use_transform"] = np.bool_(False)
    if source_pt is not None:
        payload["source_pt"] = str(source_pt)
    np.savez(npz_path, **payload)
    return npz_path


def load_softpq_npz(npz_path: Union[str, Path]) -> Dict[str, Any]:
    """Load SoftPQ / ORFC-compatible ``.npz`` into a plain dict of arrays + meta.

    Returns keys: ``R``, ``codebooks``, ``pmf`` (or None), ``norm_mode``, ``n_prefix``,
    ``K``, ``embedding_dim`` (``d``), ``G``, ``feat_dim``.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"SoftPQ npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if "codebooks" not in data:
        raise KeyError(f"No 'codebooks' in {npz_path}")
    codebooks = data["codebooks"].astype(np.float32)
    G, K, d = codebooks.shape
    D = G * d
    if "R" in data:
        R = data["R"].astype(np.float32)
    else:
        R = np.eye(D, dtype=np.float32)
    pmf = data["pmf"].astype(np.float32) if "pmf" in data else None
    norm_mode = "per_image"
    if "norm_mode" in data:
        raw = data["norm_mode"]
        norm_mode = str(raw.item() if hasattr(raw, "item") else raw)
    n_prefix = 0
    if "n_prefix" in data:
        raw = data["n_prefix"]
        n_prefix = int(raw.item() if hasattr(raw, "item") else raw)
    return {
        "R": R,
        "codebooks": codebooks,
        "pmf": pmf,
        "norm_mode": norm_mode,
        "n_prefix": n_prefix,
        "K": int(K),
        "embedding_dim": int(d),
        "G": int(G),
        "feat_dim": int(D),
        "path": str(npz_path),
    }


def load_pmf_npz(npz_path: Union[str, Path]) -> np.ndarray:
    """Load ``pmf`` array (G, K) from ``.npz``."""
    data = np.load(npz_path, allow_pickle=True)
    if "pmf" not in data:
        raise KeyError(f"No 'pmf' in {npz_path}")
    return data["pmf"].astype(np.float32)


def pmf_as_list(pmf: np.ndarray) -> List[np.ndarray]:
    """Convert (G, K) PMF to list of G arrays for rANS helpers."""
    G = pmf.shape[0]
    return [pmf[g] for g in range(G)]


def try_load_sidecar_pmf(codec_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Return PMF if ``.npz`` exists next to / at ``codec_path`` (legacy helper)."""
    npz_path = npz_path_for_codec(codec_path)
    if not npz_path.is_file():
        return None
    return load_pmf_npz(npz_path)

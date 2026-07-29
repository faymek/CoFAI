"""Offline Soft-PQ artifact export helpers for ORFC-2446.

Training may still produce an intermediate ``.pt`` FeatureCodec checkpoint; the
released artifact is a single ``.npz`` written by :func:`save_codec_npz`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from cofai.latent_codecs.orfc import load_orfc_artifact

from cofai.latent_codecs.orfc_normalization import normalize_orfc_features

_LAYER_RE = re.compile(r"^(blk\d+|slot\d+)_")


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
        raise ValueError("NPZ export requires OrthogonalTransform for R; " f"got {type(transform).__name__}")
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
        Y, _, _ = normalize_orfc_features(X, mode=norm_mode, n_prefix=n_prefix)
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
    metadata: Optional[Dict[str, Any]] = None,
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
    if metadata:
        overlap = set(payload).intersection(metadata)
        if overlap:
            raise ValueError(f"Artifact metadata cannot replace codec fields: {sorted(overlap)}")
        payload.update({key: np.asarray(value) for key, value in metadata.items()})
    np.savez(npz_path, **payload)
    return npz_path


def load_softpq_npz(npz_path: Union[str, Path]) -> Dict[str, Any]:
    """Load the runtime artifact and expose offline naming aliases."""
    artifact = load_orfc_artifact(npz_path)
    return {
        **artifact,
        "K": artifact["entries"],
        "G": artifact["groups"],
        "feat_dim": artifact["feature_dim"],
    }

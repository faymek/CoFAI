"""Rate evaluation (histogram PMF + rANS) for offline ORFC replay."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from cofai.entropy_models.orfc_model import batch_normalize_gpu
from cofai.entropy_models.soft_pq_export import (
    npz_path_for_codec,
    pmf_as_list,
    try_load_sidecar_pmf,
)

try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans as _ans

    _HAS_ANS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ANS = False


def _codec_labels(features, codec, norm_mode, device, n_prefix=0, batch_size=32):
    codec.eval()
    is_latent = hasattr(codec, "R") and hasattr(codec, "_assign_labels")
    if is_latent:
        if hasattr(codec, "norm_mode"):
            codec.norm_mode = norm_mode
        if hasattr(codec, "n_prefix"):
            codec.n_prefix = int(n_prefix)
    else:
        pq = codec.pq
    all_labels = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch = features[start:end]
            if len({f.shape[0] for f in batch}) == 1:
                X = torch.from_numpy(np.stack(batch)).float().to(device)
                if is_latent:
                    _, _, labels = codec._assign_labels(X)
                    all_labels.append(labels.cpu())
                else:
                    Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
                    _ = codec(Y)
                    all_labels.append(pq._last_labels.cpu())
                del X
                continue
            labels_parts = []
            for feat in batch:
                X1 = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                if is_latent:
                    _, _, labels = codec._assign_labels(X1)
                    labels_parts.append(labels.cpu())
                else:
                    Y, _, _ = batch_normalize_gpu(X1, mode=norm_mode, n_prefix=n_prefix)
                    _ = codec(Y)
                    labels_parts.append(pq._last_labels.cpu())
                del X1
            all_labels.append(torch.cat(labels_parts, dim=1))
    return torch.cat(all_labels, dim=1).numpy()


def _histogram_pmf(labels, G, K, smoothing=1.0):
    pmfs = []
    for g in range(G):
        counts = np.zeros(K, dtype=np.float64)
        np.add.at(counts, labels[g], 1)
        counts += smoothing
        pmfs.append(counts / counts.sum())
    return pmfs


def _rans_encode_bpt(labels_np, pmf_list, G, K, precision=16):
    if not _HAS_ANS:
        return None
    encoder = _ans.RansEncoder()
    n = labels_np.shape[1]
    cdfs, cdf_sizes = [], []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdf = _pmf_to_quantized_cdf(p.tolist(), precision)
        cdfs.append(cdf)
        cdf_sizes.append(K + 2)
    symbols, cdf_indices = [], []
    for i in range(n):
        for g in range(G):
            symbols.append(int(labels_np[g, i]))
            cdf_indices.append(g)
    byte_string = encoder.encode_with_indexes(
        symbols, cdf_indices, cdfs, cdf_sizes, [0] * G,
    )
    return len(byte_string) * 8 / n


def evaluate_rate(
    features: Sequence[np.ndarray],
    codec,
    *,
    ckpt_path: str,
    norm_mode: str,
    n_prefix: int,
    embed_dim: int,
    device,
    sideinfo_bpi: float,
    require_sidecar: bool = False,
) -> dict:
    """Dataset-level rate using train PMF sidecar (preferred) + optional rANS.

    Reports real coded rate when compressai ANS is available; otherwise
    cross-entropy under the train (or fallback) PMF. Theoretical ceiling is
    exposed separately as ``max_bpt`` / ``bpfp_max``.
    """
    if hasattr(codec, "G") and hasattr(codec, "K") and hasattr(codec, "R"):
        G, K = int(codec.G), int(codec.K)
    else:
        pq = codec.pq
        G, K = pq.G, pq.K
    labels = _codec_labels(
        features, codec, norm_mode, device, n_prefix=n_prefix,
    )
    n_tokens = labels.shape[1]

    npz_path = npz_path_for_codec(ckpt_path)
    # SoftPQFeatureCodec already holds pmf from the same .npz
    if getattr(codec, "pmf", None) is not None and str(ckpt_path).endswith(".npz"):
        pmf_np = codec.pmf.detach().cpu().numpy()
        train_pmf = pmf_as_list(pmf_np)
        pmf_source = "npz_embedded_pmf"
    else:
        pmf_np = try_load_sidecar_pmf(ckpt_path)
        if pmf_np is not None:
            train_pmf = pmf_as_list(pmf_np)
            pmf_source = "train_sidecar_npz"
        else:
            if require_sidecar:
                raise FileNotFoundError(
                    f"Train-prior PMF sidecar required but missing: {npz_path}. "
                    "Run export_npz_dinov3.py or retrain with sidecar export."
                )
            print(
                f"[rate][warn] No PMF sidecar at {npz_path}; "
                "falling back to eval-set histogram (optimistic, not train prior)"
            )
            train_pmf = _histogram_pmf(labels, G, K, smoothing=1.0)
            pmf_source = "eval_histogram_fallback"

    xent = 0.0
    for g in range(G):
        log2_p = np.log2(np.array(train_pmf[g]) + 1e-30)
        xent += -log2_p[labels[g]].sum()
    xent_bpt = xent / n_tokens

    empirical_entropy = 0.0
    test_pmf = _histogram_pmf(labels, G, K, smoothing=0)
    for g in range(G):
        pg = test_pmf[g]
        pg = pg[pg > 0]
        empirical_entropy += -np.sum(pg * np.log2(pg))

    max_bpt = G * math.log2(K)
    rans_bpt = _rans_encode_bpt(labels, train_pmf, G, K)

    avg_t = float(np.mean([f.shape[0] for f in features]))
    si_bpt = sideinfo_bpi / avg_t if avg_t > 0 else 0.0
    if rans_bpt is not None:
        codec_bpt = rans_bpt
        rate_kind = "rans_real"
    else:
        codec_bpt = xent_bpt
        rate_kind = "xent_approx"
    bpfp_codec = codec_bpt / embed_dim
    bpfp_si = si_bpt / embed_dim
    return {
        "pmf_source": pmf_source,
        "npz_path": str(npz_path),
        "rate_kind": rate_kind,
        "xent_bpt": float(xent_bpt),
        "entropy_bpt": float(empirical_entropy),
        "max_bpt": float(max_bpt),
        "rans_bpt": float(rans_bpt) if rans_bpt is not None else None,
        "codec_bpt": float(codec_bpt),
        "bpfp_codec": float(bpfp_codec),
        "bpfp_sideinfo": float(bpfp_si),
        "bpfp": float(bpfp_codec + bpfp_si),
        "bpfp_max": float(max_bpt / embed_dim + bpfp_si),
        "sideinfo_bpi": float(sideinfo_bpi),
        "n_tokens": int(n_tokens),
    }

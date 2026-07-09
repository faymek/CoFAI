"""Rate evaluation (histogram PMF + rANS) for offline ORFC replay."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from cofai.entropy_models.orfc_model import batch_normalize_gpu
from cofai.entropy_models.soft_pq_export import pmf_as_list, try_load_sidecar_pmf

try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans as _ans

    _HAS_ANS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ANS = False


def _codec_labels(features, codec, norm_mode, device, n_prefix=0, batch_size=32,
                  prefix_bypass=False):
    codec.eval()
    pq = codec.pq
    all_labels = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch = features[start:end]
            if len({f.shape[0] for f in batch}) == 1:
                X = torch.from_numpy(np.stack(batch)).float().to(device)
                Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
                if prefix_bypass and n_prefix > 0:
                    _ = codec(Y[:, n_prefix:, :])
                else:
                    _ = codec(Y)
                all_labels.append(pq._last_labels.cpu())
                del X, Y
                continue
            labels_parts = []
            for feat in batch:
                X1 = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                Y, _, _ = batch_normalize_gpu(X1, mode=norm_mode, n_prefix=n_prefix)
                if prefix_bypass and n_prefix > 0:
                    _ = codec(Y[:, n_prefix:, :])
                else:
                    _ = codec(Y)
                labels_parts.append(pq._last_labels.cpu())
                del X1, Y
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
    prefix_bypass: bool = False,
) -> dict:
    """Dataset-level rate using train PMF sidecar."""
    pq = codec.pq
    G, K = pq.G, pq.K
    labels = _codec_labels(
        features, codec, norm_mode, device, n_prefix=n_prefix,
        prefix_bypass=prefix_bypass,
    )
    n_tokens = labels.shape[1]

    pmf_np = try_load_sidecar_pmf(ckpt_path)
    if pmf_np is None:
        train_pmf = _histogram_pmf(labels, G, K, smoothing=1.0)
    else:
        train_pmf = pmf_as_list(pmf_np)

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
    codec_bpt = rans_bpt if rans_bpt is not None else xent_bpt
    bpfp_codec = codec_bpt / embed_dim
    bpfp_si = si_bpt / embed_dim
    return {
        "xent_bpt": float(xent_bpt),
        "entropy_bpt": float(empirical_entropy),
        "max_bpt": float(max_bpt),
        "rans_bpt": float(rans_bpt) if rans_bpt is not None else None,
        "bpfp_codec": float(bpfp_codec),
        "bpfp_sideinfo": float(bpfp_si),
        "bpfp": float(bpfp_codec + bpfp_si),
        "sideinfo_bpi": float(sideinfo_bpi),
        "n_tokens": int(n_tokens),
        "prefix_bypass": bool(prefix_bypass),
    }

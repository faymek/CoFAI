"""Real bitstream accounting for offline ORFC replay."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from cofai.engine.bitrate import bits_from_coded_unit


@torch.inference_mode()
def evaluate_rate(
    features: Sequence[np.ndarray],
    codec,
    *,
    embed_dim: int,
    device,
) -> dict:
    """Measure exactly the streams emitted by ``OrthoRotationFeatureCodec``."""
    label_bits = 0
    stats_bits = 0
    cross_entropy_bits = 0.0
    n_tokens = 0

    for feature in features:
        tokens = torch.from_numpy(feature).float().unsqueeze(0).to(device)
        likelihoods = codec(tokens)["likelihoods"]["orfc"]
        cross_entropy_bits += float((-torch.log2(likelihoods)).sum().item())

        coded = codec.compress(tokens)
        stream_bits = bits_from_coded_unit(coded)
        label_bits += stream_bits["orfc"]
        stats_bits += stream_bits["orfc_stats"]
        n_tokens += int(feature.shape[0])

    if n_tokens == 0:
        raise ValueError("Rate evaluation requires at least one feature")

    rans_bpt = label_bits / n_tokens
    stats_bpt = stats_bits / n_tokens
    xent_bpt = cross_entropy_bits / n_tokens
    max_bpt = codec.G * math.log2(codec.K)
    total_bpfp = (rans_bpt + stats_bpt) / embed_dim
    return {
        "pmf_source": "artifact",
        "rate_kind": "rans_real",
        "xent_bpt": float(xent_bpt),
        "max_bpt": float(max_bpt),
        "rans_bpt": float(rans_bpt),
        "codec_bpt": float(rans_bpt),
        "stats_bpt": float(stats_bpt),
        "bpfp_codec": float(rans_bpt / embed_dim),
        "bpfp_sideinfo": float(stats_bpt / embed_dim),
        "bpfp": float(total_bpfp),
        "bpfp_max": float((max_bpt + stats_bpt) / embed_dim),
        "sideinfo_bpi": float(stats_bits / len(features)),
        "n_tokens": int(n_tokens),
    }

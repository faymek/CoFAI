"""Transparent BPFP accounting for token grouping experiments."""

from __future__ import annotations


def bitrate_record(
    token_count: int,
    kept_tokens: int,
    feature_bits: int,
    map_bits: int,
    metadata_bits: int,
    dense_feature_bits: int,
) -> dict[str, float | int]:
    if token_count <= 0 or kept_tokens <= 0 or kept_tokens > token_count:
        raise ValueError("invalid token counts")
    total_bits = feature_bits + map_bits + metadata_bits
    dense_bits = dense_feature_bits + metadata_bits
    return {
        "token_count": token_count,
        "kept_tokens": kept_tokens,
        "keep_ratio": kept_tokens / token_count,
        "feature_bits": feature_bits,
        "map_bits": map_bits,
        "metadata_bits": metadata_bits,
        "total_bits": total_bits,
        "feature_bpfp": feature_bits / token_count,
        "map_bpfp": (map_bits + metadata_bits) / token_count,
        "total_bpfp": total_bits / token_count,
        "side_information_fraction": (map_bits + metadata_bits) / total_bits,
        "rate_saving": 1.0 - total_bits / dense_bits,
    }

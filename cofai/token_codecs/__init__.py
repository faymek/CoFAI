"""Token codecs.

The selection-map codec has no optional compression dependency. Existing
feature codecs remain available when their optional runtime is installed.
"""

try:
    from .base import UniformTokenCodec
except ModuleNotFoundError:  # pragma: no cover - dependency-specific fallback
    UniformTokenCodec = None

try:
    from .naive_codec import NaiveCodec
except ModuleNotFoundError:  # pragma: no cover - dependency-specific fallback
    NaiveCodec = None

from .token_selection_map import (
    SelectionMapPacket,
    decode_selection_map,
    encode_selection_map,
    packet_from_indices,
    selection_map_bits,
)

__all__ = [
    "UniformTokenCodec",
    "NaiveCodec",
    "SelectionMapPacket",
    "decode_selection_map",
    "encode_selection_map",
    "packet_from_indices",
    "selection_map_bits",
]

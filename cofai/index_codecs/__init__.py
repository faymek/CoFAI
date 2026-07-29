"""Codecs for bounded discrete indices."""

from .adaptive_bitmap_index import (
    AdaptiveBitmapIndexCodec,
    EncodedSelectionMap,
)

__all__ = [
    "AdaptiveBitmapIndexCodec",
    "EncodedSelectionMap",
]

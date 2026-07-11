"""Bit-true codecs for token selection maps and ordered index sets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import torch


@dataclass(frozen=True)
class SelectionMapPacket:
    token_count: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class EncodedSelectionMap:
    coding: str
    token_count: int
    kept_count: int
    payload: bytes

    @property
    def payload_bits(self) -> int:
        return len(self.payload) * 8


def encode_selection_map(mask: torch.Tensor) -> bytes:
    """Encode a boolean map as a compact bitmap with a two-field header."""
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("mask must be a one-dimensional boolean tensor")
    token_count = int(mask.numel())
    payload = bytearray((token_count + 7) // 8)
    for index, selected in enumerate(mask.tolist()):
        if selected:
            payload[index // 8] |= 1 << (index % 8)
    return token_count.to_bytes(4, "big") + bytes(payload)


def decode_selection_map(stream: bytes) -> torch.Tensor:
    """Decode a bitmap produced by :func:`encode_selection_map`."""
    if len(stream) < 4:
        raise ValueError("selection-map stream is truncated")
    token_count = int.from_bytes(stream[:4], "big")
    payload_size = (token_count + 7) // 8
    if len(stream) != 4 + payload_size:
        raise ValueError("selection-map stream has an invalid length")
    mask = torch.zeros(token_count, dtype=torch.bool)
    for index in range(token_count):
        mask[index] = bool(stream[4 + index // 8] & (1 << (index % 8)))
    return mask


def packet_from_indices(indices: torch.Tensor, token_count: int) -> SelectionMapPacket:
    """Create a canonical, sorted selection packet."""
    if indices.ndim != 1 or torch.any(indices < 0) or torch.any(indices >= token_count):
        raise ValueError("indices must be one-dimensional and within token_count")
    values = sorted(set(int(value) for value in indices.tolist()))
    return SelectionMapPacket(token_count=token_count, indices=tuple(values))


def selection_map_bits(mask: torch.Tensor) -> int:
    """Return the exact number of bits in the bitmap stream."""
    return len(encode_selection_map(mask)) * 8


def _validate_indices(indices, token_count: int) -> list[int]:
    values = sorted(int(index) for index in indices)
    if token_count <= 0 or any(index < 0 or index >= token_count for index in values):
        raise ValueError("indices must be within token_count")
    if len(values) != len(set(values)):
        raise ValueError("indices must not contain duplicates")
    return values


def _pack_fixed_width(values: list[int], width: int) -> bytes:
    accumulator = 0
    accumulated_bits = 0
    payload = bytearray()
    for value in values:
        accumulator = (accumulator << width) | value
        accumulated_bits += width
        while accumulated_bits >= 8:
            shift = accumulated_bits - 8
            payload.append((accumulator >> shift) & 0xFF)
            accumulated_bits -= 8
            accumulator &= (1 << accumulated_bits) - 1 if accumulated_bits else 0
    if accumulated_bits:
        payload.append((accumulator << (8 - accumulated_bits)) & 0xFF)
    return bytes(payload)


def _unpack_fixed_width(payload: bytes, count: int, width: int) -> list[int]:
    values = []
    accumulator = 0
    accumulated_bits = 0
    mask = (1 << width) - 1
    for byte in payload:
        accumulator = (accumulator << 8) | byte
        accumulated_bits += 8
        while accumulated_bits >= width and len(values) < count:
            shift = accumulated_bits - width
            values.append((accumulator >> shift) & mask)
            accumulated_bits -= width
            accumulator &= (1 << accumulated_bits) - 1 if accumulated_bits else 0
    if len(values) != count:
        raise ValueError("index payload is truncated")
    return values


def encode_selection_indices(indices, token_count: int) -> EncodedSelectionMap:
    """Encode indices using the smaller of a bitmap and fixed-width index list."""
    values = _validate_indices(indices, token_count)
    bitmap = bytearray(ceil(token_count / 8))
    for index in values:
        bitmap[index // 8] |= 1 << (index % 8)
    width = max(1, ceil(log2(token_count)))
    index_payload = _pack_fixed_width(values, width)
    if len(index_payload) < len(bitmap):
        return EncodedSelectionMap("index", token_count, len(values), index_payload)
    return EncodedSelectionMap("bitmap", token_count, len(values), bytes(bitmap))


def decode_selection_indices(encoded: EncodedSelectionMap) -> list[int]:
    """Decode an index set produced by :func:`encode_selection_indices`."""
    if encoded.coding == "index":
        values = _unpack_fixed_width(
            encoded.payload,
            encoded.kept_count,
            max(1, ceil(log2(encoded.token_count))),
        )
    elif encoded.coding == "bitmap":
        expected_size = ceil(encoded.token_count / 8)
        if len(encoded.payload) != expected_size:
            raise ValueError("bitmap payload has an invalid length")
        values = [
            index
            for index in range(encoded.token_count)
            if encoded.payload[index // 8] & (1 << (index % 8))
        ]
    else:
        raise ValueError(f"unsupported selection-map coding: {encoded.coding}")
    values = _validate_indices(values, encoded.token_count)
    if len(values) != encoded.kept_count:
        raise ValueError("decoded selection-map cardinality does not match the header")
    return values

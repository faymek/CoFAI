"""Adaptive bitmap/index-list coding for token-selection index sets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from struct import Struct

import torch
import torch.nn as nn

_STREAM_HEADER = Struct(">4sBII")
_STREAM_MAGIC = b"BIS1"
_CODING_TO_ID = {"bitmap": 0, "index": 1}
_ID_TO_CODING = {value: key for key, value in _CODING_TO_ID.items()}


@dataclass(frozen=True)
class BoundedIndexSetBatch:
    """A batch of unique index sets sharing one finite universe."""

    indices: torch.Tensor
    universe_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.indices, torch.Tensor) or self.indices.ndim != 2:
            raise TypeError("indices must be a two-dimensional tensor")
        if self.universe_size <= 0:
            raise ValueError("universe_size must be positive")


@dataclass(frozen=True)
class EncodedIndexSet:
    coding: str
    universe_size: int
    selected_count: int
    payload: bytes

    @property
    def payload_bits(self) -> int:
        return len(self.payload) * 8

    @property
    def stream_bits(self) -> int:
        """Return the size of the independently decodable byte stream."""
        return len(self.to_bytes()) * 8

    def to_bytes(self) -> bytes:
        """Serialize the coding choice, universe, cardinality, and payload."""
        try:
            coding_id = _CODING_TO_ID[self.coding]
        except KeyError as exc:
            raise ValueError(f"unsupported index-set coding: {self.coding}") from exc
        if not 0 < self.universe_size <= 0xFFFFFFFF:
            raise ValueError("universe_size must fit in an unsigned 32-bit integer")
        if not 0 <= self.selected_count <= self.universe_size:
            raise ValueError("selected_count must be within universe_size")
        return (
            _STREAM_HEADER.pack(
                _STREAM_MAGIC,
                coding_id,
                self.universe_size,
                self.selected_count,
            )
            + self.payload
        )

    @classmethod
    def from_bytes(cls, stream: bytes | bytearray | memoryview) -> EncodedIndexSet:
        """Parse a self-contained bounded-index-set stream."""
        stream = bytes(stream)
        if len(stream) < _STREAM_HEADER.size:
            raise ValueError("index-set stream is truncated")
        magic, coding_id, universe_size, selected_count = _STREAM_HEADER.unpack_from(
            stream
        )
        if magic != _STREAM_MAGIC:
            raise ValueError("index-set stream has an invalid magic")
        try:
            coding = _ID_TO_CODING[coding_id]
        except KeyError as exc:
            raise ValueError(f"unsupported index-set coding id: {coding_id}") from exc
        encoded = cls(
            coding,
            universe_size,
            selected_count,
            stream[_STREAM_HEADER.size :],
        )
        _validate_encoded_payload(encoded)
        return encoded


class AdaptiveBitmapIndexCodec(nn.Module):
    """Encode a unique bounded index set using its shorter byte representation.

    Sparse selections use a fixed-width list of absolute indices. Dense selections
    use one bitmap bit per possible index. The serialized stream includes the
    selected representation, total index count, and retained index count.
    """

    def encode(self, indices, universe_size: int) -> EncodedIndexSet:
        """Encode indices using the smaller bitmap or fixed-width index payload."""
        values = _validate_indices(indices, universe_size)
        bitmap = bytearray(ceil(universe_size / 8))
        for index in values:
            bitmap[index // 8] |= 1 << (index % 8)
        width = max(1, ceil(log2(universe_size)))
        index_payload = _pack_fixed_width(values, width)
        if len(index_payload) < len(bitmap):
            return EncodedIndexSet(
                "index",
                universe_size,
                len(values),
                index_payload,
            )
        return EncodedIndexSet(
            "bitmap",
            universe_size,
            len(values),
            bytes(bitmap),
        )

    def decode(
        self,
        encoded: EncodedIndexSet | bytes | bytearray | memoryview,
    ) -> list[int]:
        """Decode an encoded object or independently decodable byte stream."""
        if not isinstance(encoded, EncodedIndexSet):
            encoded = EncodedIndexSet.from_bytes(encoded)
        _validate_encoded_payload(encoded)
        if encoded.coding == "index":
            values = _unpack_fixed_width(
                encoded.payload,
                encoded.selected_count,
                max(1, ceil(log2(encoded.universe_size))),
            )
        elif encoded.coding == "bitmap":
            values = [
                index
                for index in range(encoded.universe_size)
                if encoded.payload[index // 8] & (1 << (index % 8))
            ]
        else:
            raise ValueError(f"unsupported index-set coding: {encoded.coding}")
        values = _validate_indices(values, encoded.universe_size)
        if len(values) != encoded.selected_count:
            raise ValueError("decoded index-set cardinality does not match the header")
        return values

    def encode_batch(self, batch: BoundedIndexSetBatch) -> list[list[bytes]]:
        """Encode a batch into the canonical one-stream-per-sample layout."""
        return [
            [self.encode(indices, batch.universe_size).to_bytes()]
            for indices in batch.indices.detach().cpu().tolist()
        ]

    def decode_batch(self, rows: list[list[bytes]]) -> list[list[int]]:
        """Decode canonical stream rows and validate every index set."""
        decoded = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 1:
                raise ValueError(
                    "each index-set stream row must contain one byte string"
                )
            decoded.append(self.decode(row[0]))
        return decoded


def _validate_indices(indices, universe_size: int) -> list[int]:
    values = sorted(int(index) for index in indices)
    if universe_size <= 0 or any(
        index < 0 or index >= universe_size for index in values
    ):
        raise ValueError("indices must be within universe_size")
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
    expected_size = ceil(count * width / 8)
    if len(payload) != expected_size:
        raise ValueError("index payload has an invalid length")
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
    if accumulator:
        raise ValueError("index payload has non-zero padding bits")
    return values


def _validate_encoded_payload(encoded: EncodedIndexSet) -> None:
    if encoded.universe_size <= 0:
        raise ValueError("universe_size must be positive")
    if not 0 <= encoded.selected_count <= encoded.universe_size:
        raise ValueError("selected_count must be within universe_size")
    if encoded.coding == "index":
        width = max(1, ceil(log2(encoded.universe_size)))
        expected_size = ceil(encoded.selected_count * width / 8)
    elif encoded.coding == "bitmap":
        expected_size = ceil(encoded.universe_size / 8)
    else:
        raise ValueError(f"unsupported index-set coding: {encoded.coding}")
    if len(encoded.payload) != expected_size:
        raise ValueError(f"{encoded.coding} payload has an invalid length")

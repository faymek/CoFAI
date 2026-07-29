"""Adaptive bitmap/index-list coding for token-selection index sets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from struct import Struct

_STREAM_HEADER = Struct(">4sBII")
_STREAM_MAGIC = b"TSM1"
_CODING_TO_ID = {"bitmap": 0, "index": 1}
_ID_TO_CODING = {value: key for key, value in _CODING_TO_ID.items()}


@dataclass(frozen=True)
class EncodedSelectionMap:
    coding: str
    token_count: int
    kept_count: int
    payload: bytes

    @property
    def payload_bits(self) -> int:
        return len(self.payload) * 8

    @property
    def stream_bits(self) -> int:
        """Return the size of the independently decodable byte stream."""
        return len(self.to_bytes()) * 8

    def to_bytes(self) -> bytes:
        """Serialize the coding choice, dimensions, and payload."""
        try:
            coding_id = _CODING_TO_ID[self.coding]
        except KeyError as exc:
            raise ValueError(
                f"unsupported selection-map coding: {self.coding}"
            ) from exc
        if not 0 < self.token_count <= 0xFFFFFFFF:
            raise ValueError("token_count must fit in an unsigned 32-bit integer")
        if not 0 <= self.kept_count <= self.token_count:
            raise ValueError("kept_count must be within token_count")
        return (
            _STREAM_HEADER.pack(
                _STREAM_MAGIC,
                coding_id,
                self.token_count,
                self.kept_count,
            )
            + self.payload
        )

    @classmethod
    def from_bytes(cls, stream: bytes | bytearray | memoryview) -> EncodedSelectionMap:
        """Parse a self-contained selection-map stream."""
        stream = bytes(stream)
        if len(stream) < _STREAM_HEADER.size:
            raise ValueError("selection-map stream is truncated")
        magic, coding_id, token_count, kept_count = _STREAM_HEADER.unpack_from(stream)
        if magic != _STREAM_MAGIC:
            raise ValueError("selection-map stream has an invalid magic")
        try:
            coding = _ID_TO_CODING[coding_id]
        except KeyError as exc:
            raise ValueError(
                f"unsupported selection-map coding id: {coding_id}"
            ) from exc
        encoded = cls(coding, token_count, kept_count, stream[_STREAM_HEADER.size :])
        _validate_encoded_payload(encoded)
        return encoded


class AdaptiveBitmapIndexCodec:
    """Encode a unique bounded index set using its shorter byte representation.

    Sparse selections use a fixed-width list of absolute indices. Dense selections
    use one bitmap bit per possible index. The serialized stream includes the
    selected representation, total index count, and retained index count.
    """

    def encode(self, indices, token_count: int) -> EncodedSelectionMap:
        """Encode indices using the smaller bitmap or fixed-width index payload."""
        values = _validate_indices(indices, token_count)
        bitmap = bytearray(ceil(token_count / 8))
        for index in values:
            bitmap[index // 8] |= 1 << (index % 8)
        width = max(1, ceil(log2(token_count)))
        index_payload = _pack_fixed_width(values, width)
        if len(index_payload) < len(bitmap):
            return EncodedSelectionMap(
                "index",
                token_count,
                len(values),
                index_payload,
            )
        return EncodedSelectionMap(
            "bitmap",
            token_count,
            len(values),
            bytes(bitmap),
        )

    def decode(
        self,
        encoded: EncodedSelectionMap | bytes | bytearray | memoryview,
    ) -> list[int]:
        """Decode an encoded object or independently decodable byte stream."""
        if not isinstance(encoded, EncodedSelectionMap):
            encoded = EncodedSelectionMap.from_bytes(encoded)
        _validate_encoded_payload(encoded)
        if encoded.coding == "index":
            values = _unpack_fixed_width(
                encoded.payload,
                encoded.kept_count,
                max(1, ceil(log2(encoded.token_count))),
            )
        elif encoded.coding == "bitmap":
            values = [
                index
                for index in range(encoded.token_count)
                if encoded.payload[index // 8] & (1 << (index % 8))
            ]
        else:
            raise ValueError(f"unsupported selection-map coding: {encoded.coding}")
        values = _validate_indices(values, encoded.token_count)
        if len(values) != encoded.kept_count:
            raise ValueError(
                "decoded selection-map cardinality does not match the header"
            )
        return values


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


def _validate_encoded_payload(encoded: EncodedSelectionMap) -> None:
    if encoded.token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0 <= encoded.kept_count <= encoded.token_count:
        raise ValueError("kept_count must be within token_count")
    if encoded.coding == "index":
        width = max(1, ceil(log2(encoded.token_count)))
        expected_size = ceil(encoded.kept_count * width / 8)
    elif encoded.coding == "bitmap":
        expected_size = ceil(encoded.token_count / 8)
    else:
        raise ValueError(f"unsupported selection-map coding: {encoded.coding}")
    if len(encoded.payload) != expected_size:
        raise ValueError(f"{encoded.coding} payload has an invalid length")

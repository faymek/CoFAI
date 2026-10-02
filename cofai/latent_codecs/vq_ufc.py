"""CoFAI latent-codec adapter for the VQ-UFC index bitstream.

This module reuses CoFAI's ``DiscreteEntropyModel`` and its CompressAI rANS
implementation. It adapts the VQ-UFC
``forward/compress/decompress`` methods to the interface consumed by
``DinoFeatureCodecModel`` and ``cofai-eval``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from cofai.entropy_models.vq_ufc_model import VQUFCModel


class VQUFCFeatureCodec(nn.Module):
    """Expose an all-token RMS VQ-UFC checkpoint through the CoFAI codec API."""

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"VQ-UFC checkpoint not found: {path}")

        self.checkpoint_path = str(path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint.get("model_config")
        if not config:
            raise ValueError("VQ-UFC checkpoint is missing model_config")
        required = ("num_embeddings", "embedding_dim", "num_chunks")
        if any(name not in config for name in required):
            raise ValueError("VQ-UFC checkpoint has an incomplete model_config")
        expected = {
            "num_prefix_tokens": 5,
            "normalization_mode": "prefix_position_patch_channel_rms",
            "vector_mode": "legacy_sequence",
        }
        for name, value in expected.items():
            if config.get(name) != value:
                raise ValueError(f"Checkpoint {name} must be {value!r}")

        self.num_prefix_tokens = 5

        self.vq_ufc = VQUFCModel(
            num_embeddings=int(config["num_embeddings"]),
            embedding_dim=int(config["embedding_dim"]),
            num_chunks=int(config["num_chunks"]),
            lmbda=float(config.get("lmbda", 1.0)),
            vector_mode="legacy_sequence",
            use_transform=bool(config.get("use_transform", False)),
            transform_input_tokens=int(config.get("transform_input_tokens", 256)),
            transform_tokens=int(config.get("transform_tokens", 128)),
            commit_weight=float(config.get("commit_weight", 0.25)),
            use_soft_assignment=False,
        )
        self.vq_ufc.load_compatible_checkpoint(self.checkpoint_path, checkpoint=checkpoint)
        state = checkpoint.get("vqvae_state_dict", checkpoint.get("state_dict", checkpoint))
        if "base_logits" not in state:
            raise KeyError(
                "VQ-UFC checkpoint does not contain base_logits; refusing to "
                "build a new uniform CDF that would differ from training"
            )
        self.vq_ufc.eval()

        normalization = checkpoint.get("normalization", {})
        patch_rms = torch.as_tensor(normalization["patch_channel_rms"], dtype=torch.float32).reshape(-1)
        prefix_rms = torch.as_tensor(normalization["prefix_position_channel_rms"], dtype=torch.float32)
        expected_shape = (self.num_prefix_tokens, patch_rms.numel())
        if prefix_rms.shape != expected_shape:
            raise ValueError(f"Expected prefix RMS shape {expected_shape}, got {tuple(prefix_rms.shape)}")
        if not torch.isfinite(prefix_rms).all() or not torch.isfinite(patch_rms).all():
            raise ValueError("Checkpoint RMS contains non-finite values")
        if torch.any(prefix_rms <= 0) or torch.any(patch_rms <= 0):
            raise ValueError("Checkpoint RMS contains non-positive values")
        self.register_buffer("patch_channel_rms", patch_rms)
        self.register_buffer("prefix_position_channel_rms", prefix_rms)

        # Build the exact 16-bit CDF tables from the checkpoint's base_logits.
        # Both encoder and decoder below use this same model instance and tables.
        self.vq_ufc.prepare_entropy_model_for_compression()
        self._entropy_signature = self._make_entropy_signature()

    def _make_entropy_signature(self) -> str:
        digest = hashlib.sha256()
        logits = self.vq_ufc.base_logits.detach().cpu().contiguous().numpy()
        digest.update(np.asarray(logits, dtype="<f4").tobytes(order="C"))
        entropy_model = self.vq_ufc.entropy_model
        digest.update(np.asarray(entropy_model.cdf, dtype="<i4").tobytes(order="C"))
        digest.update(np.asarray(entropy_model.cdf_length, dtype="<i4").tobytes(order="C"))
        digest.update(np.asarray(entropy_model.cdf_offset, dtype="<i4").tobytes(order="C"))
        return digest.hexdigest()

    def _scale_features(self, h: torch.Tensor, inverse: bool) -> torch.Tensor:
        h = h.float()
        channels = h.shape[-1]
        prefix_rms = self.prefix_position_channel_rms.view(1, self.num_prefix_tokens, channels)
        patch_rms = self.patch_channel_rms.view(1, 1, channels)
        prefix, patch = h[:, : self.num_prefix_tokens], h[:, self.num_prefix_tokens :]
        if inverse:
            return torch.cat((prefix * prefix_rms, patch * patch_rms), dim=1)
        return torch.cat((prefix / prefix_rms, patch / patch_rms), dim=1)

    def _prepare_input(self, h: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
        if h.ndim != 3:
            raise ValueError(f"VQ-UFC expects B x T x C features, got {tuple(h.shape)}")
        context = {
            "batch": int(h.shape[0]),
            "full_tokens": int(h.shape[1]),
            "channels": int(h.shape[2]),
        }
        if h.shape[1] <= self.num_prefix_tokens:
            raise ValueError(f"Expected prefix+patch tokens, got feature shape {tuple(h.shape)}")
        channels = h.shape[-1]
        if self.prefix_position_channel_rms.shape != (self.num_prefix_tokens, channels):
            raise ValueError("Feature channels do not match checkpoint RMS")
        h = self._scale_features(h, inverse=False)
        if not torch.isfinite(h).all():
            raise FloatingPointError("VQ-UFC received non-finite normalized features")
        return h, context

    def _restore_output(self, h_hat: torch.Tensor, context: dict[str, int]) -> torch.Tensor:
        h_hat = self._scale_features(h_hat, inverse=True)
        expected = (context["batch"], context["full_tokens"], context["channels"])
        if tuple(h_hat.shape) != expected:
            raise RuntimeError(f"Restored VQ-UFC shape {tuple(h_hat.shape)} does not match {expected}")
        return h_hat

    def _likelihoods_from_indices(self, indices: list[torch.Tensor]) -> torch.Tensor:
        pmf = torch.softmax(self.vq_ufc.base_logits, dim=-1).reshape(-1)
        flat = torch.cat([item.reshape(-1).long() for item in indices], dim=0)
        return pmf[flat].clamp_min(torch.finfo(pmf.dtype).tiny)

    def forward(self, h, token_res=None, qp=0, **kwargs) -> dict[str, Any]:
        prepared, context = self._prepare_input(h)
        output = self.vq_ufc(prepared)
        h_hat, indices = self._restore_output(output[0], context), output[-1]
        return {
            "h_hat": h_hat,
            "likelihoods": {
                "vq_ufc": self._likelihoods_from_indices(indices),
            },
        }

    def compress(self, h, token_res=None, qp=0, **kwargs) -> dict[str, Any]:
        prepared, context = self._prepare_input(h)
        _h_hat, _mse, raw_streams, _indices, shape_info = self.vq_ufc.compress(prepared)
        if not all(isinstance(stream, bytes) for stream in raw_streams):
            raise TypeError("VQ-UFC entropy encoder must return bytes for every chunk")
        return {
            "strings": {
                # CoFAI counts len(stream[0]) for every coded stream.
                "vq_ufc": [[stream] for stream in raw_streams],
            },
            "pstate": {
                "shape_info": shape_info,
                "feature_context": context,
                "entropy_signature": self._entropy_signature,
            },
        }

    def decompress(self, strings, pstate, **kwargs) -> dict[str, torch.Tensor]:
        signature = str(pstate.get("entropy_signature", ""))
        if signature != self._entropy_signature:
            raise RuntimeError(
                "VQ-UFC entropy-table mismatch: encoder and decoder are not using "
                "the same checkpoint base_logits/CDF"
            )
        try:
            raw_streams = [item[0] for item in strings["vq_ufc"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("VQ-UFC coded unit is missing index streams") from exc
        if len(raw_streams) != len(pstate["shape_info"]["chunk_shapes"]):
            raise ValueError("VQ-UFC stream count does not match the saved chunk-shape count")
        h_hat = self.vq_ufc.decompress(raw_streams, pstate["shape_info"])
        return {"h_hat": self._restore_output(h_hat, pstate["feature_context"])}

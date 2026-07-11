"""Reusable GPS graph-partition token grouping.

The module operates on patch tokens and attention matrices. It has no dataset,
checkpoint, or task-metric dependency, so it can be reused by other examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenGroupingResult:
    """Output of one grouping operation."""

    tokens: torch.Tensor
    kept_indices: torch.Tensor
    deleted_indices: torch.Tensor
    deleted_sequence_indices: torch.Tensor
    keep_mask: torch.Tensor


class GraphTokenGrouper:
    """Select patch tokens using attention propagation from the class token."""

    def __init__(self, max_iter: int = 10, beta: float = 0.1, eps: float = 1e-3):
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not 0.0 < beta <= 1.0:
            raise ValueError("beta must be in (0, 1]")
        self.max_iter = max_iter
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(
        self,
        tokens: torch.Tensor,
        attention: torch.Tensor,
        keep_ratio: float,
        original_indices: torch.Tensor | None = None,
    ) -> TokenGroupingResult:
        if tokens.ndim != 3 or attention.ndim != 4:
            raise ValueError("tokens must be [B, N, C] and attention must be [B, H, N, N]")
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        batch, sequence_length, _ = tokens.shape
        if attention.shape[0] != batch or attention.shape[-2:] != (sequence_length, sequence_length):
            raise ValueError("attention and tokens have incompatible shapes")

        patch_count = sequence_length - 1
        if original_indices is None:
            original_indices = torch.arange(patch_count, device=tokens.device).expand(batch, -1).clone()
        if original_indices.shape != (batch, patch_count):
            raise ValueError("original_indices must have shape [B, N-1]")

        keep_count = max(int(patch_count * keep_ratio), 1)
        if keep_count == patch_count:
            empty = torch.empty(batch, 0, dtype=torch.long, device=tokens.device)
            return TokenGroupingResult(
                tokens=tokens,
                kept_indices=original_indices,
                deleted_indices=empty,
                deleted_sequence_indices=empty,
                keep_mask=torch.ones(batch, patch_count, dtype=torch.bool, device=tokens.device),
            )

        affinity = attention.mean(dim=1)
        affinity = (affinity + affinity.transpose(-1, -2)) / 2.0
        affinity = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        score = torch.zeros(batch, sequence_length, device=tokens.device, dtype=affinity.dtype)
        score[:, 0] = 1.0
        for _ in range(self.max_iter):
            next_score = self.beta * torch.nn.functional.one_hot(
                torch.zeros(batch, dtype=torch.long, device=tokens.device), sequence_length
            ).to(affinity.dtype)
            next_score = next_score + (1.0 - self.beta) * torch.bmm(
                affinity.transpose(1, 2), score.unsqueeze(-1)
            ).squeeze(-1)
            if torch.all((next_score - score).abs().sum(dim=1) < self.eps):
                score = next_score
                break
            score = next_score

        top_indices = torch.topk(score[:, 1:], keep_count, dim=1).indices.sort(dim=1).values
        compact_tokens = torch.gather(tokens[:, 1:], 1, top_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        compact_tokens = torch.cat([tokens[:, :1], compact_tokens], dim=1)
        keep_mask = torch.zeros(batch, patch_count, dtype=torch.bool, device=tokens.device)
        keep_mask.scatter_(1, top_indices, True)
        deleted_sequence_indices = (~keep_mask).nonzero(as_tuple=False).reshape(batch, patch_count - keep_count, 2)[..., 1]
        kept_indices = torch.gather(original_indices, 1, top_indices)
        deleted_indices = torch.gather(original_indices, 1, deleted_sequence_indices)
        return TokenGroupingResult(
            tokens=compact_tokens,
            kept_indices=kept_indices,
            deleted_indices=deleted_indices,
            deleted_sequence_indices=deleted_sequence_indices,
            keep_mask=keep_mask,
        )

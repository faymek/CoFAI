"""Semantic grouping and distribution calibration for ProTRAC."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def semantic_weights(features, prototypes, topm, temperature, eps=1e-6):
    similarity = F.normalize(features.float(), dim=-1, eps=eps) @ prototypes.t()
    values, indices = similarity.topk(topm, dim=-1, sorted=False)
    weights = torch.zeros_like(similarity)
    weights.scatter_(-1, indices, torch.softmax(values / temperature, dim=-1))
    return weights


def clip_delta(delta, reference, ratio, eps=1e-6):
    limit = ratio * reference.norm(dim=-1, keepdim=True)
    scale = (limit / delta.norm(dim=-1, keepdim=True).clamp_min(eps)).clamp(max=1)
    return delta * scale


def projection_delta(features, means, bases):
    centered = features - means
    coefficients = torch.einsum("...d,...dr->...r", centered, bases)
    return torch.einsum("...r,...dr->...d", coefficients, bases) - centered


@dataclass
class ProTRACStats:
    source: torch.Tensor
    mean: torch.Tensor
    count: torch.Tensor

    @classmethod
    def create(cls, source, prior=0.0):
        return cls(source, source.clone(), source.new_full(source.shape[:-1], prior))

    def reset(self, prior=0.0):
        self.mean.copy_(self.source)
        self.count.fill_(prior)

    def shift(self, weights):
        return weights @ (self.source - self.mean)

    def update(self, values, weights, min_mass=0.0, per_image=False, eps=1e-6):
        mass = weights.sum(dim=0)
        weighted_sum = weights.t() @ values
        update = mass >= min_mass if per_image else mass > eps
        contribution = torch.ones_like(mass[update]) if per_image else mass[update]
        count = self.count[update] + contribution
        incoming = weighted_sum[update]
        if per_image:
            incoming = incoming / mass[update, None].clamp_min(eps)
        self.mean[update] = (self.mean[update] * self.count[update, None] + incoming) / count[:, None].clamp_min(eps)
        self.count[update] = count

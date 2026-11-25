"""Frozen PQFCv2 stage 1 with trainable blockwise token transforms."""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cofai.entropy_models.static_categorical import StaticCategoricalEntropyModel
from cofai.entropy_models.vq_ufc_model import SequenceAnalysisTransform, SequenceSynthesisTransform
from cofai.ops.orfc import batched_assign


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Stage2Model(nn.Module):
    def __init__(self, stage1_path, block_tokens=241, latent_tokens=120):
        super().__init__()
        if not 0 < latent_tokens <= block_tokens:
            raise ValueError('latent_tokens must be between 1 and block_tokens')
        with np.load(stage1_path, allow_pickle=False) as f:
            for name, key in [('rotation', 'R'), ('codebooks', 'codebooks'), ('prefix_rms', 'prefix_position_channel_rms'), ('patch_rms', 'patch_channel_rms'), ('pmf', 'pmf')]:
                self.register_buffer(name, torch.from_numpy(np.array(f[key], dtype=np.float32)))
        self.groups, self.K, self.dim = self.codebooks.shape
        self.channels = self.groups * self.dim
        self.n_prefix = self.prefix_rms.shape[0]
        if self.rotation.shape != (self.channels, self.channels) or self.patch_rms.shape != (self.channels,) or self.prefix_rms.shape[1] != self.channels or self.pmf.shape != (self.groups, self.K):
            raise ValueError('inconsistent stage 1 artifact shapes')
        if any(not torch.isfinite(x).all() for x in (self.rotation, self.codebooks, self.prefix_rms, self.patch_rms, self.pmf)) or (self.prefix_rms <= 0).any() or (self.patch_rms <= 0).any() or (self.pmf < 0).any() or (self.pmf.sum(1) <= 0).any():
            raise ValueError('invalid stage 1 values')
        self.pmf.div_(self.pmf.sum(1, keepdim=True))
        self.block_tokens, self.latent_tokens = int(block_tokens), int(latent_tokens)
        self.analysis = SequenceAnalysisTransform(block_tokens, latent_tokens)
        self.synthesis = SequenceSynthesisTransform(latent_tokens, block_tokens)
        self.entropy = StaticCategoricalEntropyModel(self.pmf)

    def normalize(self, x):
        if x.ndim != 3 or x.shape[-1] != self.channels or x.shape[1] <= self.n_prefix:
            raise ValueError(f'bad input shape {tuple(x.shape)}')
        return torch.cat((x[:, :self.n_prefix] / self.prefix_rms, x[:, self.n_prefix:] / self.patch_rms), 1)

    def denormalize(self, x):
        return torch.cat((x[:, :self.n_prefix] * self.prefix_rms, x[:, self.n_prefix:] * self.patch_rms), 1)

    def hard_labels(self, y):
        b, n, c = y.shape
        rotated = (y.reshape(-1, c) @ self.rotation).reshape(-1, self.groups, self.dim).permute(1, 0, 2).contiguous()
        return batched_assign(rotated, self.codebooks, device=y.device)[1]

    def decode_labels(self, labels, batch, latent_length, length):
        idx = labels.unsqueeze(-1).expand(-1, -1, self.dim)
        q = torch.gather(self.codebooks, 1, idx).permute(1, 0, 2).reshape(batch, latent_length, self.channels)
        return self.denormalize(self.synthesis(q @ self.rotation.T, length))

    def forward(self, h, token_res=None, qp=0, **kwargs):
        y = self.analysis(self.normalize(h.float()))
        labels = self.hard_labels(y)
        return {'h_hat': self.decode_labels(labels, h.shape[0], y.shape[1], h.shape[1]), 'likelihoods': {'pq': self.entropy(labels)}}

    def train_forward(self, h, temperature=0.5, group_chunk=8):
        if temperature <= 0 or group_chunk <= 0:
            raise ValueError('temperature and group_chunk must be positive')
        z = self.analysis(self.normalize(h.float()))
        b, n, c = z.shape
        rotated = (z.reshape(-1, c) @ self.rotation).reshape(-1, self.groups, self.dim).permute(1, 0, 2)
        outputs, rates = [], []
        for start in range(0, self.groups, group_chunk):
            v = rotated[start:start + group_chunk]
            cb = self.codebooks[start:start + group_chunk]
            # Checkpoint avoids retaining all group distance matrices in GPU memory.
            def quantize(v, cb, pmf):
                dist = (v.square().sum(-1, keepdim=True) + cb.square().sum(-1).unsqueeze(1) - 2 * torch.bmm(v, cb.transpose(1, 2))).clamp_min(0)
                labels = dist.argmin(-1)
                soft = torch.softmax(-dist / temperature, -1)
                hard = F.one_hot(labels, self.K).to(soft.dtype)
                weights = hard - soft.detach() + soft
                q = torch.bmm(weights, cb)
                rate = (weights * (-torch.log2(pmf.clamp_min(1e-12))).unsqueeze(1)).sum()
                return q, rate
            from torch.utils.checkpoint import checkpoint
            q, rate = checkpoint(quantize, v, cb, self.pmf[start:start + group_chunk], use_reentrant=False)
            outputs.append(q); rates.append(rate)
        y = torch.cat(outputs, 0).permute(1, 0, 2).reshape(b, n, c) @ self.rotation.T
        return self.denormalize(self.synthesis(y, h.shape[1])), torch.stack(rates).sum()

    def compress(self, h, token_res=None, qp=0, **kwargs):
        y = self.analysis(self.normalize(h.float()))
        labels = self.hard_labels(y)
        return {'strings': {'pq': [[s] for s in self.entropy.compress(labels)]}, 'pstate': {'shape': tuple(map(int, h.shape)), 'latent_length': int(y.shape[1])}}

    def decompress(self, strings, pstate, **kwargs):
        b, length, c = map(int, pstate['shape']); n = int(pstate['latent_length'])
        if c != self.channels or n != math.ceil(length / self.block_tokens) * self.latent_tokens:
            raise ValueError('inconsistent coded shape')
        labels = self.entropy.decompress([s[0] for s in strings['pq']], num_samples=b*n)
        return {'h_hat': self.decode_labels(labels, b, n, length)}


class Pqfcv2Stage2Codec(Stage2Model):
    def __init__(self, stage1_path, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['stage1_sha256'] != file_sha256(stage1_path):
            raise ValueError('stage 1 artifact does not match stage 2 checkpoint')
        super().__init__(stage1_path, checkpoint['block_tokens'], checkpoint['latent_tokens'])
        self.analysis.load_state_dict(checkpoint['analysis'])
        self.synthesis.load_state_dict(checkpoint['synthesis'])

        if 'rotation' in checkpoint:
            raise ValueError('rotation fine-tuning checkpoints are not supported by the frozen stage-1 baseline codec')

import copy

import numpy as np
import pytest
import torch

from cofai.backbone.frozen_tail import FrozenTail
from examples.orfc_2446.offline.soft_pq import FeatureCodec, SoftPQ
from examples.pqfcv2.codec import PqfCv2OpqCodec
from examples.pqfcv2.train_stage1 import accumulate_token_normalized_batch


def test_soft_rate_has_assignment_gradient_and_hard_forward():
    torch.manual_seed(7)
    model = SoftPQ(2, 4, 2, lmbda=3.0, soft_rate=True)
    model.temperature = 0.8
    with torch.no_grad():
        model.log_prior.copy_(torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 0.0, 2.0]]))
    x = torch.randn(1, 5, 4, requires_grad=True)
    model(x)
    bits = -torch.log2(model.log_prior.softmax(-1))
    expected = bits.gather(1, model._last_labels).sum(0).mean()
    torch.testing.assert_close(model._last_rate, expected)
    model._last_rate.backward()
    assert x.grad.abs().sum() > 0
    assert model.codebooks.grad.abs().sum() > 0


@pytest.mark.parametrize("relative_vector", [False, True])
def test_token_microbatches_match_full_batch_loss_gradients(relative_vector):
    torch.manual_seed(8)
    pq = SoftPQ(2, 5, 2, lmbda=5.0, soft_rate=True)
    pq.temperature = 0.7
    with torch.no_grad():
        pq.log_prior.copy_(torch.randn_like(pq.log_prior))
    small, full = FeatureCodec(pq), FeatureCodec(copy.deepcopy(pq))
    tail = FrozenTail([], torch.nn.LayerNorm(4))
    raw = torch.randn(2, 7, 4)
    prefix, patch = torch.ones(1, 4), torch.ones(4)
    a = accumulate_token_normalized_batch(raw, small, prefix, patch, 1, 3.0, 0.5, tail, 4, relative_vector, 0.4)
    b = accumulate_token_normalized_batch(raw, full, prefix, patch, 1, 3.0, 0.5, tail, 14, relative_vector, 0.4)
    torch.testing.assert_close(torch.tensor(a[:3]), torch.tensor(b[:3]), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a[3], b[3])
    for p, q in zip(small.parameters(), full.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=3e-6)


def test_ecvq_assignment_and_real_rans_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    artifact = tmp_path / "codec.npz"
    rms = tmp_path / "rms.npz"
    pmf = np.array([[0.7, 0.2, 0.09, 0.01], [0.02, 0.08, 0.3, 0.6]], dtype=np.float32)
    np.savez(artifact, R=np.eye(4, dtype=np.float32),
             codebooks=rng.normal(size=(2, 4, 2)).astype(np.float32), pmf=pmf, lmbda=2.0)
    np.savez(rms, patch_channel_rms=np.ones(4, dtype=np.float32))
    codec = PqfCv2OpqCodec(str(artifact), str(rms))
    tokens = torch.from_numpy(rng.normal(size=(1, 13, 4)).astype(np.float32))
    labels = codec._labels(tokens)
    subvectors = tokens.reshape(13, 2, 2).permute(1, 0, 2)
    expected = (torch.cdist(subvectors, codec.codebooks).square() + codec.rans_bits[:, None] / 2.0).argmin(-1)
    torch.testing.assert_close(labels, expected)
    packed = codec.compress(tokens)
    decoded = codec.decompress(packed["strings"], packed["pstate"])["h_hat"]
    torch.testing.assert_close(decoded, codec._tokens_from_labels(labels, 1, 13))


def test_domain_oversampling_retains_original_training_files(tmp_path):
    from collections import Counter
    from examples.pqfcv2.train_stage1 import oversample_train_domain

    files = [tmp_path / 'ade' / f'{i}.npy' for i in range(10)]
    files += [tmp_path / 'nyu' / f'{i}.npy' for i in range(8)]
    sampled = oversample_train_domain(files, tmp_path, 'ade', 0.5, 42)
    assert sampled == oversample_train_domain(files, tmp_path, 'ade', 0.5, 42)
    counts = Counter(sampled)
    assert len(sampled) == 23 and set(sampled) == set(files)
    assert sum(counts[p] - 1 for p in files if p.parent.name == 'ade') == 5
    assert all(counts[p] == 1 for p in files if p.parent.name == 'nyu')

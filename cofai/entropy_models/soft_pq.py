"""
Differentiable Soft Product Quantization (orfc_2446).

Adapted from ORFC/coding/orfc/soft_pq.py for integration into CoFAI.
Uses CoFAI's existing batch normalization utilities.

Components:
- FeatureTransform: Linear bottleneck D_in -> D_out (dimensionality reduction)
- OrthogonalTransform: Cayley-parametrised differentiable rotation R in SO(D)
- SoftPQ: Differentiable PQ with straight-through softmax assignment
- FeatureCodec: Composes transform + PQ into a full codec
- FrozenTail / CLIPFrozenTail / Siglip2FrozenTail: Frozen ViT tail wrappers
- train_soft_pq: End-to-end codec training loop
- soft_pq_encode_decode: Evaluation-time hard PQ encode/decode
- save_codec / load_codec: Serialization utilities
"""

import math
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_fn
from torch.utils.data import Dataset, DataLoader

from cofai.entropy_models.orfc_model import batch_normalize_gpu, batch_inv_normalize_gpu, batched_kmeans


# ================================================================
#                    Feature Transform (bottleneck D_in > D_out)
# ================================================================

class FeatureTransform(nn.Module):
    """Linear encoder/decoder for dimensionality reduction (D_in > D_out).

    For same-dim orthogonal rotation, use OrthogonalTransform instead.
    """

    def __init__(self, D_in, D_out):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.encoder = nn.Linear(D_in, D_out, bias=False)
        self.decoder = nn.Linear(D_out, D_in, bias=False)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=1.0)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


# ================================================================
#                    Orthogonal Transform (Cayley parametrisation)
# ================================================================

class OrthogonalTransform(nn.Module):
    """Differentiable orthogonal rotation via Cayley parametrisation.

    R = (I - A)^{-1}(I + A) where A is skew-symmetric.
    Guarantees R^T R = I throughout training.
    Free parameters: D(D-1)/2 upper-triangular entries of A.
    """

    def __init__(self, D):
        super().__init__()
        self.D = D
        n_params = D * (D - 1) // 2
        self.triu_params = nn.Parameter(torch.zeros(n_params))
        idx = torch.triu_indices(D, D, offset=1)
        self.register_buffer('_triu_row', idx[0])
        self.register_buffer('_triu_col', idx[1])

    def _get_skew(self):
        A = torch.zeros(self.D, self.D,
                        device=self.triu_params.device,
                        dtype=self.triu_params.dtype)
        A[self._triu_row, self._triu_col] = self.triu_params
        return A - A.t()

    def get_rotation(self):
        A = self._get_skew()
        I = torch.eye(self.D, device=A.device, dtype=A.dtype)
        return torch.linalg.solve(I - A, I + A)

    def encode(self, x):
        R = self.get_rotation()
        return x @ R

    def decode(self, x):
        R = self.get_rotation()
        return x @ R.t()

    def init_from_opq(self, R_np):
        """Warm-start from an orthogonal matrix R [D, D] (numpy)."""
        R = torch.from_numpy(R_np.astype(np.float64))
        if torch.linalg.det(R) < 0:
            R = R.clone()
            R[:, -1] *= -1
        I = torch.eye(self.D, dtype=torch.float64)
        A = torch.linalg.solve(R + I, R - I)
        A = (A - A.T) / 2
        triu_idx = np.triu_indices(self.D, k=1)
        triu_vals = A[triu_idx[0], triu_idx[1]].float()
        with torch.no_grad():
            self.triu_params.copy_(triu_vals)

    @torch.no_grad()
    def orth_error(self):
        """||R^T R - I||_F — diagnostic."""
        R = self.get_rotation()
        I = torch.eye(self.D, device=R.device, dtype=R.dtype)
        return (R.t() @ R - I).norm().item()


# ================================================================
#                    SoftPQ Module
# ================================================================

class SoftPQ(nn.Module):
    """Differentiable Product Quantisation.

    Training (temperature > 0):
      Straight-through softmax — forward is hard argmin, backward flows
      through temperature-scaled softmax.

    Eval (temperature == 0):
      Standard hard argmin PQ.
    """

    def __init__(self, G, K, d, lmbda=0.0, prior_floor=0.0):
        super().__init__()
        self.G = G
        self.K = K
        self.d = d
        self.D = G * d
        self.lmbda = lmbda
        self.use_rate = (lmbda > 0)
        self.prior_floor = prior_floor
        self.temperature = 0.0

        self.codebooks = nn.Parameter(torch.randn(G, K, d) * 0.01)
        self._last_rate = None
        self._last_rate_per_group = None
        self._last_labels = None
        if self.use_rate:
            self.log_prior = nn.Parameter(torch.zeros(G, K))

    def init_from_kmeans(self, Z_flat, device='cuda', max_iter=100):
        """Initialise codebooks with batched k-means."""
        if isinstance(Z_flat, np.ndarray):
            Z_flat = torch.from_numpy(Z_flat).float()
        Z_flat = Z_flat.to(device)
        N = Z_flat.shape[0]
        sub = Z_flat.reshape(N, self.G, self.d).permute(1, 0, 2).contiguous()
        del Z_flat
        with torch.no_grad():
            centroids = batched_kmeans(sub, self.K,
                                       max_iter=max_iter,
                                       device=device, verbose=False)
            self.codebooks.data.copy_(centroids.cpu())
            del centroids, sub
        gc.collect()
        torch.cuda.empty_cache()

    def init_codebooks(self, codebooks_list):
        """Warm-start codebooks from a list of [K, d] numpy arrays."""
        with torch.no_grad():
            for g, c_g in enumerate(codebooks_list):
                self.codebooks.data[g] = torch.from_numpy(c_g).float()

    def init_prior_from_freq(self, usage_counts, smoothing=1.0):
        """Initialise log_prior from empirical assignment frequency."""
        if not self.use_rate:
            return
        if isinstance(usage_counts, torch.Tensor):
            usage_counts = usage_counts.cpu().numpy()
        counts = usage_counts.astype(np.float64) + smoothing
        freq = counts / counts.sum(axis=-1, keepdims=True)
        lp = np.log(freq + 1e-30).astype(np.float32)
        with torch.no_grad():
            self.log_prior.data.copy_(torch.from_numpy(lp))

    def _quantise(self, Z_flat):
        """ECVQ assignment with optional soft straight-through gradient."""
        N = Z_flat.shape[0]
        C = self.codebooks
        sub_g = Z_flat.reshape(N, self.G, self.d).permute(1, 0, 2)  # [G, N, d]

        dists_sq = torch.cdist(sub_g, C).pow(2)  # [G, N, K]

        log2_pmf = None
        if self.use_rate:
            log_p = F_fn.log_softmax(self.log_prior, dim=-1)
            if self.prior_floor > 0:
                p = log_p.exp()
                p = (1.0 - self.prior_floor) * p + self.prior_floor / self.K
                log2_pmf = -(p + 1e-30).log() / math.log(2)
            else:
                log2_pmf = -log_p / math.log(2)
            cost = dists_sq + log2_pmf.unsqueeze(1) / self.lmbda
        else:
            cost = dists_sq

        labels = cost.argmin(dim=-1)  # [G, N]
        self._last_labels = labels.detach()

        if self.training and self.temperature > 0:
            logits = -cost / self.temperature
            soft = F_fn.softmax(logits, dim=-1)
            hard = torch.zeros_like(soft).scatter_(
                -1, labels.unsqueeze(-1), 1.0)
            weights = hard - soft.detach() + soft  # ST trick
            Z_hat_g = torch.einsum('gnk,gkd->gnd', weights, C)
        else:
            Z_hat_g = torch.gather(
                C.unsqueeze(1).expand(-1, N, -1, -1), 2,
                labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.d),
            ).squeeze(2)

        Z_hat_flat = Z_hat_g.permute(1, 0, 2).reshape(N, self.D)

        ones = torch.ones(N, device=Z_flat.device)
        usage = torch.zeros(self.G, self.K, device=Z_flat.device)
        usage.scatter_add_(1, labels, ones.unsqueeze(0).expand(self.G, -1))

        if self.use_rate:
            gathered_rate = torch.gather(log2_pmf, 1, labels)
            self._last_rate = gathered_rate.sum(0).mean()
            self._last_rate_per_group = gathered_rate.mean(1)
        else:
            self._last_rate = torch.tensor(0.0, device=Z_flat.device)
            self._last_rate_per_group = None

        return Z_hat_flat, usage

    def forward(self, Z_norm):
        """Z_norm [B, T, D'] -> Z_hat [B, T, D']."""
        B, T, Dp = Z_norm.shape
        Z_hat, usage = self._quantise(Z_norm.reshape(B * T, Dp))
        return Z_hat.reshape(B, T, Dp), usage

    @torch.no_grad()
    def get_prior_pmf(self):
        """Return [G, K] numpy PMF array."""
        if not self.use_rate:
            return np.full((self.G, self.K), 1.0 / self.K)
        return F_fn.softmax(self.log_prior, dim=-1).cpu().numpy()


# ================================================================
#                    FeatureCodec (transform + PQ)
# ================================================================

class FeatureCodec(nn.Module):
    """Composes an optional OrthogonalTransform with a SoftPQ quantiser.

    Pipeline:  Y_norm --[encode]--> Z --[PQ]--> Z_hat --[decode]--> Y_hat
    """

    def __init__(self, pq, transform=None):
        super().__init__()
        self.pq = pq
        self.transform = transform

    def forward(self, Y_norm):
        """Y_norm [B, T, D] -> Y_hat [B, T, D]."""
        B, T, D = Y_norm.shape
        flat = Y_norm.reshape(B * T, D)
        if self.transform is not None and hasattr(self.transform, 'get_rotation'):
            R = self.transform.get_rotation()
            Z = flat @ R
            Z_hat, usage = self.pq._quantise(Z)
            Y_hat = Z_hat @ R.t()
        else:
            Z = self.transform.encode(flat) if self.transform else flat
            Z_hat, usage = self.pq._quantise(Z)
            Y_hat = self.transform.decode(Z_hat) if self.transform else Z_hat
        return Y_hat.reshape(B, T, D), usage

    @property
    def use_rate(self):
        return self.pq.use_rate

    @property
    def _last_rate(self):
        return self.pq._last_rate

    @property
    def lmbda(self):
        return self.pq.lmbda

    @torch.no_grad()
    def get_prior_pmf(self):
        return self.pq.get_prior_pmf()


# ================================================================
#                    Frozen tail wrapper
# ================================================================

class FrozenTail:
    """Wraps tail ViT blocks + final norm for loss computation.

    Used during training to compute distortion:
      D = ||FrozenTail(H) - FrozenTail(H_hat)||^2
    """

    def __init__(self, blocks, norm_layer, device='cuda'):
        self.blocks = list(blocks)
        self.norm = norm_layer
        self.device = device
        for blk in self.blocks:
            blk.to(device).eval()
            for p in blk.parameters():
                p.requires_grad_(False)
        self.norm.to(device).eval()
        for p in self.norm.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    @torch.no_grad()
    def forward_nograd(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def to(self, device):
        for blk in self.blocks:
            blk.to(device)
        self.norm.to(device)
        self.device = device
        return self


class CLIPFrozenTail:
    """Wraps CLIP tail resblocks + ln_post for loss computation.

    CLIP resblocks use [L, B, D] internally, while the codec pipeline
    works with [B, T, D].  This class handles the permutation transparently.
    """

    def __init__(self, resblocks, ln_post, device='cuda'):
        self.resblocks = list(resblocks)
        self.ln_post = ln_post
        self.device = device
        for blk in self.resblocks:
            blk.to(device).eval()
            for p in blk.parameters():
                p.requires_grad_(False)
        self.ln_post.to(device).eval()
        for p in self.ln_post.parameters():
            p.requires_grad_(False)

    def _forward(self, x):
        x = x.permute(1, 0, 2).contiguous()
        for blk in self.resblocks:
            x = blk(x)
        x = x.permute(1, 0, 2).contiguous()
        x = self.ln_post(x)
        return x

    def __call__(self, x):
        return self._forward(x)

    @torch.no_grad()
    def forward_nograd(self, x):
        return self._forward(x)

    def to(self, device):
        for blk in self.resblocks:
            blk.to(device)
        self.ln_post.to(device)
        self.device = device
        return self


class Siglip2FrozenTail:
    """Wraps SigLIP2 encoder tail layers + post_layernorm for loss computation.

    SigLIP2 encoder layers take [B, N, D] directly (no permute),
    require attention_mask=None, and return (hidden_states, ...) tuples.
    """

    def __init__(self, layers, post_ln, device='cuda'):
        self.layers = list(layers)
        self.post_ln = post_ln
        self.device = device
        for blk in self.layers:
            blk.to(device).eval()
            for p in blk.parameters():
                p.requires_grad_(False)
        self.post_ln.to(device).eval()
        for p in self.post_ln.parameters():
            p.requires_grad_(False)

    def _forward(self, x):
        for blk in self.layers:
            out = blk(x, attention_mask=None)
            x = out[0] if isinstance(out, tuple) else out
        x = self.post_ln(x)
        return x

    def __call__(self, x):
        return self._forward(x)

    @torch.no_grad()
    def forward_nograd(self, x):
        return self._forward(x)

    def to(self, device):
        for blk in self.layers:
            blk.to(device)
        self.post_ln.to(device)
        self.device = device
        return self


# ================================================================
#                    Dataset / Utilities
# ================================================================

class FeatureDataset(Dataset):
    """Wraps a pre-stacked [N, T, D] numpy array for DataLoader prefetch."""

    def __init__(self, features_array, teacher_cache=None):
        self.data = features_array
        self.teacher_cache = teacher_cache

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float()
        if self.teacher_cache is not None:
            return x, torch.from_numpy(self.teacher_cache[idx]).float()
        return x


def compute_perplexity(usage):
    """Compute per-group perplexity from usage counts [G, K]."""
    p = usage / usage.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    entropy = -(p * (p + 1e-10).log()).sum(dim=-1)
    return entropy.exp().mean().item()


# ================================================================
#                    Training
# ================================================================

def _is_norm_only_tail(tail) -> bool:
    """True when frozen tail has no blocks (final-layer split, norm only)."""
    return tail is not None and len(getattr(tail, "blocks", [])) == 0


def _patch_only_active(train_tokens: str, n_prefix: int) -> bool:
    return train_tokens == "patch" and n_prefix > 0


def codec_forward(Y, codec, n_prefix: int = 0, prefix_bypass: bool = False):
    """Forward codec; optionally keep prefix tokens identity in norm space."""
    if prefix_bypass and n_prefix > 0 and n_prefix < Y.shape[1]:
        Y_hat = Y.clone()
        patch_hat, usage = codec(Y[:, n_prefix:, :])
        Y_hat[:, n_prefix:, :] = patch_hat
        return Y_hat, usage
    return codec(Y)


def train_soft_pq(
    features_train,
    tail,
    G, K, d,
    norm_mode='per_image',
    n_prefix=0,
    epochs=100,
    lr=1e-3,
    batch_size=4,
    device='cuda',
    seed=42,
    val_features=None,
    verbose=True,
    transform=None,
    R_init=None,
    codebooks_init=None,
    kmeans_max_samples=2_000_000,
    use_mse_loss=False,
    lmbda=0.0,
    prior_init_counts=None,
    grad_clip=1.0,
    freeze_transform=False,
    freeze_codebooks=False,
    prior_floor=0.0,
    tau_start=1.0,
    tau_end=0.01,
    tau_schedule='exponential',
    precompute_teacher=None,
    train_tokens='all',
):
    """Train FeatureCodec (transform + PQ) to minimise J = R + lambda * D.

    Args:
        train_tokens: ``'all'`` (default) or ``'patch'`` — when ``'patch'`` and
            ``n_prefix > 0``, OPQ/SoftPQ forward, distortion, and rate use patch
            tokens only; prefix is identity in norm space.

    Returns:
        codec: trained FeatureCodec module.
        history: list of epoch dicts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    patch_only = _patch_only_active(train_tokens, n_prefix)

    N_img = len(features_train)
    D = features_train[0].shape[1]
    if isinstance(features_train, np.ndarray):
        features_array = features_train
    else:
        features_array = np.stack(features_train)
    pq = SoftPQ(G, K, d, lmbda=lmbda, prior_floor=prior_floor).to(device)
    if transform is not None:
        transform = transform.to(device)
    codec = FeatureCodec(pq, transform).to(device)

    _use_soft = (tau_start > 0)

    if R_init is not None and codebooks_init is not None and transform is not None:
        if verbose:
            print(f"  Warm-start from OPQ (transform + codebooks)")
        transform.init_from_opq(R_init)
        pq.init_codebooks(codebooks_init)
        if pq.use_rate and prior_init_counts is not None:
            pq.init_prior_from_freq(prior_init_counts)
            if verbose:
                print(f"  log_prior init from OPQ empirical frequency")
    elif codebooks_init is not None:
        if verbose:
            print(f"  Warm-start codebooks only")
        pq.init_codebooks(codebooks_init)
        if pq.use_rate and prior_init_counts is not None:
            pq.init_prior_from_freq(prior_init_counts)
    else:
        if verbose:
            scope = "patch tokens" if patch_only else f"{N_img} images"
            print(f"  K-means init for codebooks ({scope})...")
        all_Z = []
        for start in range(0, N_img, 200):
            end = min(start + 200, N_img)
            X = torch.from_numpy(features_array[start:end]).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
                if patch_only:
                    flat = Y[:, n_prefix:, :].reshape(-1, D)
                else:
                    flat = Y.reshape(-1, D)
                Z = transform.encode(flat) if transform else flat
            all_Z.append(Z.cpu())
            del X, Y, flat, Z
        Z_flat = torch.cat(all_Z, dim=0)
        max_km = kmeans_max_samples
        if Z_flat.shape[0] > max_km:
            idx = np.random.choice(Z_flat.shape[0], max_km, replace=False)
            Z_flat = Z_flat[idx]
        pq.init_from_kmeans(Z_flat, device=device)
        del all_Z, Z_flat
        if verbose:
            print(f"  K-means init done.")
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated(device) / 1e9
                reserved = torch.cuda.memory_reserved(device) / 1e9
                print(f"  CUDA after k-means: alloc={alloc:.2f}GB reserved={reserved:.2f}GB")

    if freeze_transform and transform is not None:
        for p in transform.parameters():
            p.requires_grad_(False)
        if verbose:
            n_p = sum(p.numel() for p in transform.parameters())
            print(f"  Frozen: transform ({n_p:,} params)")
    if freeze_codebooks:
        pq.codebooks.requires_grad_(False)
        if pq.use_rate:
            pq.log_prior.requires_grad_(False)
        if verbose:
            print(f"  Frozen: codebooks ({pq.codebooks.numel():,} params)")

    trainable = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    if verbose:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_trainable:,}")
        if transform is not None and hasattr(transform, 'orth_error'):
            print(f"  Orthogonal transform: ||R'R-I||={transform.orth_error():.2e}")
        if _use_soft:
            print(f"  Soft PQ: tau {tau_start:.2f} -> {tau_end:.4f} ({tau_schedule})")
        else:
            print(f"  Hard PQ (tau=0)")
        if patch_only:
            print(f"  Patch-only training: n_prefix={n_prefix} (prefix bypass in norm space)")

    if precompute_teacher is None:
        precompute_teacher = not _is_norm_only_tail(tail)

    teacher_cache = None
    if not use_mse_loss:
        if precompute_teacher:
            if verbose:
                print(f"  Pre-computing teacher outputs ({N_img} images)...")
            t_pre = time.time()
            teacher_cache = np.empty_like(features_array)
            with torch.no_grad():
                for start in range(0, N_img, batch_size):
                    end = min(start + batch_size, N_img)
                    X_chunk = torch.from_numpy(
                        features_array[start:end]).float().to(device)
                    teacher_cache[start:end] = tail.forward_nograd(
                        X_chunk).cpu().numpy()
                    del X_chunk
            torch.cuda.empty_cache()
            if verbose:
                cache_gb = teacher_cache.nbytes / 1e9
                print(f"  Teacher cache: {cache_gb:.1f} GB CPU "
                      f"({time.time() - t_pre:.1f}s)")
        elif verbose:
            print("  Skipping teacher precompute (norm-only tail; on-the-fly per batch)")

    val_array = None
    val_teacher_cache = None
    if val_features is not None and len(val_features) > 0:
        if isinstance(val_features, np.ndarray):
            val_array = val_features
        else:
            val_array = np.stack(val_features)
        if not use_mse_loss and precompute_teacher:
            n_val = len(val_features)
            val_teacher_cache = np.empty_like(val_array)
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    X_v = torch.from_numpy(
                        val_array[vs:ve]).float().to(device)
                    val_teacher_cache[vs:ve] = tail.forward_nograd(
                        X_v).cpu().numpy()
                    del X_v
            torch.cuda.empty_cache()

    train_dataset = FeatureDataset(features_array, teacher_cache=teacher_cache)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    history = []

    for epoch in range(epochs):
        t_epoch = time.time()

        if _use_soft and epochs > 1:
            progress = epoch / (epochs - 1)
            if tau_schedule == 'linear':
                tau = tau_start + (tau_end - tau_start) * progress
            else:
                tau = tau_start * (tau_end / tau_start) ** progress
            pq.temperature = tau
        elif _use_soft:
            pq.temperature = tau_start
        else:
            pq.temperature = 0.0

        total_distortion = 0.0
        total_rate = 0.0
        total_tokens = 0
        usage_acc = torch.zeros(G, K, device=device)

        codec.train()
        for batch in train_loader:
            if teacher_cache is not None:
                X, Y_teacher = batch
                X = X.to(device, non_blocking=True)
                Y_teacher = Y_teacher.to(device, non_blocking=True)
            else:
                X = batch.to(device, non_blocking=True)
                Y_teacher = None
            B = X.shape[0]
            T = X.shape[1]

            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)

            Y_hat, usage = codec_forward(
                Y, codec, n_prefix, prefix_bypass=patch_only,
            )

            if patch_only:
                n_rate_tok = T - n_prefix
                y_ref = Y[:, n_prefix:, :]
                y_hat_ref = Y_hat[:, n_prefix:, :]
            else:
                n_rate_tok = T
                y_ref = Y
                y_hat_ref = Y_hat

            if use_mse_loss:
                distortion = ((y_ref - y_hat_ref) ** 2).sum() / B
            else:
                if Y_teacher is None:
                    with torch.no_grad():
                        Y_teacher = tail.forward_nograd(X)
                X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
                X_hat_out = tail(X_hat)
                if patch_only:
                    distortion = (
                        (Y_teacher[:, n_prefix:] - X_hat_out[:, n_prefix:]) ** 2
                    ).sum() / B
                else:
                    distortion = ((Y_teacher - X_hat_out) ** 2).sum() / B
                del Y_teacher, X_hat, X_hat_out

            if codec.use_rate:
                rate_bits = codec._last_rate * n_rate_tok
                loss = rate_bits + lmbda * distortion
            else:
                loss = distortion

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), grad_clip)
            optimizer.step()

            total_distortion += distortion.item() * B
            if codec.use_rate:
                total_rate += codec._last_rate.item() * n_rate_tok * B
            total_tokens += B * n_rate_tok
            usage_acc += usage.detach()

            del X, Y, Mu, Std, Y_hat, loss, distortion

        scheduler.step()

        avg_distortion = total_distortion / N_img
        avg_rate = total_rate / N_img if codec.use_rate else 0.0
        perplexity = compute_perplexity(usage_acc)
        dead_entries = int((usage_acc == 0).sum().item())

        val_loss = None
        if val_array is not None:
            val_loss_sum = 0.0
            n_val = val_array.shape[0]
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    Bv = X_v.shape[0]
                    Y_v, Mu_v, Std_v = batch_normalize_gpu(
                        X_v, mode=norm_mode, n_prefix=n_prefix
                    )
                    Yh_v, _ = codec_forward(
                        Y_v, codec, n_prefix, prefix_bypass=patch_only,
                    )
                    if use_mse_loss:
                        if patch_only:
                            d_v = ((Y_v[:, n_prefix:] - Yh_v[:, n_prefix:]) ** 2).sum()
                        else:
                            d_v = ((Y_v - Yh_v) ** 2).sum()
                        val_loss_sum += d_v.item() / Bv * Bv
                    else:
                        if val_teacher_cache is not None:
                            Yt_v = torch.from_numpy(
                                val_teacher_cache[vs:ve]).float().to(device)
                        else:
                            Yt_v = tail.forward_nograd(X_v)
                        Xh_v = batch_inv_normalize_gpu(Yh_v, Mu_v, Std_v)
                        Xo_v = tail.forward_nograd(Xh_v)
                        if patch_only:
                            d_v = (
                                (Yt_v[:, n_prefix:] - Xo_v[:, n_prefix:]) ** 2
                            ).sum()
                        else:
                            d_v = ((Yt_v - Xo_v) ** 2).sum()
                        val_loss_sum += d_v.item() / Bv * Bv
                        del Yt_v, Xh_v, Xo_v
                    del X_v, Y_v, Mu_v, Std_v, Yh_v
            val_loss = val_loss_sum / n_val

        T_tokens = features_train[0].shape[0]
        T_rate = T_tokens - n_prefix if patch_only else T_tokens
        rate_bits_bpt = avg_rate
        rate_per_image = avg_rate * T_rate if codec.use_rate else 0.0

        epoch_info = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'loss_distortion': avg_distortion,
            'perplexity': perplexity,
            'val_loss': val_loss,
            'rate_bits': rate_bits_bpt,
            'rate_per_image': rate_per_image,
            'dead_entries': dead_entries,
            'temperature': pq.temperature,
            'time': time.time() - t_epoch,
        }
        if transform is not None and hasattr(transform, 'orth_error'):
            epoch_info['orth_error'] = transform.orth_error()
        if pq.use_rate and pq._last_rate_per_group is not None:
            epoch_info['rate_per_group'] = pq._last_rate_per_group.detach().cpu().tolist()
        history.append(epoch_info)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            val_str = f"  val={val_loss:.1f}" if val_loss else ""
            rate_str = ""
            if codec.use_rate:
                rate_str = (f"  R={rate_bits_bpt:.2f}b/t"
                            f"  lD={lmbda * avg_distortion:.1f}"
                            f"  dead={dead_entries}")
            tau_str = f"  tau={pq.temperature:.4f}" if _use_soft else ""
            print(f"  ep {epoch:3d}/{epochs}  "
                  f"D={avg_distortion:.1f}  ppl={perplexity:.1f}"
                  f"{rate_str}{tau_str}{val_str}  ({time.time() - t_epoch:.1f}s)")

    return codec, history


# ================================================================
#                    Hard-PQ encode/decode for evaluation
# ================================================================

def soft_pq_encode_decode(features, codec, norm_mode, device,
                          chunk_images=None, n_prefix=0, prefix_bypass=False):
    """Encode/decode features using trained FeatureCodec (hard PQ at eval)."""
    codec.eval()
    N = len(features)

    if chunk_images is None:
        tokens_per_img = features[0].shape[0]
        G, K = codec.pq.G, codec.pq.K
        max_tokens = max(tokens_per_img, int(3e9 / (G * K * 4)))
        chunk_images = max(1, min(200, max_tokens // tokens_per_img))

    all_xhat = []
    with torch.no_grad():
        for start in range(0, N, chunk_images):
            end = min(start + chunk_images, N)
            X = torch.from_numpy(np.stack(features[start:end])).float().to(device)
            B = X.shape[0]
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
            Y_hat, _ = codec_forward(
                Y, codec, n_prefix, prefix_bypass=prefix_bypass,
            )
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            for i in range(B):
                all_xhat.append(X_hat[i].cpu().numpy())
            del X, Y, Mu, Std, Y_hat, X_hat
        torch.cuda.empty_cache()
    return all_xhat


# ================================================================
#                    Codec save / load
# ================================================================

def save_codec(codec, path, **extra_meta):
    """Save codec architecture params + state dict."""
    pq = codec.pq
    meta = {
        'G': pq.G, 'K': pq.K, 'd': pq.d,
        'lmbda': pq.lmbda, 'prior_floor': pq.prior_floor,
        'has_transform': codec.transform is not None,
        'transform_type': (type(codec.transform).__name__
                           if codec.transform else None),
    }
    meta.update(extra_meta)
    if codec.transform is not None and hasattr(codec.transform, 'D'):
        meta['D'] = codec.transform.D
    elif codec.transform is not None and hasattr(codec.transform, 'D_in'):
        meta['D_in'] = codec.transform.D_in
        meta['D_out'] = codec.transform.D_out
    meta['state_dict'] = codec.state_dict()
    torch.save(meta, path)


def load_codec_meta(path):
    """Load checkpoint metadata without building the module."""
    return torch.load(path, map_location='cpu', weights_only=False)


def load_codec(path, device='cuda'):
    """Reconstruct a FeatureCodec from a saved checkpoint."""
    meta = load_codec_meta(path)
    G, K, d = meta['G'], meta['K'], meta['d']
    pq = SoftPQ(G, K, d,
                lmbda=meta.get('lmbda', 0.0),
                prior_floor=meta.get('prior_floor', 0.0))
    transform = None
    if meta.get('has_transform'):
        ttype = meta.get('transform_type')
        if ttype == 'OrthogonalTransform':
            transform = OrthogonalTransform(meta['D'])
        elif ttype == 'FeatureTransform':
            transform = FeatureTransform(meta['D_in'], meta['D_out'])
    codec = FeatureCodec(pq, transform)
    codec.load_state_dict(meta['state_dict'])
    return codec.to(device).eval()

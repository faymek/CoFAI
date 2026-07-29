"""Numerical operations for orthogonal-rotation product quantization.

Reference: Ge T. et al., "Optimized Product Quantization", TPAMI 2014

These stateless tensor routines are shared by offline ORFC training and the
runtime latent codec. They intentionally contain no probability model,
bitstream format, or training loop.
"""

from __future__ import annotations

import numpy as np
import torch


__all__ = [
    "batched_kmeans",
    "batched_assign",
    "learn_pca_rotation",
    "learn_orfc_rotation",
]


def _as_device_tensor(x, device):
    """Convert a NumPy array or tensor to float32 on ``device``."""
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)


# ================================================================
#                    GPU Batch K-Means
# ================================================================


def batched_kmeans(sub_vectors_3d, K, max_iter=100, device="cuda", verbose=False):
    """
    GPU batch k-means: run k-means on G groups simultaneously.

    Args:
        sub_vectors_3d: [G, N, dim] numpy or GPU tensor
        K:              codebook size
        max_iter:       max iterations
        device:         GPU device
        verbose:        print progress

    Returns:
        centroids: [G, K, dim] GPU tensor float32
    """
    X = _as_device_tensor(sub_vectors_3d, device)
    G, N, dim = X.shape

    max_mem_bytes = 1 * 1024**3
    chunk_size = max(1, min(N, max_mem_bytes // (G * K * 4)))

    init_indices = torch.stack([torch.randperm(N, device=device)[:K] for _ in range(G)])
    centroids = torch.gather(X, 1, init_indices.unsqueeze(-1).expand(-1, -1, dim))

    offset = torch.arange(G, device=device).unsqueeze(1) * K

    for it in range(max_iter):
        flat_sums = torch.zeros(G * K, dim, device=device)
        flat_counts = torch.zeros(G * K, device=device)

        for i in range(0, N, chunk_size):
            batch = X[:, i : i + chunk_size, :]
            bs = batch.shape[1]

            with torch.no_grad():
                dists = torch.cdist(batch, centroids)
                labels = dists.argmin(dim=2)

            flat_labels = (labels + offset).reshape(-1)
            flat_batch = batch.reshape(G * bs, dim)

            flat_sums.scatter_add_(
                0, flat_labels.unsqueeze(1).expand(-1, dim), flat_batch
            )
            flat_counts.scatter_add_(0, flat_labels, torch.ones(G * bs, device=device))

        counts = flat_counts.reshape(G, K)
        counts_safe = counts.unsqueeze(-1).clamp(min=1)
        new_centroids = flat_sums.reshape(G, K, dim) / counts_safe

        empty = counts == 0
        if empty.any():
            eg = empty.nonzero(as_tuple=True)
            n_empty = eg[0].shape[0]
            random_n = torch.randint(N, (n_empty,), device=device)
            new_centroids[eg[0], eg[1]] = X[eg[0], random_n]

        shifts = (new_centroids - centroids).reshape(G, -1).norm(dim=1)
        max_shift = shifts.max().item()
        centroids = new_centroids

        if verbose and (it + 1) % 10 == 0:
            print(
                f"    batched_kmeans iter {it + 1}/{max_iter}: "
                f"max_shift={max_shift:.6f}"
            )

        if max_shift < 1e-4:
            if verbose:
                print(f"    batched_kmeans converged at iter {it + 1}")
            break

    return centroids


def batched_assign(sub_vectors_3d, centroids_3d, device="cuda", chunk_size=None):
    """
    GPU batch nearest-neighbor assignment + reconstruction.

    Args:
        sub_vectors_3d: [G, N, dim] numpy or GPU tensor
        centroids_3d:   [G, K, dim] numpy or GPU tensor
        device:         GPU device
        chunk_size:     chunk along N dimension (None -> auto)

    Returns:
        recon:  [G, N, dim] GPU tensor float32
        labels: [G, N] GPU tensor int64
    """
    X = _as_device_tensor(sub_vectors_3d, device)
    C = _as_device_tensor(centroids_3d, device)
    G, N, dim = X.shape
    K = C.shape[1]

    if chunk_size is None:
        max_mem = 1 * 1024**3
        chunk_size = max(1, min(N, max_mem // (G * K * 4)))

    all_recon = []
    all_labels = []

    for i in range(0, N, chunk_size):
        batch = X[:, i : i + chunk_size, :]

        with torch.no_grad():
            dists = torch.cdist(batch, C)
            labels = dists.argmin(dim=2)
            labels_exp = labels.unsqueeze(-1).expand(-1, -1, dim)
            recon = torch.gather(C, 1, labels_exp)

        all_recon.append(recon)
        all_labels.append(labels)

    recon = torch.cat(all_recon, dim=1)
    labels = torch.cat(all_labels, dim=1)
    return recon, labels


# ================================================================
#                    PCA / ORFC Rotation (GPU)
# ================================================================


def learn_pca_rotation(vectors, num_groups, embedding_dim, device="cuda", verbose=True):
    """
    PCA rotation + interleave assignment (GPU).

    Args:
        vectors:       [N, D] numpy or GPU tensor
        num_groups:    number of groups
        embedding_dim: per-group dimension
        device:        GPU device
        verbose:       print info

    Returns:
        R:               [D, D] numpy float32, orthogonal rotation matrix
        eigenvalues:     [D] numpy float32, PCA eigenvalues (descending)
        group_variances: [num_groups] numpy float32, per-group total variance
    """
    X = _as_device_tensor(vectors, device)
    N, D = X.shape
    assert D == num_groups * embedding_dim, (
        f"D={D} != num_groups({num_groups}) * embedding_dim({embedding_dim})"
    )

    if verbose:
        print(f"    PCA rotation: N={N:,}, D={D}")

    with torch.no_grad():
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean
        cov = (X_centered.T @ X_centered) / N

        eigenvalues_t, eigenvectors_t = torch.linalg.eigh(cov)

        idx = torch.argsort(eigenvalues_t, descending=True)
        eigenvalues_t = eigenvalues_t[idx]
        eigenvectors_t = eigenvectors_t[:, idx]

        perm = torch.zeros(D, dtype=torch.long, device=device)
        for g in range(num_groups):
            for k in range(embedding_dim):
                perm[g * embedding_dim + k] = g + k * num_groups

        R = eigenvectors_t[:, perm].contiguous()

    eigenvalues_np = eigenvalues_t.cpu().numpy().astype(np.float32)
    R_np = R.cpu().numpy().astype(np.float32)

    group_variances = np.zeros(num_groups, dtype=np.float32)
    for g in range(num_groups):
        group_pca_indices = [g + k * num_groups for k in range(embedding_dim)]
        group_variances[g] = eigenvalues_np[group_pca_indices].sum()

    if verbose:
        print(
            f"    eigenvalue range: [{eigenvalues_np[-1]:.6f}, {eigenvalues_np[0]:.4f}]"
        )
        print(
            f"    group variance range: [{group_variances.min():.4f}, {group_variances.max():.4f}], "
            f"ratio={group_variances.max() / (group_variances.min() + 1e-10):.2f}"
        )
        orth_err = torch.max(torch.abs(R.T @ R - torch.eye(D, device=device))).item()
        print(f"    orthogonality error: {orth_err:.2e}")

    return R_np, eigenvalues_np, group_variances


def _pq_train_and_recon(Z, num_groups, embedding_dim, K, max_iter, device):
    """PQ train + reconstruct (internal, all GPU)."""
    N, D = Z.shape

    sub_3d = Z.reshape(N, num_groups, embedding_dim).permute(1, 0, 2).contiguous()

    centroids = batched_kmeans(
        sub_3d, K, max_iter=max_iter, device=device, verbose=False
    )

    recon_3d, _ = batched_assign(sub_3d, centroids, device=device)

    Z_hat = recon_3d.permute(1, 0, 2).reshape(N, D)

    total_mse = float(((Z - Z_hat) ** 2).mean().item())
    return Z_hat, centroids, total_mse


def learn_orfc_rotation(
    vectors,
    num_groups,
    embedding_dim,
    K,
    max_iter_orfc=20,
    max_iter_kmeans=50,
    device="cuda",
    verbose=True,
):
    """
    ORFC alternating optimization (fully GPU-accelerated).

    Alternates:
    1. Fix R, train PQ codebooks (batched_kmeans)
    2. Fix codebooks, optimize R via Procrustes (GPU SVD)

    Args:
        vectors:          [N, D] numpy, normalized full-dim vectors
        num_groups:       number of groups
        embedding_dim:    per-group dimension
        K:                codebook size
        max_iter_orfc:     outer ORFC iterations
        max_iter_kmeans:  k-means iterations per round
        device:           device
        verbose:          print progress

    Returns:
        R:          [D, D] numpy float32, optimal rotation matrix
        codebooks:  list of [K, dim] numpy, optimal per-group codebooks
        history:    list of (mse, delta), optimization history
    """
    X = _as_device_tensor(vectors, device)
    N, D = X.shape
    assert D == num_groups * embedding_dim

    if verbose:
        print("  ORFC: PCA initialization...")
    R_np, _, _ = learn_pca_rotation(
        X, num_groups, embedding_dim, device=device, verbose=verbose
    )
    R = torch.from_numpy(R_np).float().to(device)

    identity = torch.eye(D, device=device)
    prev_mse = float("inf")
    history = []
    best_R = R.clone()
    best_centroids = None
    best_mse = float("inf")

    for it in range(max_iter_orfc):
        if verbose:
            print(f"\n  ORFC iter {it + 1}/{max_iter_orfc}:")

        with torch.no_grad():
            Z = X @ R

        Z_hat, centroids, mse = _pq_train_and_recon(
            Z, num_groups, embedding_dim, K, max_iter_kmeans, device
        )

        delta = prev_mse - mse
        history.append((float(mse), float(delta)))

        if verbose:
            print(f"    PQ MSE = {mse:.8f}, delta = {delta:.2e}")

        if mse < best_mse:
            best_mse = mse
            best_R = R.clone()
            best_centroids = centroids.clone()

        if abs(delta) < 1e-8 and it > 0:
            if verbose:
                print(f"    Converged at iter {it + 1}")
            break
        prev_mse = mse

        with torch.no_grad():
            A = X.T @ Z_hat
            U, _, Vh = torch.linalg.svd(A)
            R_new = U @ Vh

            if torch.det(R_new) < 0:
                U[:, -1] *= -1
                R_new = U @ Vh

            R = R_new

        if verbose:
            orth_err = torch.max(torch.abs(R @ R.T - identity)).item()
            det_val = torch.det(R).item()
            print(f"    R orthogonality error: {orth_err:.2e}, det(R)={det_val:.6f}")

    if verbose:
        print(f"\n  ORFC done: best MSE = {best_mse:.8f} ({len(history)} iters)")

    R_np = best_R.cpu().numpy().astype(np.float32)
    centroids_np = best_centroids.cpu().numpy().astype(np.float32)
    codebooks = [centroids_np[g] for g in range(num_groups)]

    return R_np, codebooks, history

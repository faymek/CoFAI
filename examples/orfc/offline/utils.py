"""
Utility functions for ORFC evaluation pipeline.
"""

import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def set_seed(seed):
    """Fix random seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_single_feature(f):
    """Load a single feature file (for parallel loading)."""
    return np.load(f).astype(np.float32), f.stem


def preload_features(feat_files, num_workers=8, verbose=True):
    """
    Preload features with optional multi-threaded parallel loading.

    Args:
        feat_files: list of Path objects to .npy files
        num_workers: number of threads (1 = sequential)
        verbose: show progress bar

    Returns:
        features: list of numpy arrays [T, D]
        basenames: list of filename stems
    """
    if num_workers <= 1:
        features = []
        basenames = []
        for f in tqdm(feat_files, desc="Loading", disable=not verbose):
            features.append(np.load(f).astype(np.float32))
            basenames.append(f.stem)
        return features, basenames

    if verbose:
        print(f"  Parallel loading features (workers={num_workers})...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_load_single_feature, feat_files),
            total=len(feat_files),
            desc="Loading",
            disable=not verbose
        ))

    features = [r[0] for r in results]
    basenames = [r[1] for r in results]
    return features, basenames


def load_gt(gt_path):
    """
    Load ground truth labels.

    Expected format: each line is "<basename> <class_index>"

    Returns:
        dict mapping basename -> int label
    """
    gt_dict = {}
    with open(gt_path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                parts = ln.split()
                if len(parts) >= 2:
                    gt_dict[parts[0]] = int(parts[1])
    return gt_dict


def evaluate_accuracy(xhat_list, basenames, gt_dict, wrapper, layer_idx, device):
    """
    Evaluate top-1 classification accuracy.

    Args:
        xhat_list: list of reconstructed features [T, D] numpy arrays
        basenames: list of image basenames
        gt_dict: dict mapping basename -> label
        wrapper: Dinov2Wrapper or ClipWrapper
        layer_idx: block index where features were extracted
        device: torch device

    Returns:
        accuracy (float)
    """
    correct = 0
    total = 0
    for x_hat, basename in zip(xhat_list, basenames):
        if basename in gt_dict:
            label = gt_dict[basename]
            feat_tensor = torch.from_numpy(x_hat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = wrapper.forward_from_tokens(feat_tensor, layer_idx)
                pred = torch.argmax(logits, dim=1).item()
            if pred == label:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

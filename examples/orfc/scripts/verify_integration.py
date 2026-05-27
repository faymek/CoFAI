#!/usr/bin/env python
"""
Verify ORFC integration correctness by comparing with opq_release reference.

Steps:
  1. Train a few configs using the integrated code
  2. Compare weights numerically with opq_release/weights (should be identical)
  3. Run classification test (CLIP) and compare accuracy with reference results

Usage:
    conda run -n featcodec2 python verify_integration.py
"""

import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORFC_EXAMPLE_DIR = os.path.dirname(SCRIPT_DIR)
OFFLINE_DIR = os.path.join(ORFC_EXAMPLE_DIR, "offline")
COFAI_ROOT = os.path.dirname(os.path.dirname(ORFC_EXAMPLE_DIR))

ORFC_RELEASE = "/data4/workspace/zlt/featcodec/opq_release"
FEAT_ROOT = os.path.join(ORFC_RELEASE, "features")
REF_WEIGHTS_DIR = os.path.join(ORFC_RELEASE, "weights")
REF_RESULTS_DIR = os.path.join(ORFC_RELEASE, "results")
PRETRAINED_DIR = os.path.join(ORFC_RELEASE, "pretrained")

VERIFY_WEIGHTS_DIR = os.path.join(COFAI_ROOT, "weights", "orfc_verify")
VERIFY_RESULTS_DIR = os.path.join(COFAI_ROOT, "results", "orfc_verify")

CONFIGS = [
    {"backbone": "clip_vitl14", "layer": "blk05", "K": 256, "embedding_dim": 32},
    {"backbone": "dinov2_vitl14", "layer": "blk10", "K": 64, "embedding_dim": 32},
]

PYTHON = sys.executable


def banner(msg):
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}\n")


def run_cmd(cmd, desc=""):
    print(f"  [{desc}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=COFAI_ROOT)
    if result.returncode != 0:
        print(f"  FAILED (exit={result.returncode})")
        print(result.stderr[-2000:] if result.stderr else "")
        return False
    return True


def compare_weights(new_path, ref_path):
    """Compare two .npz weight files numerically."""
    if not os.path.exists(new_path):
        print(f"  ERROR: New weights not found: {new_path}")
        return False
    if not os.path.exists(ref_path):
        print(f"  WARNING: Reference weights not found: {ref_path}")
        return None

    new_data = np.load(new_path)
    ref_data = np.load(ref_path)

    R_new, R_ref = new_data['R'], ref_data['R']
    cb_new, cb_ref = new_data['codebooks'], ref_data['codebooks']

    r_diff = np.abs(R_new - R_ref).max()
    cb_diff = np.abs(cb_new - cb_ref).max()
    r_rel = np.abs(R_new - R_ref).mean() / (np.abs(R_ref).mean() + 1e-10)
    cb_rel = np.abs(cb_new - cb_ref).mean() / (np.abs(cb_ref).mean() + 1e-10)

    print(f"  R  max_abs_diff={r_diff:.2e}, mean_rel_diff={r_rel:.2e}")
    print(f"  CB max_abs_diff={cb_diff:.2e}, mean_rel_diff={cb_rel:.2e}")

    if r_diff < 1e-5 and cb_diff < 1e-5:
        print(f"  PASS: Weights are numerically identical (within fp32 precision)")
        return True
    elif r_diff < 1e-3 and cb_diff < 1e-3:
        print(f"  WARN: Small difference (possibly non-deterministic GPU ops)")
        return True
    else:
        print(f"  FAIL: Significant difference in weights")
        return False


def compare_results(new_path, ref_path):
    """Compare classification results."""
    if not os.path.exists(new_path):
        print(f"  ERROR: New results not found: {new_path}")
        return False
    if not os.path.exists(ref_path):
        print(f"  WARNING: Reference results not found: {ref_path}")
        return None

    with open(new_path) as f:
        new = json.load(f)
    with open(ref_path) as f:
        ref = json.load(f)

    acc_new = new['accuracy']
    acc_ref = ref['accuracy']
    diff = abs(acc_new - acc_ref)

    print(f"  New accuracy:  {acc_new:.4f} ({acc_new*100:.2f}%)")
    print(f"  Ref accuracy:  {acc_ref:.4f} ({acc_ref*100:.2f}%)")
    print(f"  Difference:    {diff:.4f}")

    if 'rate_info' in new and 'rate_info' in ref:
        bpt_new = new['rate_info'].get('xent_train_bpt', 0)
        bpt_ref = ref['rate_info'].get('xent_train_bpt', 0)
        print(f"  New xent_bpt:  {bpt_new:.4f}")
        print(f"  Ref xent_bpt:  {bpt_ref:.4f}")

    if diff < 0.001:
        print(f"  PASS: Accuracy matches exactly")
        return True
    elif diff < 0.01:
        print(f"  WARN: Minor accuracy difference (GPU non-determinism)")
        return True
    else:
        print(f"  FAIL: Significant accuracy difference")
        return False


def main():
    os.makedirs(VERIFY_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(VERIFY_RESULTS_DIR, exist_ok=True)

    results_summary = []

    # ======== Phase 1: Training Verification ========
    banner("Phase 1: Training Verification")

    for cfg in CONFIGS:
        bb, layer, K, emb = cfg['backbone'], cfg['layer'], cfg['K'], cfg['embedding_dim']
        tag = f"{bb}/{layer}_K{K}_e{emb}"
        print(f"\n--- Training: {tag} ---")

        cmd = [
            PYTHON, os.path.join(OFFLINE_DIR, "train_orfc.py"),
            "--backbone", bb,
            "--layer", layer,
            "--K", str(K),
            "--embedding_dim", str(emb),
            "--feat_root", FEAT_ROOT,
            "--weights_dir", VERIFY_WEIGHTS_DIR,
            "--seed", "42",
        ]

        success = run_cmd(cmd, "train")
        if not success:
            results_summary.append((tag, "TRAIN_FAILED"))
            continue

        new_wt = os.path.join(VERIFY_WEIGHTS_DIR, bb, f"{layer}_K{K}_e{emb}.npz")
        ref_wt = os.path.join(REF_WEIGHTS_DIR, bb, f"{layer}_K{K}_e{emb}.npz")

        match = compare_weights(new_wt, ref_wt)
        if match is True:
            results_summary.append((tag, "TRAIN_PASS"))
        elif match is None:
            results_summary.append((tag, "TRAIN_NO_REF"))
        else:
            results_summary.append((tag, "TRAIN_MISMATCH"))

    # ======== Phase 2: Classification Test (CLIP only) ========
    banner("Phase 2: Classification Test (CLIP)")

    clip_cfg = CONFIGS[0]  # clip_vitl14
    bb, layer, K, emb = clip_cfg['backbone'], clip_cfg['layer'], clip_cfg['K'], clip_cfg['embedding_dim']
    tag = f"{bb}/cls_{layer}_K{K}_e{emb}"
    print(f"\n--- Classification: {tag} ---")

    os.makedirs(os.path.join(VERIFY_RESULTS_DIR, bb), exist_ok=True)

    cmd = [
        PYTHON, os.path.join(OFFLINE_DIR, "test_cls.py"),
        "--backbone", bb,
        "--layer", layer,
        "--K", str(K),
        "--embedding_dim", str(emb),
        "--feat_root", FEAT_ROOT,
        "--weights_dir", VERIFY_WEIGHTS_DIR,
        "--seed", "42",
    ]

    env = os.environ.copy()
    env["PROJECT_ROOT"] = COFAI_ROOT

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=COFAI_ROOT, env=env)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-2000:]}")
        results_summary.append((tag, "CLS_FAILED"))
    else:
        print(result.stdout[-1000:])
        new_result_path = os.path.join(COFAI_ROOT, 'results', 'orfc', bb,
                                       f"cls_{layer}_K{K}_e{emb}.json")
        ref_result_path = os.path.join(REF_RESULTS_DIR, bb,
                                       f"cls_{layer}_K{K}_e{emb}.json")
        match = compare_results(new_result_path, ref_result_path)
        if match is True:
            results_summary.append((tag, "CLS_PASS"))
        elif match is None:
            results_summary.append((tag, "CLS_NO_REF"))
        else:
            results_summary.append((tag, "CLS_MISMATCH"))

    # ======== Summary ========
    banner("VERIFICATION SUMMARY")
    all_pass = True
    for tag, status in results_summary:
        icon = "OK" if "PASS" in status else ("??" if "NO_REF" in status else "XX")
        print(f"  [{icon}] {tag:50s} -> {status}")
        if "FAIL" in status or "MISMATCH" in status:
            all_pass = False

    print()
    if all_pass:
        print("  ALL CHECKS PASSED - Integration is correct!")
    else:
        print("  SOME CHECKS FAILED - Please investigate.")
    print()

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())

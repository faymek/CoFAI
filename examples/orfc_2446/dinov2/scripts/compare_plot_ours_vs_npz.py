#!/usr/bin/env python3
"""Build Ours reference table from plot_rate_task.py + CSV configs, compare with NPZ online eval."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ORFC_RESULTS = PROJECT_ROOT.parent / "ORFC" / "coding" / "orfc" / "results"
OUT_CSV = PROJECT_ROOT / "eval_results" / "ours_plot_vs_npz_comparison.csv"
OUT_TXT = PROJECT_ROOT / "eval_results" / "ours_plot_vs_npz_summary.txt"

W_L = "weights/orfc_2446/dinov2_vitl14_ori"
W_G = "weights/orfc_2446/dinov2_vitg14_ori"

SLOT_MAP = {
    "blk05": "slot06", "blk10": "slot11", "blk15": "slot16", "blk20": "slot21",
    "blk09": "slot10", "blk19": "slot20", "blk29": "slot30",
}

# Ours points from plot_rate_task.py lines 520-711
# Each entry: (backbone, task, block_num, layer, config_label, orfc_bpfp, orfc_metric)
# config_label must match run_online_eval_all / CSV semantics
PLOT_OURS: list[tuple] = [
    # ---- vitl14 cls ----
    ("vitl14", "cls", 5, "blk05", "K=4, e32, lambda=0.0, lr=5e-4, ep=300", 0.0569, 93.80),
    ("vitl14", "cls", 5, "blk05", "K=8, e32", 0.087, 95.00),
    ("vitl14", "cls", 5, "blk05", "K=16, e32", 0.1179, 96.00),
    ("vitl14", "cls", 5, "blk05", "K=64, e32, lr=5e-4", 0.1796, 96.80),
    ("vitl14", "cls", 5, "blk05", "K=256, e32", 0.2361, 97.00),
    ("vitl14", "cls", 5, "blk05", "K=64, e16", 0.3534, 97.40),
    ("vitl14", "cls", 5, "blk05", "K=256, e16", 0.4607, 97.80),
    ("vitl14", "cls", 10, "blk10", "K=4, e32", 0.0602, 96.00),
    ("vitl14", "cls", 10, "blk10", "K=8, e32", 0.0913, 97.00),
    ("vitl14", "cls", 10, "blk10", "K=16, e32", 0.1228, 97.80),
    ("vitl14", "cls", 10, "blk10", "K=64, e32, lambda=0.0", 0.1849, 98.00),
    ("vitl14", "cls", 10, "blk10", "K=256, e32, lambda=0.0, lr=5e-4", 0.246, 98.00),
    ("vitl14", "cls", 15, "blk15", "K=4, e32", 0.061, 93.80),
    ("vitl14", "cls", 15, "blk15", "K=8, e32", 0.0922, 95.00),
    ("vitl14", "cls", 15, "blk15", "K=16, e32", 0.123, 96.00),
    ("vitl14", "cls", 15, "blk15", "K=64, e32, lr=5e-4", 0.1849, 96.80),
    ("vitl14", "cls", 15, "blk15", "K=256, e32", 0.2471, 97.00),
    ("vitl14", "cls", 15, "blk15", "K=64, e16", 0.3707, 97.40),
    ("vitl14", "cls", 15, "blk15", "K=256, e16, lambda=0.2", 0.4476, 97.60),
    ("vitl14", "cls", 15, "blk15", "K=256, e16", 0.488, 97.80),
    ("vitl14", "cls", 20, "blk20", "K=8, e32", 0.0928, 87.00),
    ("vitl14", "cls", 20, "blk20", "K=16, e32, lr=5e-4", 0.1239, 92.40),
    ("vitl14", "cls", 20, "blk20", "K=32, e32", 0.1538, 93.00),
    ("vitl14", "cls", 20, "blk20", "K=64, e32, lambda=0.0", 0.1842, 94.00),
    ("vitl14", "cls", 20, "blk20", "K=256, e32, lambda=0.0, lr=5e-4", 0.2469, 95.60),
    ("vitl14", "cls", 20, "blk20", "K=256, e16", 0.4883, 97.40),
    # ---- vitl14 seg ----
    ("vitl14", "seg", 5, "blk05", "K=4, e32, lambda=0.0, lr=5e-4, ep=300", 0.0565, 78.30),
    ("vitl14", "seg", 5, "blk05", "K=8, e32", 0.0864, 78.90),
    ("vitl14", "seg", 5, "blk05", "K=16, e32", 0.1172, 79.40),
    ("vitl14", "seg", 5, "blk05", "K=64, e32, lr=5e-4", 0.1788, 79.70),
    ("vitl14", "seg", 5, "blk05", "K=256, e32", 0.2339, 80.00),
    ("vitl14", "seg", 5, "blk05", "K=64, e16", 0.3497, 80.40),
    ("vitl14", "seg", 10, "blk10", "K=4, e32", 0.0597, 78.40),
    ("vitl14", "seg", 10, "blk10", "K=8, e32", 0.091, 80.10),
    ("vitl14", "seg", 10, "blk10", "K=64, e32, lambda=0.0", 0.1803, 80.40),
    ("vitl14", "seg", 10, "blk10", "K=256, e32, lambda=0.0, lr=5e-4", 0.2389, 80.90),
    ("vitl14", "seg", 10, "blk10", "K=64, e16", 0.3678, 81.20),
    ("vitl14", "seg", 15, "blk15", "K=4, e32", 0.0607, 78.30),
    ("vitl14", "seg", 15, "blk15", "K=8, e32", 0.092, 78.90),
    ("vitl14", "seg", 15, "blk15", "K=16, e32", 0.1227, 79.40),
    ("vitl14", "seg", 15, "blk15", "K=64, e32, lr=5e-4", 0.1849, 79.70),
    ("vitl14", "seg", 15, "blk15", "K=256, e32", 0.2472, 80.00),
    ("vitl14", "seg", 15, "blk15", "K=64, e16", 0.3708, 80.40),
    ("vitl14", "seg", 20, "blk20", "K=8, e32", 0.0942, 68.60),
    ("vitl14", "seg", 20, "blk20", "K=16, e32, lr=5e-4", 0.1244, 71.00),
    ("vitl14", "seg", 20, "blk20", "K=64, e32, lambda=0.0", 0.177, 72.80),
    ("vitl14", "seg", 20, "blk20", "K=256, e16", 0.485, 75.50),
    # ---- vitg14 cls ----
    ("vitg14", "cls", 9, "blk09", "K=4, e32, lambda=0.5", 0.0603, 97.40),
    ("vitg14", "cls", 9, "blk09", "K=8, e32, lambda=0.5", 0.0902, 97.80),
    ("vitg14", "cls", 9, "blk09", "K=16, e32, lambda=0.5", 0.1207, 99.00),
    ("vitg14", "cls", 9, "blk09", "K=64, e32, lambda=0.5", 0.1808, 99.20),
    ("vitg14", "cls", 9, "blk09", "K=256, e32, lambda=0.5", 0.239, 99.20),
    ("vitg14", "cls", 19, "blk19", "K=4, e32, lambda=0.5", 0.0605, 97.60),
    ("vitg14", "cls", 19, "blk19", "K=8, e32, lambda=0.5", 0.092, 98.80),
    ("vitg14", "cls", 19, "blk19", "K=16, e32, lambda=0.5", 0.1231, 99.40),
    ("vitg14", "cls", 19, "blk19", "K=64, e32, lambda=0.5", 0.1849, 99.40),
    ("vitg14", "cls", 19, "blk19", "K=256, e32, lambda=0.5", 0.2442, 99.40),
    ("vitg14", "cls", 29, "blk29", "K=4, e32, lambda=0.5", 0.0616, 95.60),
    ("vitg14", "cls", 29, "blk29", "K=8, e32, lambda=0.5", 0.0929, 96.20),
    ("vitg14", "cls", 29, "blk29", "K=16, e32, lambda=0.5", 0.1241, 97.80),
    ("vitg14", "cls", 29, "blk29", "K=64, e32, lambda=0.0", 0.1853, 98.00),
    ("vitg14", "cls", 29, "blk29", "K=256, e32, lambda=0.0, lr=5e-4", 0.2462, 98.80),
    # ---- vitg14 seg ----
    ("vitg14", "seg", 9, "blk09", "K=4, e32, lambda=0.5", 0.0599, 82.22),
    ("vitg14", "seg", 9, "blk09", "K=8, e32, lambda=0.5", 0.09, 82.92),
    ("vitg14", "seg", 9, "blk09", "K=16, e32, lambda=0.5", 0.1206, 83.06),
    ("vitg14", "seg", 9, "blk09", "K=64, e32, lambda=0.0", 0.1798, 83.25),
    ("vitg14", "seg", 19, "blk19", "K=4, e32, lambda=0.5", 0.0602, 83.13),
    ("vitg14", "seg", 19, "blk19", "K=8, e32, lambda=0.5", 0.0918, 83.43),
    ("vitg14", "seg", 19, "blk19", "K=256, e32, lambda=0.0, lr=5e-4", 0.2448, 83.48),
    ("vitg14", "seg", 29, "blk29", "K=4, e32, lambda=0.5", 0.0616, 79.75),
    ("vitg14", "seg", 29, "blk29", "K=8, e32, lambda=0.5", 0.0928, 80.80),
    ("vitg14", "seg", 29, "blk29", "K=16, e32, lambda=0.5", 0.1242, 81.38),
    ("vitg14", "seg", 29, "blk29", "K=64, e32, lambda=0.0", 0.184, 81.93),
    ("vitg14", "seg", 29, "blk29", "K=256, e32, lambda=0.0, lr=5e-4", 0.2442, 82.50),
]

# Map config_label -> checkpoint filename (from run_online_eval_all.sh)
CONFIG_TO_CKPT: dict[tuple[str, str, str], str] = {
    ("vitl14", "cls", "blk05", "K=4, e32, lambda=0.0, lr=5e-4, ep=300"): f"{W_L}/blk05_K4_emb32_bt1024_ws_lmbda0.0_tau0.5_lr0.0005_ep300_n5000_s42.npz",
    ("vitl14", "cls", "blk05", "K=8, e32"): f"{W_L}/blk05_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk05", "K=16, e32"): f"{W_L}/blk05_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk05", "K=64, e32, lr=5e-4"): f"{W_L}/blk05_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk05", "K=256, e32"): f"{W_L}/blk05_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk05", "K=64, e16"): f"{W_L}/blk05_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk05", "K=256, e16"): f"{W_L}/blk05_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk10", "K=4, e32"): f"{W_L}/blk10_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk10", "K=8, e32"): f"{W_L}/blk10_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk10", "K=16, e32"): f"{W_L}/blk10_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk10", "K=64, e32, lambda=0.0"): f"{W_L}/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk10", "K=256, e32, lambda=0.0, lr=5e-4"): f"{W_L}/blk10_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk15", "K=4, e32"): f"{W_L}/blk15_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk15", "K=8, e32"): f"{W_L}/blk15_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk15", "K=16, e32"): f"{W_L}/blk15_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk15", "K=64, e32, lr=5e-4"): f"{W_L}/blk15_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk15", "K=256, e32"): f"{W_L}/blk15_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk15", "K=64, e16"): f"{W_L}/blk15_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk15", "K=256, e16, lambda=0.2"): f"{W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.2_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk15", "K=256, e16"): f"{W_L}/blk15_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk20", "K=8, e32"): f"{W_L}/blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk20", "K=16, e32, lr=5e-4"): f"{W_L}/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk20", "K=32, e32"): f"{W_L}/blk20_K32_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk20", "K=64, e32, lambda=0.0"): f"{W_L}/blk20_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk20", "K=256, e32, lambda=0.0, lr=5e-4"): f"{W_L}/blk20_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "cls", "blk20", "K=256, e16"): f"{W_L}/blk20_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    # seg vitl14
    ("vitl14", "seg", "blk05", "K=4, e32, lambda=0.0, lr=5e-4, ep=300"): f"{W_L}/blk05_K4_emb32_bt1024_ws_lmbda0.0_tau0.5_lr0.0005_ep300_n5000_s42.npz",
    ("vitl14", "seg", "blk05", "K=8, e32"): f"{W_L}/blk05_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk05", "K=16, e32"): f"{W_L}/blk05_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk05", "K=64, e32, lr=5e-4"): f"{W_L}/blk05_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk05", "K=256, e32"): f"{W_L}/blk05_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk05", "K=64, e16"): f"{W_L}/blk05_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk10", "K=4, e32"): f"{W_L}/blk10_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk10", "K=8, e32"): f"{W_L}/blk10_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk10", "K=64, e32, lambda=0.0"): f"{W_L}/blk10_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk10", "K=256, e32, lambda=0.0, lr=5e-4"): f"{W_L}/blk10_K256_emb32_bt1024_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk10", "K=64, e16"): f"{W_L}/blk10_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk15", "K=4, e32"): f"{W_L}/blk15_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk15", "K=8, e32"): f"{W_L}/blk15_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk15", "K=16, e32"): f"{W_L}/blk15_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk15", "K=64, e32, lr=5e-4"): f"{W_L}/blk15_K64_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk15", "K=256, e32"): f"{W_L}/blk15_K256_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk15", "K=64, e16"): f"{W_L}/blk15_K64_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk20", "K=8, e32"): f"{W_L}/blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk20", "K=16, e32, lr=5e-4"): f"{W_L}/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk20", "K=64, e32, lambda=0.0"): f"{W_L}/blk20_K64_emb32_bt1024_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitl14", "seg", "blk20", "K=256, e16"): f"{W_L}/blk20_K256_emb16_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    # vitg14 cls
    ("vitg14", "cls", "blk09", "K=4, e32, lambda=0.5"): f"{W_G}/blk09_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk09", "K=8, e32, lambda=0.5"): f"{W_G}/blk09_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk09", "K=16, e32, lambda=0.5"): f"{W_G}/blk09_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk09", "K=64, e32, lambda=0.5"): f"{W_G}/blk09_K64_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk09", "K=256, e32, lambda=0.5"): f"{W_G}/blk09_K256_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk19", "K=4, e32, lambda=0.5"): f"{W_G}/blk19_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk19", "K=8, e32, lambda=0.5"): f"{W_G}/blk19_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk19", "K=16, e32, lambda=0.5"): f"{W_G}/blk19_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk19", "K=64, e32, lambda=0.5"): f"{W_G}/blk19_K64_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk19", "K=256, e32, lambda=0.5"): f"{W_G}/blk19_K256_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk29", "K=4, e32, lambda=0.5"): f"{W_G}/blk29_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk29", "K=8, e32, lambda=0.5"): f"{W_G}/blk29_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk29", "K=16, e32, lambda=0.5"): f"{W_G}/blk29_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk29", "K=64, e32, lambda=0.0"): f"{W_G}/blk29_K64_emb32_bt1536_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "cls", "blk29", "K=256, e32, lambda=0.0, lr=5e-4"): f"{W_G}/blk29_K256_emb32_bt1536_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    # vitg14 seg
    ("vitg14", "seg", "blk09", "K=4, e32, lambda=0.5"): f"{W_G}/blk09_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk09", "K=8, e32, lambda=0.5"): f"{W_G}/blk09_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk09", "K=16, e32, lambda=0.5"): f"{W_G}/blk09_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk09", "K=64, e32, lambda=0.0"): f"{W_G}/blk09_K64_emb32_bt1536_ws_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk19", "K=4, e32, lambda=0.5"): f"{W_G}/blk19_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk19", "K=8, e32, lambda=0.5"): f"{W_G}/blk19_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk19", "K=256, e32, lambda=0.0, lr=5e-4"): f"{W_G}/blk19_K256_emb32_bt1536_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk29", "K=4, e32, lambda=0.5"): f"{W_G}/blk29_K4_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk29", "K=8, e32, lambda=0.5"): f"{W_G}/blk29_K8_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk29", "K=16, e32, lambda=0.5"): f"{W_G}/blk29_K16_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk29", "K=64, e32, lambda=0.0"): f"{W_G}/blk29_K64_emb32_bt1536_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.npz",
    ("vitg14", "seg", "blk29", "K=256, e32, lambda=0.0, lr=5e-4"): f"{W_G}/blk29_K256_emb32_bt1536_ws_tau0.5_lr0.0005_ep100_n5000_s42.npz",
}


def parse_train_params(ckpt: str) -> dict:
    stem = Path(ckpt).stem
    m = re.search(
        r"^(blk\d+)_K(\d+)_emb(\d+)_bt(\d+)_ws(?:_lmbda([\d.]+))?_(?:tau([\d.]+)_)?lr([\d.]+)_ep(\d+)",
        stem,
    )
    if not m:
        return {}
    return {
        "layer": m.group(1),
        "K": int(m.group(2)),
        "emb": int(m.group(3)),
        "bt": int(m.group(4)),
        "lambda": float(m.group(5)) if m.group(5) else 0.0,
        "tau": float(m.group(6)) if m.group(6) else 0.5,
        "lr": float(m.group(7)),
        "epochs": int(m.group(8)),
    }


def codec_tag(ckpt: str) -> str:
    stem = Path(ckpt).stem
    m = re.search(r"_K(\d+)_emb(\d+)_", stem)
    tag = f"K{m.group(1)}e{m.group(2)}"
    for key in ("lmbda", "tau", "lr", "ep"):
        mm = re.search(rf"_{key}([0-9.]+)", stem)
        if mm:
            tag += f"_{key}{mm.group(1)}"
    return tag


def resolve_result_path(bb: str, layer: str, task: str, ckpt: str) -> Path | None:
    dataset = "imagenet-sel500" if task == "cls" else "voc2012-sel100"
    bb_path = f"dinov2-{bb}-slide" if task == "seg" else f"dinov2-{bb}"
    slot = SLOT_MAP[layer]
    p = PROJECT_ROOT / f"eval_results/SoftPQ/{dataset}/{bb_path}/{slot}/{codec_tag(ckpt)}/result.json"
    return p if p.exists() else None


def load_npz_metrics(bb: str, layer: str, task: str, ckpt: str) -> tuple[float | None, float | None]:
    p = resolve_result_path(bb, layer, task, ckpt)
    if not p:
        return None, None
    data = json.loads(p.read_text())["results"]
    bpfp = float(data["bpfp"])
    if task == "cls":
        metric = float(data["cls_top-1"])
    else:
        m = float(data["semseg_mIoU"])
        metric = m * 100 if m <= 1.0 else m
    return bpfp, metric


def main() -> None:
    rows = []
    match_bpfp = match_metric = mismatch = missing_npz = 0

    for bb, task, block, layer, config, orfc_bpfp, orfc_metric in PLOT_OURS:
        key = (bb, task, layer, config)
        ckpt = CONFIG_TO_CKPT.get(key)
        if not ckpt:
            raise KeyError(f"No checkpoint mapping for {key}")
        params = parse_train_params(ckpt)
        npz_bpfp, npz_metric = load_npz_metrics(bb, layer, task, ckpt)

        d_bpfp = d_metric = None
        status = "missing_npz"
        if npz_bpfp is not None:
            d_bpfp = npz_bpfp - orfc_bpfp
            d_metric = npz_metric - orfc_metric
            ok_bpfp = abs(d_bpfp) <= 0.01
            ok_metric = abs(d_metric) <= 0.2
            if ok_bpfp and ok_metric:
                status = "match"
                match_bpfp += 1
            else:
                status = "mismatch"
                mismatch += 1
        else:
            missing_npz += 1

        rows.append({
            "backbone": bb,
            "task": task,
            "block": block,
            "layer": layer,
            "config": config,
            "K": params.get("K", ""),
            "emb": params.get("emb", ""),
            "lambda": params.get("lambda", ""),
            "lr": params.get("lr", ""),
            "epochs": params.get("epochs", ""),
            "tau": params.get("tau", ""),
            "bt": params.get("bt", ""),
            "checkpoint": Path(ckpt).name,
            "orfc_bpfp": f"{orfc_bpfp:.4f}",
            "orfc_metric": f"{orfc_metric:.2f}",
            "npz_bpfp": f"{npz_bpfp:.4f}" if npz_bpfp is not None else "",
            "npz_metric": f"{npz_metric:.2f}" if npz_metric is not None else "",
            "delta_bpfp": f"{d_bpfp:.4f}" if d_bpfp is not None else "",
            "delta_metric": f"{d_metric:.2f}" if d_metric is not None else "",
            "status": status,
        })

    fieldnames = list(rows[0].keys())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Summary
    lines = [
        f"Ours (plot_rate_task.py) vs NPZ online eval — {len(rows)} configs",
        f"match (BPFP±0.01 & metric±0.2%): {match_bpfp}",
        f"mismatch: {mismatch}",
        f"missing_npz: {missing_npz}",
        "",
        "Mismatch breakdown:",
    ]
    bpfp_only = metric_only = both = 0
    for r in rows:
        if r["status"] != "mismatch":
            continue
        db = abs(float(r["delta_bpfp"])) if r["delta_bpfp"] else 0
        dm = abs(float(r["delta_metric"])) if r["delta_metric"] else 0
        if db > 0.01 and dm <= 0.2:
            bpfp_only += 1
        elif dm > 0.2 and db <= 0.01:
            metric_only += 1
        elif db > 0.01 and dm > 0.2:
            both += 1
    lines += [
        f"  BPFP only (>0.01): {bpfp_only}",
        f"  metric only (>0.2%): {metric_only}",
        f"  both: {both}",
        "",
        "Largest metric gaps:",
    ]
    mism = [r for r in rows if r["status"] == "mismatch"]
    mism.sort(key=lambda r: -abs(float(r["delta_metric"] or 0)))
    for r in mism[:10]:
        lines.append(
            f"  {r['backbone']} {r['layer']} {r['task']} {r['config']}: "
            f"Δmetric={r['delta_metric']}% Δbpfp={r['delta_bpfp']}"
        )
    OUT_TXT.write_text("\n".join(lines))
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TXT}")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()

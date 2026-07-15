#!/usr/bin/env python3
"""DINOv3-L RAEtail (+ optional CLS MSE) SoftPQ training for orfc_2446_dinov3.

Mirrors CoFAI-dev_zlt/examples/orfc_2446/train.py Mode B, adapted for DINOv3-L:
  - Dinov3TimmBackbone (patch16, D=1024, always has register tokens)
  - RoPE-aware frozen tail (Dinov3FrozenTail)
  - RAEv2 ViTXL decoder: weights/RAE/decoders/dinov3/large/decoder.pt
  - Loss: pixel L1 via RAE decoder + optional CLS MSE (affine LN)

``--train_tokens patch``: SoftPQ / OPQ only on patch tokens; CLS+reg are identity
bypass in norm space (same contract as train_soft_pq_dinov3.py). Requires a full
token sequence (auto-enables compress_reg-style loading). CLS MSE is a no-op then.

Default matches RAEv2 dinov3l-k1: slot=24 (encode all 24 blocks, empty tail = k=1),
img_size=256 → 16×16 patches.

Usage:
  poetry run python examples/orfc_2446/dinov3/offline/train_raetail_cls_dinov3.py \\
    --K 256 --use_rae_tail --cls_loss_weight 1.0 --compress_reg --device cuda:0
  # Patch-only RAEtail L1 (CLS/reg uncompressed):
  poetry run python examples/orfc_2446/dinov3/offline/train_raetail_cls_dinov3.py \\
    --K 256 --use_rae_tail --train_tokens patch --cls_loss_weight 0 \\
    --norm_mode split_cls_patch --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torchvision import transforms
from tqdm import tqdm

_COFAI_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLE_DIR = Path(__file__).resolve().parents[1]
for _p in (_COFAI_ROOT, _EXAMPLE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cofai.backbone.timm import Dinov3TimmBackbone
from cofai.entropy_models.orfc_model import (
    batch_inv_normalize_gpu,
    batch_normalize_gpu,
    learn_orfc_rotation,
)
from cofai.entropy_models.soft_pq import (
    FeatureCodec,
    FrozenTail,
    OrthogonalTransform,
    SoftPQ,
    codec_forward,
    save_codec,
)
from cofai.entropy_models.soft_pq_export import (
    compute_histogram_pmf,
    npz_path_for_codec,
    save_codec_npz,
)
from cofai.heads.rae import GeneralDecoder

from lib.dinov3_frozen_tail import build_dinov3_tail

# RAEv2 DINOv3 decoder emits near-[0,1] pixels; do not apply ImageNet denorm.
PIXEL_MEAN = (0.0, 0.0, 0.0)
PIXEL_STD = (1.0, 1.0, 1.0)

DECODER_CONFIGS = {
    "ViTXL": {
        "cfg": {
            "decoder_hidden_size": 1152,
            "decoder_intermediate_size": 4096,
            "decoder_num_attention_heads": 16,
            "decoder_num_hidden_layers": 28,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "num_channels": 3,
            "qkv_bias": True,
        },
        "default_path": "weights/RAE/decoders/dinov3/large/decoder.pt",
        "tag": "_decXL",
        "encoder_hidden_size": 1024,
    },
}


def build_transform(img_size):
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])


def load_image_paths(pathname_list, imagenet_root):
    paths = []
    with open(pathname_list, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_id, img_name = parts[0], parts[1]
                img_path = Path(imagenet_root) / class_id / f"{img_name}.JPEG"
                if img_path.exists():
                    paths.append(img_path)
    return paths


def _num_reg(dino) -> int:
    if hasattr(dino, "num_reg_tokens"):
        return int(dino.num_reg_tokens)
    return max(int(dino.num_prefix_tokens) - 1, 0)


def extract_features(backbone, img_files, img_size, device, batch_size=4, compress_reg=False):
    tfm = build_transform(img_size)
    num_prefix = backbone.model.num_prefix_tokens
    num_reg = _num_reg(backbone.model)
    cls_patches_list = []
    reg_list = []

    print(f"  Extracting features from {len(img_files)} images (img_size={img_size})...")
    print(f"  compress_reg={compress_reg}  prefix={num_prefix} reg={num_reg}")
    with torch.inference_mode():
        for i in tqdm(range(0, len(img_files), batch_size), desc="Extract"):
            batch_files = img_files[i:i + batch_size]
            imgs = [tfm(Image.open(fp).convert("RGB")) for fp in batch_files]
            x = torch.stack(imgs).to(device)
            h = backbone.encode(x)
            for j in range(h.shape[0]):
                if compress_reg:
                    cls_patches_list.append(h[j].cpu().numpy())
                    reg_list.append(np.zeros((0, h.shape[2]), dtype=np.float32))
                else:
                    cls_token = h[j, 0:1, :]
                    reg_tokens = h[j, 1:1 + num_reg, :]
                    patch_tokens = h[j, 1 + num_reg:, :]
                    cls_patches_list.append(
                        torch.cat([cls_token, patch_tokens], dim=0).cpu().numpy()
                    )
                    reg_list.append(reg_tokens.cpu().numpy())
            del x, h

    print(f"  Extracted {len(cls_patches_list)} samples, shape={cls_patches_list[0].shape}")
    return cls_patches_list, reg_list


def _load_single_npy(args_tuple):
    npy_path, num_reg, compress_reg = args_tuple
    feat = np.load(npy_path)
    if compress_reg:
        return feat, np.zeros((0, feat.shape[1]), dtype=feat.dtype)
    cls_token = feat[0:1, :]
    reg_tokens = feat[1:1 + num_reg, :]
    patch_tokens = feat[1 + num_reg:, :]
    return np.concatenate([cls_token, patch_tokens], axis=0), reg_tokens


def load_cached_features(feat_cache_dir, img_names, num_reg=4, compress_reg=False, num_workers=8):
    cache_dir = Path(feat_cache_dir)
    npy_paths = []
    for name in img_names:
        npy_path = cache_dir / f"{name}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Feature cache not found: {npy_path}")
        npy_paths.append(npy_path)

    work_args = [(p, num_reg, compress_reg) for p in npy_paths]
    cls_patches_list = [None] * len(npy_paths)
    reg_list = [None] * len(npy_paths)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, (cls_patch, reg) in enumerate(executor.map(_load_single_npy, work_args)):
            cls_patches_list[i] = cls_patch
            reg_list[i] = reg

    print(
        f"  Loaded {len(cls_patches_list)} cached features from {cache_dir} "
        f"({num_workers} workers), shape={cls_patches_list[0].shape}"
    )
    return cls_patches_list, reg_list


def compute_perplexity(usage):
    p = usage / usage.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    entropy = -(p * (p + 1e-10).log()).sum(dim=-1)
    return entropy.exp().mean().item()


def decoder_forward(decoder, features, token_res, use_checkpoint=False):
    x_ = decoder.decoder_embed(features)
    h_p, w_p = token_res
    if h_p * w_p != decoder.num_patches:
        pos_embed = decoder.interpolate_pos_encoding(token_res)
    else:
        pos_embed = decoder.decoder_pos_embed
    cls_tok = decoder.trainable_cls_token.expand(x_.shape[0], -1, -1)
    h = torch.cat([cls_tok, x_], dim=1) + pos_embed
    for layer in decoder.decoder_layers:
        if use_checkpoint:
            h = torch_checkpoint(layer, h, None, False, use_reentrant=False)
        else:
            h = layer(h, None, False)
        if isinstance(h, tuple):
            h = h[0]
    h = decoder.decoder_norm(h)
    return decoder.decoder_pred(h)[:, 1:, :]


def run_tail_blocks(h_full, tail):
    if hasattr(tail, "forward_blocks"):
        return tail.forward_blocks(h_full)
    for blk in getattr(tail, "blocks", []):
        h_full = blk(h_full)
    return h_full


def extract_rae_patches(h_after_blocks, num_prefix, norm_eps):
    patches = h_after_blocks[:, num_prefix:, :]
    mean = patches.mean(dim=-1, keepdim=True)
    var = patches.var(dim=-1, keepdim=True, unbiased=False)
    return (patches - mean) / torch.sqrt(var + norm_eps)


def _assemble_full(cp, reg, compress_reg, num_reg):
    if compress_reg:
        return cp
    cls_tokens = cp[:, 0:1, :]
    patch_tokens = cp[:, 1:, :]
    return torch.cat([cls_tokens, reg, patch_tokens], dim=1)


def _opq_vectors(Y: torch.Tensor, train_tokens: str, n_prefix: int, D: int) -> np.ndarray:
    """Flatten normalized tokens for OPQ; patch-only when train_tokens=='patch'."""
    if train_tokens == "patch" and n_prefix > 0:
        return Y[:, n_prefix:, :].reshape(-1, D).cpu().numpy()
    return Y.reshape(-1, D).cpu().numpy()


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.environ.setdefault("PROJECT_ROOT", str(_COFAI_ROOT))

    print(f"\n{'=' * 70}")
    print(f"  DINOv3-L RAEtail(+CLS) SoftPQ  img={args.img_size}  slot={args.slot}")
    print(f"  K={args.K} emb={args.embedding_dim} λ={args.lmbda} ep={args.epochs}")
    print(f"  use_rae_tail={args.use_rae_tail}  cls_w={args.cls_loss_weight}")
    print(f"  train_tokens={args.train_tokens}")
    print(f"  norm_mode={args.norm_mode}  n_prefix={args.n_prefix}")
    print(f"  device={device}")
    print(f"{'=' * 70}")

    print("\n[1/4] Loading DINOv3-L backbone...")
    ckpt = None if args.backbone_ckpt in ("none", "", None) else args.backbone_ckpt
    backbone = Dinov3TimmBackbone(
        model_size="large",
        img_size=args.img_size,
        patch_size=args.patch_size,
        dynamic_size=False,
        slot=args.slot,
        n_last_blocks=1,
        pretrained=False,
        ckpt_path=ckpt,
        device=str(device),
    )
    backbone.model.to(device).eval()
    dino = backbone.model
    num_blocks = len(dino.blocks)
    num_prefix = int(dino.num_prefix_tokens)
    num_reg = _num_reg(dino)
    slot = args.slot
    encode_end = num_blocks + slot if slot < 0 else slot
    num_tail = num_blocks - encode_end
    token_per_dim = args.img_size // args.patch_size
    num_patches = token_per_dim ** 2
    print(f"  blocks={num_blocks}  encode 0..{encode_end - 1}  tail={num_tail}")
    print(f"  prefix={num_prefix} (cls=1, reg={num_reg})  D={dino.embed_dim}")
    print(f"  token_res=({token_per_dim},{token_per_dim})  patches={num_patches}")

    # split_* modes need n_prefix = CLS + reg count (DINOv3-L: 5).
    # train_tokens=patch also needs n_prefix for SoftPQ prefix bypass.
    if args.n_prefix <= 0:
        if args.norm_mode.startswith("split_") or args.train_tokens == "patch":
            args.n_prefix = num_prefix
        else:
            args.n_prefix = 0
    if args.norm_mode == "split_reg_cls_patch" and args.n_prefix < 2:
        raise ValueError(
            f"split_reg_cls_patch requires n_prefix>=2 (got {args.n_prefix}); "
            "need separate reg group vs cls+patch."
        )
    if args.train_tokens == "patch":
        if args.n_prefix < 1:
            raise ValueError(
                "train_tokens=patch requires n_prefix>=1 (CLS+reg bypass length)."
            )
        # Full sequence required so CLS/reg stay in place while SoftPQ sees patches only.
        if not args.compress_reg:
            print("  [train_tokens=patch] enabling full-sequence feature load (compress_reg)")
            args.compress_reg = True
        if args.cls_loss_weight > 0:
            print(
                "  [warn] train_tokens=patch bypasses CLS → CLS MSE≈0; "
                "consider --cls_loss_weight 0"
            )
        print(
            f"  Patch-only SoftPQ: n_prefix={args.n_prefix} "
            f"(CLS+reg identity bypass in norm space)"
        )

    print("[2/4] Loading images / features...")
    if args.pathname_list and args.pathname_list != "none":
        img_files = load_image_paths(args.pathname_list, args.imagenet_root)
        print(f"  Loaded {len(img_files)} images from pathname list")
    else:
        img_root = Path(args.imagenet_root)
        img_files = (
            sorted(img_root.rglob("*.JPEG"))
            + sorted(img_root.rglob("*.jpg"))
            + sorted(img_root.rglob("*.png"))
        )
        print(f"  Found {len(img_files)} images from root")

    if args.max_images and args.max_images < len(img_files):
        img_files = img_files[: args.max_images]
        print(f"  Truncated to {len(img_files)} images")

    if args.feat_cache_dir and args.feat_cache_dir != "none":
        print(f"  Loading cached features from: {args.feat_cache_dir}")
        img_names = [Path(f).stem for f in img_files]
        cls_patches_list, reg_list = load_cached_features(
            args.feat_cache_dir, img_names, num_reg=num_reg, compress_reg=args.compress_reg,
        )
        n_patch_cache = cls_patches_list[0].shape[0] - (1 if not args.compress_reg else num_prefix)
        if args.compress_reg:
            n_patch_cache = cls_patches_list[0].shape[0] - num_prefix
        if n_patch_cache != num_patches:
            raise ValueError(
                f"Feature cache patches={n_patch_cache} != img_size/patch "
                f"({args.img_size}/{args.patch_size} → {num_patches}). "
                f"Use matching --img_size or extract without --feat_cache_dir."
            )
    else:
        print("  Extracting features (no cache)...")
        cls_patches_list, reg_list = extract_features(
            backbone, img_files, args.img_size, device,
            batch_size=args.extract_batch_size, compress_reg=args.compress_reg,
        )

    D = cls_patches_list[0].shape[1]
    T_cp = cls_patches_list[0].shape[0]
    T_reg = reg_list[0].shape[0]
    T_full = T_cp + T_reg
    num_groups = D // args.embedding_dim
    patch_only = args.train_tokens == "patch" and args.n_prefix > 0
    T_codec = (T_cp - args.n_prefix) if patch_only else T_cp
    print(f"  D={D} T_codec={T_codec} T_seq={T_cp} T_reg={T_reg} T_full={T_full} G={num_groups}")
    if patch_only:
        print(f"  SoftPQ tokens: patches only ({T_codec}); prefix={args.n_prefix} bypassed")

    print("[3/4] Building frozen tail...")
    layer_idx = encode_end - 1  # last encoded block index
    token_hw = (token_per_dim, token_per_dim)
    if num_tail == 0:
        tail = FrozenTail([], dino.norm, device=str(device))
        print("  FrozenTail: norm-only (slot encodes all blocks; RAEv2 k=1)")
    else:
        # Keep backbone on GPU briefly to prime RoPE, then move model weights used by tail
        tail = build_dinov3_tail(backbone, layer_idx, token_hw, device)
        print(f"  Dinov3FrozenTail: {len(tail.blocks)} blocks + norm (RoPE-aware)")

    backbone.model.cpu()
    torch.cuda.empty_cache()

    rae_decoder = None
    token_res = token_hw
    norm_eps = float(dino.norm.eps)

    cls_patches_array = np.stack(cls_patches_list)
    reg_array = np.stack(reg_list)
    del cls_patches_list, reg_list
    N_img = len(cls_patches_array)
    print(f"  cls_patches_array: {cls_patches_array.shape} ({cls_patches_array.nbytes / 1e9:.2f} GB)")

    print("[4/4] Training FeatureCodec...")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pq = SoftPQ(
        num_groups, args.K, args.embedding_dim,
        lmbda=args.lmbda, prior_floor=args.prior_floor,
    ).to(device)

    if args.no_transform:
        transform = None
        print("  Transform: DISABLED")
        print("  K-means init...")
        all_Z = []
        for start in range(0, N_img, 100):
            end = min(start + 100, N_img)
            X = torch.from_numpy(cls_patches_array[start:end]).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(
                    X, mode=args.norm_mode, n_prefix=args.n_prefix,
                )
            if patch_only:
                all_Z.append(Y[:, args.n_prefix:, :].reshape(-1, D).cpu())
            else:
                all_Z.append(Y.reshape(-1, D).cpu())
            del X, Y
        Z_flat = torch.cat(all_Z, dim=0)
        max_km = min(Z_flat.shape[0], args.kmeans_max_samples // num_groups)
        if Z_flat.shape[0] > max_km:
            Z_flat = Z_flat[np.random.choice(Z_flat.shape[0], max_km, replace=False)]
        pq.init_from_kmeans(Z_flat, device=str(device))
        del all_Z, Z_flat
        torch.cuda.empty_cache()
    else:
        opq_scope = "patch" if patch_only else "all"
        print(
            f"  OPQ warmup (norm={args.norm_mode}, n_prefix={args.n_prefix}, "
            f"tokens={opq_scope})..."
        )
        all_vectors = []
        for start in range(0, N_img, 100):
            end = min(start + 100, N_img)
            X = torch.from_numpy(cls_patches_array[start:end]).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(
                    X, mode=args.norm_mode, n_prefix=args.n_prefix,
                )
            all_vectors.append(
                _opq_vectors(Y, args.train_tokens, args.n_prefix, D)
            )
            del X, Y
        full_vectors = np.concatenate(all_vectors, axis=0)
        del all_vectors
        max_flat = min(full_vectors.shape[0], args.kmeans_max_samples // num_groups)
        if full_vectors.shape[0] > max_flat:
            full_vectors = full_vectors[
                np.random.choice(full_vectors.shape[0], max_flat, replace=False)
            ]
        R_opq, codebooks_opq, opq_history = learn_orfc_rotation(
            full_vectors, num_groups, args.embedding_dim, args.K,
            max_iter_orfc=20, max_iter_kmeans=100, device=str(device), verbose=True,
        )
        del full_vectors
        torch.cuda.empty_cache()
        if np.linalg.det(R_opq) < 0:
            R_opq = R_opq.copy()
            R_opq[:, -1] *= -1
            codebooks_opq = [c.copy() for c in codebooks_opq]
            codebooks_opq[-1][:, -1] *= -1
        transform = OrthogonalTransform(D).to(device)
        transform.init_from_opq(R_opq)
        pq.init_codebooks(codebooks_opq)
        print(f"  OPQ done, final MSE={opq_history[-1][0]:.8f}")

    codec = FeatureCodec(pq, transform).to(device)
    tail.to(str(device))

    if args.use_rae_tail:
        dec_info = DECODER_CONFIGS[args.decoder_type]
        decoder_path = (
            args.decoder_path
            if args.decoder_path != "auto"
            else str(_COFAI_ROOT / dec_info["default_path"])
        )
        print(f"  Loading RAE Decoder ({args.decoder_type}) from {decoder_path}")
        rae_decoder = GeneralDecoder(
            decoder_config=dec_info["cfg"],
            patch_size=16,
            image_size=16 * token_per_dim,
            pretrained_path=decoder_path,
            encoder_img_size=args.img_size,
            encoder_patch_size=args.patch_size,
            encoder_hidden_size=dec_info["encoder_hidden_size"],
            encoder_mean=list(PIXEL_MEAN),
            encoder_std=list(PIXEL_STD),
            device=device,
        )
        rae_decoder.eval()
        for p in rae_decoder.parameters():
            p.requires_grad_(False)
        print(f"  RAE Decoder frozen ({sum(p.numel() for p in rae_decoder.parameters()):,} params)")

    teacher_cache = None
    teacher_pixel_cache = None
    teacher_cls_cache = None

    if args.use_rae_tail:
        print("  Pre-computing teacher pixels (bypass RAEtail)...")
        dec_img_size = 16 * token_per_dim
        teacher_pixel_cache = np.empty(
            (N_img, 3, dec_img_size, dec_img_size), dtype=np.float16,
        )
        teacher_bs = min(args.batch_size, 4)
        with torch.no_grad():
            for start in tqdm(range(0, N_img, teacher_bs), desc="Teacher pixels"):
                end = min(start + teacher_bs, N_img)
                cp = torch.from_numpy(cls_patches_array[start:end]).float().to(device)
                reg = torch.from_numpy(reg_array[start:end]).float().to(device)
                h_full = _assemble_full(cp, reg, args.compress_reg, num_reg)
                h_after = run_tail_blocks(h_full, tail)
                feat = extract_rae_patches(h_after, num_prefix, norm_eps)
                pixels = rae_decoder.predict(feat, token_res=token_res, token_format="patch")
                teacher_pixel_cache[start:end] = pixels.cpu().half().numpy()
                del cp, reg, h_full, h_after, feat, pixels
        torch.cuda.empty_cache()
        print(f"  Teacher pixel cache: {teacher_pixel_cache.nbytes / 1e9:.2f} GB")

        if args.cls_loss_weight > 0:
            print("  Pre-computing teacher CLS (tail + affine LN)...")
            teacher_cls_cache = np.empty((N_img, D), dtype=np.float32)
            with torch.no_grad():
                for start in tqdm(range(0, N_img, args.batch_size), desc="Teacher CLS"):
                    end = min(start + args.batch_size, N_img)
                    cp = torch.from_numpy(cls_patches_array[start:end]).float().to(device)
                    reg = torch.from_numpy(reg_array[start:end]).float().to(device)
                    h_full = _assemble_full(cp, reg, args.compress_reg, num_reg)
                    h_after = run_tail_blocks(h_full, tail)
                    h_normed = tail.norm(h_after)
                    teacher_cls_cache[start:end] = h_normed[:, 0, :].cpu().numpy()
                    del cp, reg, h_full, h_after, h_normed
            torch.cuda.empty_cache()
            print(f"  Teacher CLS cache: {teacher_cls_cache.nbytes / 1e6:.1f} MB")
    else:
        print("  Pre-computing FrozenTail teacher...")
        teacher_cache = np.empty((N_img, T_full, D), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, N_img, args.batch_size):
                end = min(start + args.batch_size, N_img)
                cp = torch.from_numpy(cls_patches_array[start:end]).float().to(device)
                reg = torch.from_numpy(reg_array[start:end]).float().to(device)
                h_full = _assemble_full(cp, reg, args.compress_reg, num_reg)
                teacher_cache[start:end] = tail.forward_nograd(h_full).cpu().numpy()
                del cp, reg, h_full
        torch.cuda.empty_cache()

    trainable = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    indices = np.arange(N_img)
    for epoch in range(args.epochs):
        t_epoch = time.time()
        if args.epochs > 1:
            progress = epoch / (args.epochs - 1)
            tau = args.tau_start * (args.tau_end / args.tau_start) ** progress
        else:
            tau = args.tau_start
        pq.temperature = tau

        np.random.shuffle(indices)
        total_distortion = 0.0
        total_rate = 0.0
        total_cls_loss = 0.0
        usage_acc = torch.zeros(num_groups, args.K, device=device)

        codec.train()
        for start in range(0, N_img, args.batch_size):
            end = min(start + args.batch_size, N_img)
            batch_idx = indices[start:end]
            B = len(batch_idx)

            X_cp = torch.from_numpy(cls_patches_array[batch_idx]).float().to(device)
            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(
                    X_cp, mode=args.norm_mode, n_prefix=args.n_prefix,
                )
            Y_hat, usage = codec_forward(
                Y, codec, n_prefix=args.n_prefix, prefix_bypass=patch_only,
            )
            X_cp_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)

            X_reg = torch.from_numpy(reg_array[batch_idx]).float().to(device)
            h_hat_full = _assemble_full(X_cp_hat, X_reg, args.compress_reg, num_reg)

            if args.use_rae_tail:
                h_after = run_tail_blocks(h_hat_full, tail)
                feat_hat = extract_rae_patches(h_after, num_prefix, norm_eps)
                logits = decoder_forward(
                    rae_decoder, feat_hat, token_res,
                    use_checkpoint=args.grad_checkpoint,
                )
                x_hat = rae_decoder.unpatchify(logits, token_res)
                x_hat = (
                    x_hat * rae_decoder.encoder_std.to(x_hat.device)
                    + rae_decoder.encoder_mean.to(x_hat.device)
                )
                Y_teacher_px = torch.from_numpy(
                    teacher_pixel_cache[batch_idx]
                ).float().to(device)
                distortion = F.l1_loss(x_hat, Y_teacher_px, reduction="sum") / B

                cls_loss = torch.tensor(0.0, device=device)
                if args.cls_loss_weight > 0 and teacher_cls_cache is not None:
                    h_normed = tail.norm(h_after)
                    cls_hat_normed = h_normed[:, 0, :]
                    cls_teacher = torch.from_numpy(
                        teacher_cls_cache[batch_idx]
                    ).float().to(device)
                    cls_loss = F.mse_loss(cls_hat_normed, cls_teacher, reduction="sum") / B
                    del h_normed, cls_hat_normed, cls_teacher
                del feat_hat, logits, x_hat, Y_teacher_px, h_after
            else:
                Y_teacher = torch.from_numpy(teacher_cache[batch_idx]).float().to(device)
                X_hat_out = tail(h_hat_full)
                err = (Y_teacher - X_hat_out) ** 2
                if patch_only:
                    distortion = err[:, args.n_prefix:, :].sum() / B
                else:
                    distortion = (
                        err[:, 0:1, :].sum() + err[:, num_prefix:, :].sum()
                    ) / B
                del Y_teacher, X_hat_out, err
                cls_loss = torch.tensor(0.0, device=device)

            if codec.use_rate:
                rate_bits = codec._last_rate * T_codec
                loss = rate_bits + args.lmbda * distortion
            else:
                loss = distortion
            if args.use_rae_tail and args.cls_loss_weight > 0:
                loss = loss + args.cls_loss_weight * cls_loss

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), args.grad_clip)
            optimizer.step()

            total_distortion += distortion.item() * B
            if args.use_rae_tail and args.cls_loss_weight > 0:
                total_cls_loss += cls_loss.item() * B
            if codec.use_rate:
                total_rate += codec._last_rate.item() * B
            usage_acc += usage.detach()
            del X_cp, X_reg, Y, Mu, Std, Y_hat, X_cp_hat, h_hat_full, loss, distortion

        scheduler.step()
        avg_d = total_distortion / N_img
        ppl = compute_perplexity(usage_acc)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            rate_str = f"  R={total_rate / N_img:.2f}b/t" if codec.use_rate else ""
            cls_str = (
                f"  CLS={total_cls_loss / N_img:.4f}"
                if (args.use_rae_tail and args.cls_loss_weight > 0) else ""
            )
            print(
                f"  ep {epoch:3d}/{args.epochs}  D={avg_d:.1f}  ppl={ppl:.1f}"
                f"{rate_str}{cls_str}  tau={tau:.4f}  ({time.time() - t_epoch:.1f}s)"
            )

    out_dir = Path(args.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tail_tag = "_raetail" if args.use_rae_tail else ""
    notfm_tag = "_noR" if args.no_transform else ""
    reg_tag = "_wReg" if args.compress_reg else ""
    patch_tag = "_ptpatch" if args.train_tokens == "patch" else ""
    norm_tag = f"_{args.norm_mode}" if args.norm_mode != "per_image" else ""
    cls_tag = (
        f"_cls{args.cls_loss_weight}"
        if (args.use_rae_tail and args.cls_loss_weight > 0) else ""
    )
    dec_tag = DECODER_CONFIGS[args.decoder_type]["tag"] if args.use_rae_tail else ""
    ckpt_name = (
        f"slot{encode_end:02d}_K{args.K}_e{args.embedding_dim}"
        f"_lmbda{args.lmbda}_ep{args.epochs}_n{N_img}"
        f"{tail_tag}{dec_tag}{cls_tag}{notfm_tag}{norm_tag}{reg_tag}{patch_tag}.pt"
    )
    ckpt_path = out_dir / ckpt_name
    save_codec(
        codec, str(ckpt_path),
        norm_mode=args.norm_mode,
        n_prefix=args.n_prefix,
        train_tokens=args.train_tokens,
        compress_reg=bool(args.compress_reg),
        use_transform=not args.no_transform,
        slot=int(encode_end),
        img_size=int(args.img_size),
    )
    print(f"\n  Codec saved: {ckpt_path}")
    print(f"  meta: norm_mode={args.norm_mode} n_prefix={args.n_prefix} "
          f"train_tokens={args.train_tokens} compress_reg={args.compress_reg}")

    print("  Computing train-prior PMF for sidecar .npz...")
    pmf = compute_histogram_pmf(
        codec,
        cls_patches_array,
        norm_mode=args.norm_mode,
        n_prefix=args.n_prefix,
        device=device,
        batch_size=args.batch_size,
        token_slice="patch" if patch_only else "all",
    )
    npz_path = save_codec_npz(
        codec, npz_path_for_codec(ckpt_path), pmf, source_pt=ckpt_path,
    )
    pmf_ent = float(-(pmf * np.log2(pmf + 1e-30)).sum(axis=1).mean())
    print(f"  Sidecar NPZ saved: {npz_path}")
    print(f"    PMF entropy: {pmf_ent:.4f} bits/group")
    print(f"{'=' * 70}\n")
    return str(ckpt_path)


def main():
    p = argparse.ArgumentParser(description="DINOv3-L RAEtail + CLS MSE SoftPQ training")
    p.add_argument("--pathname_list", type=str,
                   default="/data4/workspace/zlt/featcodec/utils/imagenet_selected_pathname5000.txt")
    p.add_argument("--imagenet_root", type=str,
                   default="/data4/workspace/zlt/featcodec/data/imagenet/images/val")
    p.add_argument(
        "--backbone_ckpt", type=str,
        default="weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.pth",
    )
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--slot", type=int, default=24,
                   help="Encode blocks[:slot]. 24 = all blocks (RAEv2 k=1)")
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--embedding_dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--prior_floor", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--extract_batch_size", type=int, default=4)
    p.add_argument("--max_images", type=int, default=5000)
    p.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--use_rae_tail", action="store_true")
    p.add_argument("--decoder_type", type=str, default="ViTXL",
                   choices=list(DECODER_CONFIGS.keys()))
    p.add_argument("--decoder_path", type=str, default="auto")
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--cls_loss_weight", type=float, default=0.0)
    p.add_argument("--no_transform", action="store_true")
    p.add_argument("--compress_reg", action="store_true")
    p.add_argument(
        "--train_tokens", type=str, default="all", choices=["all", "patch"],
        help="SoftPQ/OPQ scope: 'all' (default) or 'patch' (CLS+reg identity bypass).",
    )
    p.add_argument(
        "--norm_mode", type=str, default="per_image",
        choices=["per_image", "split_cls_patch", "split_reg_cls_patch", "per_token_ln"],
        help="Feature norm before SoftPQ. split_reg_cls_patch: reg own stats, "
             "cls+patch shared (isolates extreme DINOv3 reg tokens).",
    )
    p.add_argument(
        "--n_prefix", type=int, default=0,
        help="Prefix length for split_* norms (0 = auto: num_prefix_tokens).",
    )
    p.add_argument("--feat_cache_dir", type=str, default="none")
    p.add_argument(
        "--weights_dir", type=str,
        default="weights/orfc_2446_dinov3/dinov3_large_256px",
    )
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

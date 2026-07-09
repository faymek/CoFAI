"""VTM encode/decode for (N, C) token matrices — per-image min-max linear quant."""

from __future__ import annotations

import os
import subprocess as sp
from pathlib import Path

import numpy as np


def quantize_linear(x: np.ndarray, bit_depth: int):
    x = x.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        scale = 1.0
    else:
        scale = ((1 << bit_depth) - 1) / (xmax - xmin)
    q = np.round((x - xmin) * scale)
    dt = np.uint16 if bit_depth > 8 else np.uint8
    q = q.astype(dt)
    meta = {"min": xmin, "max": xmax, "bit_depth": bit_depth}
    return q, meta


def dequantize_linear(q: np.ndarray, meta: dict) -> np.ndarray:
    q = q.astype(np.float32)
    b = int(meta["bit_depth"])
    xmin, xmax = float(meta["min"]), float(meta["max"])
    if xmax <= xmin:
        return np.full_like(q, xmin, dtype=np.float32)
    scale = ((1 << b) - 1) / (xmax - xmin)
    return (q / scale + xmin).astype(np.float32)


def vtm_encode_decode(
    feat_2d: np.ndarray,
    qp: int,
    stem: str,
    tmp_dir: Path,
    *,
    vtm_encoder: str,
    vtm_decoder: str,
    vtm_cfg: str,
    bit_depth: int,
) -> tuple[np.ndarray, int]:
    """Encode/decode a 2D feature frame. Returns reconstructed array and bitstream bytes."""
    h, w = feat_2d.shape
    q, meta = quantize_linear(feat_2d, bit_depth)

    uid = f"dinov3_qp{qp}_{stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yuv_path = tmp_dir / f"{uid}.y"
    bitstream = tmp_dir / f"{uid}.vvc"
    enc_log = tmp_dir / f"{uid}.enc.log"
    dec_yuv = tmp_dir / f"{uid}.dec.y"
    dec_log = tmp_dir / f"{uid}.dec.log"

    q.tofile(str(yuv_path))

    cmd_enc = [
        vtm_encoder,
        "-c",
        vtm_cfg,
        "-i",
        str(yuv_path),
        "-b",
        str(bitstream),
        f"--SourceWidth={w}",
        f"--SourceHeight={h}",
        "--FramesToBeEncoded=1",
        "--FrameRate=1",
        "--InputChromaFormat=400",
        "--ConformanceWindowMode=1",
        f"--InternalBitDepth={bit_depth}",
        f"--InputBitDepth={bit_depth}",
        f"--OutputBitDepth={bit_depth}",
        f"--QP={qp}",
    ]
    with open(enc_log, "w") as f:
        sp.run(cmd_enc, stdout=f, stderr=sp.STDOUT, check=True)

    cmd_dec = [vtm_decoder, "-b", str(bitstream), "-o", str(dec_yuv)]
    with open(dec_log, "w") as f:
        sp.run(cmd_dec, stdout=f, stderr=sp.STDOUT, check=True)

    dt = np.uint16 if bit_depth > 8 else np.uint8
    q_rec = np.fromfile(str(dec_yuv), dtype=dt).reshape(h, w)
    feat_rec = dequantize_linear(q_rec, meta)
    bs_bytes = os.path.getsize(bitstream)

    for p in (yuv_path, bitstream, enc_log, dec_yuv, dec_log):
        try:
            p.unlink()
        except OSError:
            pass

    return feat_rec, bs_bytes


def vtm_worker(task: dict) -> dict:
    """ProcessPool worker. Atomic write + resume-friendly output path."""
    try:
        token_path = task["token_path"]
        out_path = task["out_path"]
        qp = task["qp"]
        tmp_dir = Path(task["tmp_dir"])
        stem = Path(token_path).stem
        tokens = np.load(token_path)
        if tokens.ndim != 2:
            raise ValueError(f"Expected 2D tokens, got {tokens.shape}")

        feat_rec, bs_bytes = vtm_encode_decode(
            tokens.astype(np.float32),
            qp,
            stem,
            tmp_dir,
            vtm_encoder=task["vtm_encoder"],
            vtm_decoder=task["vtm_decoder"],
            vtm_cfg=task["vtm_cfg"],
            bit_depth=task["bit_depth"],
        )

        tmp_out = out_path.replace(".npy", ".tmp.npy")
        np.save(tmp_out, feat_rec.astype(np.float32))
        os.rename(tmp_out, out_path)

        n_elements = int(feat_rec.size)
        bpfp = bs_bytes * 8 / max(n_elements, 1)
        return {
            "stem": stem,
            "qp": qp,
            "bs_bytes": bs_bytes,
            "bpfp": bpfp,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "stem": Path(task.get("token_path", "unknown")).stem,
            "qp": task.get("qp"),
            "error": str(exc),
        }

import os
import sys
import json
import time
import subprocess
import numpy as np
from tempfile import mkstemp

import torch
import torch.nn as nn
from omegaconf import OmegaConf


def truncation(feat, trun_low, trun_high):
    """Truncate features to specified range.

    Args:
        feat (numpy.ndarray): Input features with shape (N, C, H, W).
        trun_low (float or list[float]): Lower bound(s) for truncation.
            If list, should have length C for per-channel truncation.
        trun_high (float or list[float]): Upper bound(s) for truncation.
            If list, should have length C for per-channel truncation.

    Returns:
        trun_feat (numpy.ndarray): Truncated features with same shape as input.
    """
    trun_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(trun_low, list):
        for idx in range(len(trun_low)):
            trun_feat[:, idx, :, :] = np.clip(
                feat[:, idx, :, :], trun_low[idx], trun_high[idx]
            )
    else:
        trun_feat = np.clip(feat, trun_low, trun_high)

    return trun_feat


def load_quantization_points(file_path: str or list[str]):
    """Load quantization points from a file or a list of files.

    Args:
        file_path (str or list[str]): Path to load the quantization points from.
            Can be a single file path (str) or a list of file paths (list[str]).

    Returns:
        quantization_points (numpy.ndarray or list[numpy.ndarray]): Loaded quantization points.
            If file_path is a single path, returns a single numpy.ndarray.
            If file_path is a list of paths, returns a list of numpy.ndarray.
    """

    def load_file(path):
        with open(path, "r") as f:
            quantization_points = np.array(json.load(f))
        # print(f"Quantization points loaded from {path}")
        return quantization_points

    if isinstance(file_path, list):
        # Load quantization points from each file in the list
        return [load_file(path) for path in file_path]
    elif isinstance(file_path, str):
        # Load quantization points from a single file
        return load_file(file_path)
    else:
        raise ValueError("file_path must be a string or a list of strings.")


def uniform_quantization(feat, min_v, max_v, bit_depth):
    """Apply uniform quantization to features.

    Args:
        feat (numpy.ndarray): Input features with shape (N, C, H, W).
        min_v (float or list[float]): Minimum value(s) for quantization range.
            If list, should have length C for per-channel quantization.
        max_v (float or list[float]): Maximum value(s) for quantization range.
            If list, should have length C for per-channel quantization.
        bit_depth (int): Bit depth for quantization (e.g., 8 or 10).

    Returns:
        quant_feat (numpy.ndarray): Quantized features as uint8 or uint16,
            depending on bit_depth.
    """
    quant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) - 1) / (max_v[idx] - min_v[idx])
            quant_feat[:, idx, :, :] = (feat[:, idx, :, :] - min_v[idx]) * scale
    else:
        scale = ((2**bit_depth) - 1) / (max_v - min_v)
        quant_feat = (feat - min_v) * scale

    quant_feat = (
        quant_feat.astype(np.uint16) if bit_depth > 8 else quant_feat.astype(np.uint8)
    )
    return quant_feat


def uniform_dequantization(feat, min_v, max_v, bit_depth):
    """Apply uniform dequantization to quantized features.

    Args:
        feat (numpy.ndarray): Quantized features (uint8 or uint16) with shape (N, C, H, W).
        min_v (float or list[float]): Minimum value(s) used during quantization.
            If list, should have length C for per-channel dequantization.
        max_v (float or list[float]): Maximum value(s) used during quantization.
            If list, should have length C for per-channel dequantization.
        bit_depth (int): Bit depth used during quantization (e.g., 8 or 10).

    Returns:
        dequant_feat (numpy.ndarray): Dequantized features as float32 with same shape as input.
    """
    feat = feat.astype(np.float32)
    dequant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) - 1) / (max_v[idx] - min_v[idx])
            dequant_feat[:, idx, :, :] = feat[:, idx, :, :] / scale + min_v[idx]
    else:
        scale = ((2**bit_depth) - 1) / (max_v - min_v)
        dequant_feat = feat / scale + min_v
    return dequant_feat


def nonlinear_quantization(data, quantization_points, bit_depth):
    """Apply nonlinear quantization to data using quantization points.

    Args:
        data (numpy.ndarray): Original floating-point array with shape (N, C, H, W).
        quantization_points (numpy.ndarray or list[numpy.ndarray]): Quantization points.
            If single array, applied to all channels. If list, should have length C
            with one array per channel.
        bit_depth (int): Bit depth for quantization (e.g., 8 or 10).

    Returns:
        quantized_data (numpy.ndarray): Quantized integer array (uint8 or uint16)
            with the same shape as the input data.
    """
    if isinstance(quantization_points, np.ndarray):
        # If quantization_points is a single array, apply it to all channels
        num_levels = len(quantization_points)
        data_flat = data.flatten()
        quantized_data_flat = np.digitize(data_flat, quantization_points) - 1
        quantized_data_flat = np.clip(quantized_data_flat, 0, num_levels - 1)
        quantized_data = quantized_data_flat.reshape(data.shape)
    elif isinstance(quantization_points, list):
        if len(quantization_points) != data.shape[1]:
            raise ValueError(
                "Length of quantization_points list must match the number of channels (C) in data."
            )

        quantized_data = np.zeros_like(data, dtype=int)
        # Apply different quantization points to each channel
        for i, qp in enumerate(quantization_points):
            num_levels = len(qp)
            channel_data = data[:, i, :, :]
            channel_data_flat = channel_data.flatten()
            quantized_channel_flat = np.digitize(channel_data_flat, qp) - 1
            quantized_channel_flat = np.clip(quantized_channel_flat, 0, num_levels - 1)
            quantized_data[:, i, :, :] = quantized_channel_flat.reshape(
                channel_data.shape
            )
    else:
        raise ValueError(
            "quantization_points must be a numpy array or a list of numpy arrays."
        )

    quantized_data = (
        quantized_data.astype(np.uint16)
        if bit_depth > 8
        else quantized_data.astype(np.uint8)
    )
    return quantized_data


def nonlinear_dequantization(quantized_data, quantization_points):
    """Dequantize quantized data back to approximate original floating-point values.

    Args:
        quantized_data (numpy.ndarray): Quantized integer array with shape (N, C, H, W).
        quantization_points (numpy.ndarray or list[numpy.ndarray]): Quantization points.
            If single array, applied to all channels. If list, should have length C
            with one array per channel. Points are automatically sorted.

    Returns:
        dequantized_data (numpy.ndarray): Dequantized floating-point array (float32)
            with the same shape as the input data.
    """
    if isinstance(quantization_points, np.ndarray):
        # If quantization_points is a single array, apply it to all channels
        quantization_points = np.sort(quantization_points)  # Ensure points are sorted
        dequantized_data = quantization_points[quantized_data]
    elif isinstance(quantization_points, list):
        if len(quantization_points) != quantized_data.shape[1]:
            raise ValueError(
                "Length of quantization_points list must match the number of channels (C) in quantized_data."
            )

        dequantized_data = np.zeros_like(quantized_data, dtype=np.float32)
        # Apply different quantization points to each channel
        for i, qp in enumerate(quantization_points):
            qp = np.sort(qp)  # Ensure points are sorted
            channel_data = quantized_data[:, i, :, :]
            dequantized_data[:, i, :, :] = qp[channel_data]
    else:
        raise ValueError(
            "quantization_points must be a numpy array or a list of numpy arrays."
        )

    # print(dequantized_data.dtype)
    dequantized_data = dequantized_data.astype(np.float32)
    return dequantized_data


def packing(feat, model_type):
    """Pack features into 2D format for video codec encoding.

    Args:
        feat (numpy.ndarray): Input features with shape (N, C, H, W).
        model_type (str): Model type determining packing strategy.
            Options: "llama3", "dinov2", "sd3".

    Returns:
        packed_feat (numpy.ndarray): Packed features as 2D array suitable for
            video codec encoding.
    """
    N, C, H, W = feat.shape
    if model_type == "llama3":
        feat = feat[0, 0, :, :]
    elif model_type == "dinov2":
        feat = feat.transpose(0, 2, 1, 3).reshape(N * H, C * W)
    elif model_type == "sd3":
        feat = (
            feat.reshape(int(C / 4), int(C / 4), H, W)
            .transpose(0, 2, 1, 3)
            .reshape(int(C / 4 * H), int(C / 4 * W))
        )
    return feat


def unpacking(feat, shape, model_type):
    """Unpack 2D features back to original 4D format.

    Args:
        feat (numpy.ndarray): Packed 2D features from video codec decoding.
        shape (tuple[int, int, int, int]): Target shape (N, C, H, W) to unpack to.
        model_type (str): Model type determining unpacking strategy.
            Options: "llama3", "dinov2", "sd3".

    Returns:
        unpacked_feat (numpy.ndarray): Unpacked features with shape (N, C, H, W).
    """
    N, C, H, W = shape
    if model_type == "llama3":
        feat = np.expand_dims(feat, axis=0)
        feat = np.expand_dims(feat, axis=0)
    elif model_type == "dinov2":
        feat = feat.reshape(N, H, C, W).transpose(0, 2, 1, 3)
    elif model_type == "sd3":
        feat = (
            feat.reshape(int(C / 4), H, int(C / 4), W)
            .transpose(0, 2, 1, 3)
            .reshape(N, C, H, W)
        )
    return feat


def run_shell(cmd, ignore_returncodes=None):
    """Run shell command and return output.

    Args:
        cmd (str or list[str]): Command to execute. If list, will be joined with spaces.
        ignore_returncodes (list[int], optional): List of return codes to ignore.
            If command returns one of these codes, output is returned instead of exiting.
            Defaults to None.

    Returns:
        output (str): Decoded command output as ASCII string.

    Raises:
        SystemExit: If command fails and return code is not in ignore_returncodes.
    """
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    try:
        rv = subprocess.check_output(cmd, shell=True)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def filesize(filepath: str) -> int:
    """Return file size in bytes.

    Args:
        filepath (str): Path to the file.

    Returns:
        size (int): File size in bytes.

    Raises:
        ValueError: If filepath is not a valid file.
    """
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size


class VtmCodec:
    """VTM (VVC Test Model) codec wrapper for video encoding and decoding.

    This class provides an interface to VTM encoder and decoder executables
    for compressing and decompressing video data.
    """

    def __init__(self, repo_dir):
        """Initialize VTM codec.

        Args:
            repo_dir (str): Path to VTM repository directory containing bin/ and cfg/ folders.
        """
        self.encoder_path = os.path.join(repo_dir, "bin", "EncoderAppStatic")
        self.decoder_path = os.path.join(repo_dir, "bin", "DecoderAppStatic")
        self.config_path = os.path.join(repo_dir, "cfg", "encoder_intra_vtm.cfg")

        self.version = run_shell(f"{self.encoder_path} |grep Version").split()[4]
        self.description = f"VTM-{self.version}"

    def compress(
        self,
        raw_path,
        bin_path,
        width,
        height,
        qp: int,
        bitdepth: int = 8,
        chroma_format: str = "400",
    ):
        """Compress raw video file using VTM encoder.

        Args:
            raw_path (str): Path to input raw YUV file.
            bin_path (str): Path to output compressed bitstream file.
            width (int): Video width in pixels.
            height (int): Video height in pixels.
            qp (int): Quantization parameter (0-51, lower is higher quality).
            bitdepth (int): Bit depth (8 or 10). Defaults to 8.
            chroma_format (str): Chroma format. Defaults to "400" (grayscale).
        """
        cmd = (
            f"{self.encoder_path} -c {self.config_path} "
            f'-i {raw_path} -o "" -b {bin_path} -q {qp} --ConformanceWindowMode=1 '
            f"-wdt {width} -hgt {height} -f 1 -fr 1 "
            f"--InternalBitDepth={bitdepth} --InputBitDepth={bitdepth} "
            f"--InputChromaFormat={chroma_format} --OutputBitDepth={bitdepth} "
        )
        run_shell(cmd)

    def decompress(self, bin_path, rec_path, bit_depth=8):
        """Decompress VTM bitstream to raw video file.

        Args:
            bin_path (str): Path to input compressed bitstream file.
            rec_path (str): Path to output reconstructed YUV file.
            bit_depth (int): Bit depth (8 or 10). Defaults to 8.
        """
        cmd = f"{self.decoder_path} -b {bin_path} -o {rec_path} -d {bit_depth}"
        run_shell(cmd)


class VtmImageCodec:
    """VTM-based image codec (placeholder class).

    This class is reserved for future image codec implementation using VTM.
    """

    pass


class VtmFeatureCodec:
    """VTM-based feature codec for compressing neural network features.

    This codec applies truncation, quantization, packing, VTM encoding/decoding,
    and post-processing to compress features from various model types (llama3, dinov2, sd3).
    """

    def __init__(self, cfg):
        """Initialize VTM feature codec.

        Args:
            cfg (dict): Configuration object containing:

                - vtm_path (str): Path to VTM repository directory.
                - trun_flag (bool): Whether to apply truncation.
                - trun_low (float or list[float]): Lower truncation bound(s).
                - trun_high (float or list[float]): Upper truncation bound(s).
                - bit_depth (int): Bit depth for quantization.
                - model_type (str): Model type ("llama3", "dinov2", or "sd3").
        """
        self.cfg = cfg
        self.codec = VtmCodec(cfg.vtm_path)

    def forward_test(self, org_feat, qp: int):
        """Forward test method for debugging (includes timing measurements).

        This method performs full encode-decode cycle and returns both compressed
        representation and decoded features with timing information.

        Args:
            org_feat (numpy.ndarray): Original features to compress.
            qp (int): Quantization parameter for VTM encoding.

        Returns:
            coded_unit (dict): Dictionary containing:

                - "strings" (dict): Compressed bitstring with key "vtm".
                - "pstate" (dict): State information including paths and shapes.
            decoded (dict): Dictionary containing:

                - "h_hat" (numpy.ndarray): Decoded features.
        """
        cfg = self.cfg
        org_feat_shape = org_feat.shape

        # Truncation
        if cfg.trun_flag is True:
            feat = truncation(org_feat, cfg.trun_low, cfg.trun_high)
        # Quantization
        feat = uniform_quantization(feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth)
        # Packing
        pack_feat = packing(feat, cfg.model_type)
        fd, raw_path = mkstemp(suffix=".yuv")
        bin_path = os.path.splitext(raw_path)[0] + ".bin"
        rec_path = os.path.splitext(raw_path)[0] + "_rec.yuv"
        with open(raw_path, "wb") as f:
            pack_feat.tofile(f)

        # VTM encoding
        start = time.time()
        self.codec.compress(
            raw_path,
            bin_path,
            pack_feat.shape[1],
            pack_feat.shape[0],
            qp,
            cfg.bit_depth,
            "400",
        )
        _enc_time = time.time() - start

        # VTM decoding
        start = time.time()
        self.codec.decompress(bin_path, rec_path, cfg.bit_depth)
        _dec_time = time.time() - start

        # Load decoded YUV
        with open(rec_path, "rb") as f:
            decoded_yuv = np.fromfile(
                f, dtype=np.uint16 if cfg.bit_depth == 10 else np.uint8
            )
            decoded_yuv = decoded_yuv.reshape(pack_feat.shape)

        # Postprocessing
        unpack_feat = unpacking(decoded_yuv, org_feat_shape, cfg.model_type)

        # Dequantization
        dequant_feat = uniform_dequantization(
            unpack_feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth
        )

        # Save features
        if cfg.model_type == "sd3":
            dequant_feat = dequant_feat.astype(np.float16)

        bitstring = open(bin_path, "rb").read()
        coded_unit = {
            "strings": {"vtm": [[bitstring]]},
            "pstate": {
                "bin_path": bin_path,
                "pack_shape": pack_feat.shape,
                "feat_shape": org_feat.shape,
                "bit_depth": cfg.bit_depth,
            },
        }
        decoded = {
            "h_hat": dequant_feat,
        }

        # cleanup encoder input
        os.close(fd)
        os.unlink(raw_path)
        os.unlink(bin_path)
        os.unlink(rec_path)

        return coded_unit, decoded

    def compress(
        self,
        org_feat,
        qp: int,
    ):
        """Compress features using VTM codec.

        Expected feature shape: (N_crop, N_layer, H*W+1, C)

        Args:
            org_feat (numpy.ndarray): Original features to compress.
            qp (int): Quantization parameter for VTM encoding.

        Returns:
            output (dict): Dictionary containing:

                - "strings" (dict): Compressed bitstring with key "vtm".
                - "pstate" (dict): State information including:
                    - "bin_path" (str): Path to compressed bitstream.
                    - "pack_shape" (tuple): Shape of packed features.
                    - "feat_shape" (tuple): Original feature shape.
                    - "bit_depth" (int): Bit depth used.
        """
        cfg = self.cfg

        # Truncation
        if cfg.trun_flag is True:
            feat = truncation(org_feat, cfg.trun_low, cfg.trun_high)
        # Quantization
        feat = uniform_quantization(feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth)
        # Packing
        pack_feat = packing(feat, cfg.model_type)
        fd, raw_path = mkstemp(suffix=".yuv")
        bin_path = os.path.splitext(raw_path)[0] + ".bin"
        with open(raw_path, "wb") as f:
            pack_feat.tofile(f)

        # VTM encoding
        self.codec.compress(
            raw_path,
            bin_path,
            pack_feat.shape[1],
            pack_feat.shape[0],
            qp,
            cfg.bit_depth,
            "400",
        )
        os.close(fd)
        os.unlink(raw_path)
        bitstring = open(bin_path, "rb").read()
        return {
            "strings": {"vtm": [[bitstring]]},
            "pstate": {
                "bin_path": bin_path,
                "pack_shape": pack_feat.shape,
                "feat_shape": org_feat.shape,
                "bit_depth": cfg.bit_depth,
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        """Decompress features from VTM bitstream.

        Note: model_type, bit_depth, trun_low, trun_high are fixed in self.cfg.

        Args:
            strings (dict): Dictionary with key "vtm" containing compressed bitstring.
            pstate (dict): State dictionary containing:

                - "bin_path" (str): Path to compressed bitstream.
                - "pack_shape" (tuple): Shape of packed features.
                - "feat_shape" (tuple): Target feature shape.
                - "bit_depth" (int): Bit depth used.
            **kwargs (dict): Additional keyword arguments (unused).

        Returns:
            decoded (dict): Dictionary containing:
                - "h_hat" (numpy.ndarray): Decoded features with original shape.
        """
        bin_path = pstate["bin_path"]
        pack_shape = pstate["pack_shape"]
        feat_shape = pstate["feat_shape"]
        # model_type, bit_depth, trun_low, trun_high is fixed in self.cfg
        cfg = self.cfg
        rec_path = os.path.splitext(bin_path)[0] + "_rec.yuv"
        # VTM decoding
        self.codec.decompress(bin_path, rec_path, cfg.bit_depth)

        # Load decoded YUV
        with open(rec_path, "rb") as f:
            decoded_yuv = np.fromfile(
                f, dtype=np.uint16 if cfg.bit_depth == 10 else np.uint8
            )
            decoded_yuv = decoded_yuv.reshape(pack_shape)

        # Postprocessing
        unpack_feat = unpacking(decoded_yuv, feat_shape, cfg.model_type)

        # Dequantization
        dequant_feat = uniform_dequantization(
            unpack_feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth
        )

        # Save features
        if cfg.model_type == "sd3":
            dequant_feat = dequant_feat.astype(np.float16)

        decoded = {
            "h_hat": dequant_feat,
        }
        os.unlink(bin_path)
        os.unlink(rec_path)

        return decoded


class VtmLatentCodec(nn.Module):
    """Adapt :class:`VtmFeatureCodec` to the latent-codec API.

    ``DinoFeatureCodecModel`` / ``DinoSlideFeatureCodecModel`` expect a codec that
    exposes ``forward(h, token_res, qp) -> {"h_hat", "strings", ...}``,
    ``compress(h, token_res, qp) -> {"strings", "pstate"}`` and
    ``decompress(strings, pstate) -> {"h_hat"}`` operating on encoder token
    tensors of shape ``(B, N, C)``. The raw :class:`VtmFeatureCodec` instead works
    on numpy arrays packed as ``(B, 1, N, C)`` (its ``model_type="dinov2"``
    packing) and keeps the VVC bitstream on disk between ``compress`` and
    ``decompress``.

    This thin wrapper bridges the two: tokens ``(B, N, C)`` are reshaped to the
    ``(B, 1, N, C)`` array VTM expects, run through truncation + uniform
    quantization + VTM intra coding, and the dequantized result is returned as a
    torch tensor on the codec's device. Unlike VQFC, VTM compresses *all* tokens
    (cls / register prefix included), matching ``Dinov2TimmVTM`` /
    ``Dinov2TimmSlideVtm``; ``token_res`` is accepted for API symmetry but unused.

    When used by :class:`DinoSlideFeatureCodecModel` with
    ``slide_codec_mode="stacked"``, the model stacks every crop into the batch
    dimension so VTM packs all crops into a single image and encodes them with one
    VTM call -- reproducing the original sliding-window VTM rate.

    Args:
        cfg (dict | DictConfig): Configuration forwarded to
            :class:`VtmFeatureCodec` (``vtm_path``, ``model_type``, ``bit_depth``,
            ``trun_flag`` / ``trun_low`` / ``trun_high``, etc.). Plain dicts are
            wrapped with OmegaConf so attribute access (``cfg.vtm_path``) works
            after the engine resolves the model config to a container.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        self.vtm = VtmFeatureCodec(cfg=cfg)
        # VtmFeatureCodec holds no tensors; this buffer tracks the device the
        # module was moved to so ``decompress`` (which has no input tensor) can
        # place ``h_hat`` correctly.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def _device(self) -> torch.device:
        return self._device_anchor.device

    @staticmethod
    def _tokens_to_array(h: torch.Tensor) -> np.ndarray:
        """(B, N, C) encoder tokens -> (B, 1, N, C) numpy for VTM dinov2 packing."""
        return h.detach().cpu().numpy()[:, None, :, :]

    def _array_to_tokens(
        self, arr: np.ndarray, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """(B, 1, N, C) VTM output -> (B, N, C) torch tokens on ``device``."""
        return torch.from_numpy(np.ascontiguousarray(arr[:, 0, :, :])).to(
            device=device, dtype=dtype
        )

    def forward(self, h, token_res=None, qp=0, **kwargs):
        feat_np = self._tokens_to_array(h)
        coded_unit, decoded = self.vtm.forward_test(feat_np, qp=int(qp))
        h_hat = self._array_to_tokens(decoded["h_hat"], device=h.device, dtype=h.dtype)
        return {
            "h_hat": h_hat,
            "strings": coded_unit["strings"],
            "pstate": coded_unit["pstate"],
        }

    def compress(self, h, token_res=None, qp=0, **kwargs):
        feat_np = self._tokens_to_array(h)
        encoded = self.vtm.compress(feat_np, qp=int(qp))
        return {"strings": encoded["strings"], "pstate": encoded["pstate"]}

    def decompress(self, strings, pstate, **kwargs):
        decoded = self.vtm.decompress(strings=strings, pstate=pstate)
        h_hat = self._array_to_tokens(
            decoded["h_hat"], device=self._device, dtype=torch.float32
        )
        return {"h_hat": h_hat}

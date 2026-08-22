# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Streamlined Entropy Model (SEM) for four-part spatial coding.

This implementation follows the SEM introduced in DCVC-UF: progressive mean
prediction is combined with scales from the common prior, allowing residual
symbols and CDF indexes to be packed and coded in one interaction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from compressai.latent_codecs.base import LatentCodec
from compressai.registry import register_module

from cofai.layers.dcvc_cuda_inference import restore_y_4x
from cofai.layers.dcvc_layers import DepthConvBlock

from .dcvc_entropy import EntropyCoder, GaussianEncoder

Shape = Union[Tuple[int, ...], Dict[str, Any]]


@register_module("StreamlinedEntropyModel")
class StreamlinedEntropyModel(LatentCodec):
    """SEM with progressive means and fixed scales across four partitions.

    The common prior uses a ``2 * channels + 2`` layout:
    ``q_enc, q_dec, sigma_all, mu_partition_0``. The fixed
    ``sigma_all`` tensor supplies scales for every partition.  The shared
    spatial prior predicts only a mean tensor for partitions one through
    three.
    """

    def __init__(
        self,
        channels: int,
        entropy_parameters: Optional[nn.Module] = None,
        y_spatial_prior_reduction: Optional[nn.Module] = None,
        y_spatial_prior_adaptor_1: Optional[nn.Module] = None,
        y_spatial_prior_adaptor_2: Optional[nn.Module] = None,
        y_spatial_prior_adaptor_3: Optional[nn.Module] = None,
        y_spatial_prior: Optional[nn.Module] = None,
        force_zero_thres: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(
                "StreamlinedEntropyModel requires channels divisible by 4, "
                f"got {channels}"
            )
        self._kwargs = kwargs
        self.channels = channels
        self.force_zero_thres = force_zero_thres
        self.entropy_parameters = entropy_parameters or nn.Identity()

        self.y_spatial_prior_reduction = y_spatial_prior_reduction or nn.Conv2d(
            channels * 2 + 2, channels, 1
        )
        self.y_spatial_prior_adaptor_1 = y_spatial_prior_adaptor_1 or DepthConvBlock(
            channels * 2, channels * 2, force_adaptor=True
        )
        self.y_spatial_prior_adaptor_2 = y_spatial_prior_adaptor_2 or DepthConvBlock(
            channels * 2, channels * 2, force_adaptor=True
        )
        self.y_spatial_prior_adaptor_3 = y_spatial_prior_adaptor_3 or DepthConvBlock(
            channels * 2, channels * 2, force_adaptor=True
        )
        self.y_spatial_prior = y_spatial_prior or nn.Sequential(
            DepthConvBlock(channels * 2, channels * 2),
            DepthConvBlock(channels * 2, channels * 2),
            DepthConvBlock(channels * 2, channels * 2),
            nn.Conv2d(channels * 2, channels, 1),
        )

        self.entropy_coder = None
        self.gaussian_encoder = GaussianEncoder()
        self.gaussian_encoder.force_zero_thres = force_zero_thres
        self.masks = {}

    def forward(self, y: Tensor, side_params: Tensor) -> Dict[str, Any]:
        common_params = self.entropy_parameters(side_params)
        self._validate_common_params(y, common_params, require_single_batch=False)
        y_info = self._forward_prior_4x(y, common_params)
        return {
            "likelihoods": {"y": y_info["y_likelihood"]},
            "y_hat": y_info["y_hat"],
        }

    @staticmethod
    def add_noise(x: Tensor) -> Tensor:
        noise = torch.empty_like(x).uniform_(-0.5, 0.5)
        return x + noise.detach()

    def compress(
        self, y: Tensor, side_params: Tensor, ec_part: int = 1
    ) -> Dict[str, Any]:
        common_params = self.entropy_parameters(side_params)
        self._validate_common_params(y, common_params, require_single_batch=True)
        self._ensure_updated()
        self.entropy_coder.set_use_two_entropy_coders(bool(ec_part))

        residuals, scales, y_hat = self._compress_prior_4x(y, common_params)
        self.entropy_coder.reset()
        self.gaussian_encoder.encode_y(residuals, scales)
        self.entropy_coder.flush()

        return {
            "strings": [[self.entropy_coder.get_encoded_stream()]],
            "shape": {"y": tuple(y.shape[1:]), "ec_part": ec_part},
            "y_hat": y_hat,
        }

    def decompress(
        self,
        strings: List[List[bytes]],
        shape: Shape,
        side_params: Tensor,
        ec_part: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        y_shape, shape_ec_part = self._parse_shape(shape)
        if ec_part is None:
            ec_part = shape_ec_part

        if len(strings) != 1 or len(strings[0]) != 1:
            raise ValueError("Expected exactly one main-latent bitstream")
        (bit_stream,) = strings[0]
        common_params = self.entropy_parameters(side_params)
        expected = common_params.new_empty(
            (common_params.shape[0], self.channels, *common_params.shape[-2:])
        )
        self._validate_common_params(expected, common_params, require_single_batch=True)
        if y_shape is not None and tuple(expected.shape[1:]) != y_shape:
            raise ValueError(
                f"Bitstream shape {y_shape} does not match side parameters "
                f"{tuple(expected.shape[1:])}"
            )

        self._ensure_updated()
        self.entropy_coder.set_use_two_entropy_coders(bool(ec_part))
        self.entropy_coder.set_stream(bit_stream)
        y_hat = self._decompress_prior_4x(common_params)
        return {"y_hat": y_hat}

    @staticmethod
    def _pack_parts(parts: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """Pack partitions in coder order: p0, p1, p2, then p3."""

        return torch.cat(parts, dim=1).contiguous()

    @staticmethod
    def _unpack_parts(packed: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if packed.shape[1] % 4 != 0:
            raise ValueError(
                f"Packed partition channels must be divisible by 4, got {packed.shape[1]}"
            )
        return packed.chunk(4, dim=1)

    def _partition_scales(self, scales: Tensor, masks: Tuple[Tensor, ...]) -> Tensor:
        parts = tuple(self._single_part_for_writing_4x(scales * mask) for mask in masks)
        return self._pack_parts(parts)

    def _predict_mean(
        self, y_hat: Tensor, common_params: Tensor, adaptor: nn.Module
    ) -> Tensor:
        params = torch.cat((y_hat, common_params), dim=1)
        means = self.y_spatial_prior(adaptor(params))
        if means.shape[1] != self.channels:
            raise ValueError(
                "The streamlined spatial prior must predict means only: "
                f"expected {self.channels} output channels, got {means.shape[1]}"
            )
        return means

    def _forward_prior_4x(self, y: Tensor, common_params: Tensor) -> Dict[str, Tensor]:
        q_enc, q_dec, scales, means = self._separate_prior(common_params)
        reduced_params = self.y_spatial_prior_reduction(common_params)
        masks = self._get_mask_4x(*y.shape, y.dtype, y.device)
        y = y * q_enc

        y_res_parts = []
        y_hat_so_far = torch.zeros_like(y)
        for index, (mask, adaptor) in enumerate(
            zip(
                masks,
                (
                    None,
                    self.y_spatial_prior_adaptor_1,
                    self.y_spatial_prior_adaptor_2,
                    self.y_spatial_prior_adaptor_3,
                ),
            )
        ):
            if index:
                means = self._predict_mean(y_hat_so_far, reduced_params, adaptor)
            means_hat = means * mask
            y_res = (y - means_hat) * mask
            y_q = (torch.round(y_res) - y_res).detach() + y_res
            y_hat_part = y_q + means_hat
            y_res_parts.append(y_res)
            y_hat_so_far = y_hat_so_far + y_hat_part

        y_res = sum(y_res_parts)
        likelihood_input = self.add_noise(y_res)
        y_likelihood = self.gaussian_encoder._likelihood(
            likelihood_input, scales
        ).clamp_min(1e-9)
        return {"y_hat": y_hat_so_far * q_dec, "y_likelihood": y_likelihood}

    def _compress_prior_4x(
        self, y: Tensor, common_params: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        q_enc, q_dec, scales, means = self._separate_prior(common_params)
        reduced_params = self.y_spatial_prior_reduction(common_params)
        masks = self._get_mask_4x(*y.shape, y.dtype, y.device)
        y = y * q_enc

        residual_parts = []
        y_hat_so_far = torch.zeros_like(y)
        for index, (mask, adaptor) in enumerate(
            zip(
                masks,
                (
                    None,
                    self.y_spatial_prior_adaptor_1,
                    self.y_spatial_prior_adaptor_2,
                    self.y_spatial_prior_adaptor_3,
                ),
            )
        ):
            if index:
                means = self._predict_mean(y_hat_so_far, reduced_params, adaptor)
            _, y_q, y_hat_part, _ = self._process_with_mask(y, scales, means, mask)
            residual_parts.append(self._single_part_for_writing_4x(y_q))
            y_hat_so_far = y_hat_so_far + y_hat_part

        return (
            self._pack_parts(tuple(residual_parts)),
            self._partition_scales(scales, masks),
            y_hat_so_far * q_dec,
        )

    def _decompress_prior_4x(self, common_params: Tensor) -> Tensor:
        _, q_dec, scales, means = self._separate_prior(common_params)
        reduced_params = self.y_spatial_prior_reduction(common_params)
        batch, _, height, width = means.shape
        masks = self._get_mask_4x(
            batch, self.channels, height, width, means.dtype, means.device
        )

        packed_scales = self._partition_scales(scales, masks)
        packed_residuals = self.gaussian_encoder.decode_and_get_y(
            packed_scales, means.dtype, means.device
        )
        residual_parts = self._unpack_parts(packed_residuals)

        y_hat_so_far = torch.zeros_like(means)
        for index, (residual, mask, adaptor) in enumerate(
            zip(
                residual_parts,
                masks,
                (
                    None,
                    self.y_spatial_prior_adaptor_1,
                    self.y_spatial_prior_adaptor_2,
                    self.y_spatial_prior_adaptor_3,
                ),
            )
        ):
            if index:
                means = self._predict_mean(y_hat_so_far, reduced_params, adaptor)
            y_hat_so_far = y_hat_so_far + restore_y_4x(residual, means, mask)

        return y_hat_so_far * q_dec

    def _validate_common_params(
        self, y: Tensor, common_params: Tensor, require_single_batch: bool
    ) -> None:
        if y.ndim != 4:
            raise ValueError(f"Expected a 4D y tensor, got shape {tuple(y.shape)}")
        if y.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} y channels, got {y.shape[1]}")
        if require_single_batch and y.shape[0] != 1:
            raise ValueError("Real entropy coding currently supports batch size 1")
        expected_shape = (y.shape[0], self.channels * 2 + 2, *y.shape[-2:])
        if tuple(common_params.shape) != expected_shape:
            raise ValueError(
                "Expected common parameters with shape "
                f"{expected_shape}, got {tuple(common_params.shape)}"
            )

    def update(self, force_zero_thres: Optional[float] = None) -> bool:
        if force_zero_thres is not None:
            self.force_zero_thres = force_zero_thres
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(
            self.entropy_coder, force_zero_thres=self.force_zero_thres
        )
        return True

    def _ensure_updated(self) -> None:
        if self.entropy_coder is None:
            self.update()

    @staticmethod
    def _parse_shape(shape: Shape) -> Tuple[Optional[Tuple[int, ...]], int]:
        if isinstance(shape, dict):
            y_shape = shape.get("y")
            if y_shape is not None:
                y_shape = tuple(y_shape)
            return y_shape, int(shape.get("ec_part", 1))
        if shape is None:
            return None, 1
        return tuple(shape), 1

    @staticmethod
    def _get_one_mask(micro_mask, height, width, dtype, device) -> Tensor:
        mask = torch.tensor(micro_mask, dtype=dtype, device=device)
        mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
        mask = mask[:height, :width]
        return mask.unsqueeze(0).unsqueeze(0)

    def _get_mask_4x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_{dtype}_{device}_4x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones(
                    (batch, channel // 4, height, width), dtype=dtype, device=device
                )
                m0 = self._get_one_mask(((1, 0), (0, 0)), height, width, dtype, device)
                m1 = self._get_one_mask(((0, 1), (0, 0)), height, width, dtype, device)
                m2 = self._get_one_mask(((0, 0), (1, 0)), height, width, dtype, device)
                m3 = self._get_one_mask(((0, 0), (0, 1)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def _single_part_for_writing_4x(x: Tensor) -> Tensor:
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    @staticmethod
    def _separate_prior(params: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        q = params[:, :2, :, :]
        q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
        scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    def _process_with_mask(self, y, scales, means, mask):
        return self.gaussian_encoder.process_with_mask(y, scales, means, mask)

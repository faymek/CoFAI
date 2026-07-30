import math

import torch
from compressai.entropy_models import GaussianConditional


class FixedGaussianCodec:
    """
    A simple "model-free" compressor for continuous features.

    This codec does not inherit nn.Module and has no trainable parameters. It assumes
    all input features follow a fixed N(0, scale^2) distribution and uses
    CompressAI's GaussianConditional as an entropy coding utility.
    """

    def __init__(self, fixed_scale=1.0, device="cuda"):
        self.fixed_scale = fixed_scale
        self.device = device

        # Use GaussianConditional as a utility for building CDFs and ANS coding.
        self.gaussian_coder = GaussianConditional(
            [self.fixed_scale],
            scale_bound=fixed_scale + 0.5,
        ).to(device)
        print(
            f"[FixedModelCompressor] 初始化：使用固定的 N(0, {fixed_scale**2:.0f}) 模型。"
        )
        self.gaussian_coder.update()

    def _get_fixed_model_params(self, features):
        means = torch.zeros_like(features)
        scales = torch.full_like(features, self.fixed_scale)
        return means, scales

    def get_feat_and_rate_loss(self, features):
        """[Training] Differentiable proxy of rate loss (bits/element)."""
        means, scales = self._get_fixed_model_params(features)
        _y_hat_proxy, likelihoods = self.gaussian_coder(features, scales, means)

        num_elements = features.numel()
        if num_elements == 0:
            return torch.tensor(0.0, device=self.device)

        r_loss = torch.log(likelihoods).sum() / (-math.log(2) * num_elements)
        return _y_hat_proxy, r_loss

    @torch.no_grad()
    def compress(self, features):
        """[Inference] Entropy-code features."""
        means, scales = self._get_fixed_model_params(features)
        indexes = self.gaussian_coder.build_indexes(scales)
        strings = self.gaussian_coder.compress(features, indexes, means)
        return strings, features.shape

    @torch.no_grad()
    def decompress(self, strings, shape):
        """[Inference] Decode features."""
        dummy_tensor = torch.empty(shape, device=self.device)
        means, scales = self._get_fixed_model_params(dummy_tensor)
        indexes = self.gaussian_coder.build_indexes(scales)
        features_hat = self.gaussian_coder.decompress(
            strings, indexes=indexes, means=means
        )
        return features_hat

    @torch.no_grad()
    def inference(self, features):
        """[Inference] Full compress+decompress flow with actual stream size."""
        strings, shape = self.compress(features)
        total_bytes = sum(len(s) for s in strings)
        total_bits = total_bytes * 8
        features_hat = self.decompress(strings, shape)
        return features_hat, total_bits

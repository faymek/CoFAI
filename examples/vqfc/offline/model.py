import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Any
import torch.nn.functional as F
from abc import abstractmethod
import time
# from torch.cuda.amp import autocast
# from sklearn.mixture import GaussianMixture

from CoFAI.cofai.entropy_models.vqfc_entropy import SoftmaxPrior, DiscreteEntropyModel
import math


class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        lmbda: float,
        logits,
        entropy_model,
    ):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.lmbda = lmbda
        self.logits = logits
        self.uncondi_entropy_model = entropy_model
        self.embedding = nn.Embedding(self.K, self.D)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)
        self.embedding.weight.requires_grad = True

    # @autocast()
    def forward(self, latents: Tensor) -> Tensor:
        # Handle both 3D [N, H, W] and 4D [N, C, H, W] inputs
        original_shape = latents.shape
        if len(latents.shape) == 4:
            N, C, H, W = latents.shape
            # Reshape to [N*C, H, W] for processing
            latents = latents.view(N * C, H, W)
        else:
            N, H, W = latents.shape
            C = 1

        target_rows = H % self.D

        if target_rows == 0:
            latents_expand = latents
        else:
            pad_len = self.D - target_rows
            last_cols = latents[:, -pad_len:, :]
            latents_expand = torch.cat((latents, last_cols), dim=1)

        latents_shape = latents_expand.shape

        latents_expand = (
            latents_expand.permute(0, 2, 1)
            .contiguous()
            .view(
                latents_shape[0], latents_shape[2], latents_shape[1] // self.D, self.D
            )
            .contiguous()
            .view(
                latents_shape[0], latents_shape[2] * latents_shape[1] // self.D, self.D
            )
            .contiguous()
        )
        assert latents_expand.shape[2] == self.D

        quant_codebook = self.embedding.weight

        # prior_param = None, unconditional
        log2_pmf = self.uncondi_entropy_model.log_pmf() / (-math.log(2))
        param_bit = torch.zeros(1).to(latents_expand.device)
        prior_dist = torch.zeros(1).to(latents_expand.device)

        rate_bias = log2_pmf / self.lmbda

        flat_latents = latents_expand.view(-1, self.D)
        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(quant_codebook**2, dim=1)
            - 2 * torch.matmul(flat_latents, quant_codebook.t())
        )

        dist = dist + rate_bias

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BD x 1]
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BD x K]

        quantized_latents = torch.matmul(encoding_one_hot, quant_codebook)  # [BD, HW]

        quantized_latents = quantized_latents.view(
            latents_shape[0], latents_shape[2] * latents_shape[1] // self.D, self.D
        ).contiguous()
        # Compute the mse Losses
        mse_loss = F.mse_loss(quantized_latents, latents_expand)

        quantized_latents = quantized_latents.view(
            latents_shape[0], latents_shape[2], latents_shape[1]
        ).contiguous()
        quantized_latents = quantized_latents.permute(0, 2, 1).contiguous()
        quantized_latents = quantized_latents[:, :H, :]

        # Reshape back to original format if input was 4D
        if len(original_shape) == 4:
            # Reshape back to [N, C, H, W] format
            quantized_latents = quantized_latents.view(N, C, H, W)

        rate_uem = (encoding_one_hot * log2_pmf).sum()
        return (
            quantized_latents,
            mse_loss,
            encoding_inds,
            rate_uem,
            prior_dist,
            param_bit,
        )  # [B x D x H x W]

    def compress(self, latents: Tensor, enc_time_table=None):
        C, H, W = latents.shape
        target_rows = H % self.D
        if target_rows == 0:
            latents_expand = latents
        else:
            pad_len = self.D - target_rows
            last_cols = latents[:, -pad_len:, :]
            latents_expand = torch.cat((latents, last_cols), dim=1)

        latents_shape = latents_expand.shape

        latents_expand = (
            latents_expand.permute(0, 2, 1)
            .contiguous()
            .view(
                latents_shape[0], latents_shape[2], latents_shape[1] // self.D, self.D
            )
            .contiguous()
            .view(
                latents_shape[0], latents_shape[2] * latents_shape[1] // self.D, self.D
            )
            .contiguous()
        )
        assert latents_expand.shape[2] == self.D
        quant_codebook = self.embedding.weight
        flat_latents = latents_expand.view(-1, self.D)
        if enc_time_table is not None:
            torch.cuda.synchronize()
            t00 = time.time()

        log2_pmf = self.uncondi_entropy_model.log_pmf() / (-math.log(2))

        rate_bias = log2_pmf / self.lmbda

        if enc_time_table is not None:
            torch.cuda.synchronize()
            t0 = time.time()
            enc_time_table[1] += t0 - t00

        dist = (
            torch.sum(flat_latents**2, dim=1, keepdim=True)
            + torch.sum(quant_codebook**2, dim=1)
            - 2 * torch.matmul(flat_latents, quant_codebook.t())
        )

        dist = dist + rate_bias

        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BD x 1]

        device = latents_expand.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)

        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BD x K]
        quantized_latents = torch.matmul(encoding_one_hot, quant_codebook)  # [BD, HW]
        quantized_latents = quantized_latents.view(
            latents_shape[0], latents_shape[2] * latents_shape[1] // self.D, self.D
        ).contiguous()
        mse_loss = F.mse_loss(quantized_latents, latents_expand)
        quantized_latents = quantized_latents.view(
            latents_shape[0], latents_shape[2], latents_shape[1]
        ).contiguous()
        quantized_latents = quantized_latents.permute(0, 2, 1).contiguous()

        quantized_latents = quantized_latents[:, :H, :]
        if enc_time_table is not None:
            torch.cuda.synchronize()
            t1 = time.time()
            enc_time_table[2] += t1 - t0

        string = self.uncondi_entropy_model.compress(encoding_inds)
        if enc_time_table is not None:
            torch.cuda.synchronize()
            t2 = time.time()
            enc_time_table[3] += t2 - t1

        return quantized_latents, string, mse_loss, encoding_inds

    def decompress(self, string, latents_shape, dec_time_table=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dec_time_table is not None:
            torch.cuda.synchronize()
            t0 = time.time()
        # 2*1370*1536/dim=420864for dim10 and 211968 for dim20 and 141312 for dim30
        C, H, W = latents_shape
        target_rows = H % self.D

        # 2*1370*1536/dim=420864for dim10 and 211968 for dim20 and 141312 for dim30
        if target_rows == 0:
            pad_len = 0
        else:
            pad_len = self.D - target_rows
        vq_shape = torch.Size([(C * (H + pad_len) * W // self.D), 1])
        latents_shape = torch.randn(C, H + pad_len, W).to(device).shape

        encoding_inds = self.uncondi_entropy_model.decompress(string, vq_shape)
        encoding_inds = encoding_inds.to(device)
        if dec_time_table is not None:
            torch.cuda.synchronize()
            t1 = time.time()
            dec_time_table[1] += t1 - t0

        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)

        # 将 encoding_one_hot 中对应于 encoding_inds 的位置设置为 1，实现 one-hot 编码
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BD x K]
        quant_codebook = self.embedding.weight
        quant_codebook = quant_codebook.to(encoding_one_hot.device)

        quantized_latents = torch.matmul(encoding_one_hot, quant_codebook)  # [BD, HW]
        quantized_latents = quantized_latents.view(
            latents_shape[0], latents_shape[2] * latents_shape[1] // self.D, self.D
        ).contiguous()

        quantized_latents = quantized_latents.view(
            latents_shape[0], latents_shape[2], latents_shape[1]
        ).contiguous()
        quantized_latents = quantized_latents.permute(0, 2, 1).contiguous()
        quantized_latents = quantized_latents[:, :H, :]
        if dec_time_table is not None:
            torch.cuda.synchronize()
            t2 = time.time()
            dec_time_table[2] += t2 - t1

        return quantized_latents


class VQFC(BaseVAE):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_chunks: int,
        lmbda: float,
        **kwargs,
    ) -> None:
        super(VQFC, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.lmbda = lmbda
        self.num_chunks = num_chunks
        self.logits = nn.Parameter(torch.zeros(1, self.num_embeddings))
        self.uncondi_entropy_model = DiscreteEntropyModel(prior=SoftmaxPrior(self.logits))
        self.vq_modules = nn.ModuleList(
            [
                VectorQuantizer(
                    self.num_embeddings,
                    self.embedding_dim,
                    self.lmbda,
                    self.logits,
                    self.uncondi_entropy_model,
                )
                for _ in range(self.num_chunks)
            ]
        )

    # @autocast()
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # Handle different input shapes and convert to [N*C, H, W] format
        original_shape = input.shape
        if len(input.shape) == 2:  # [256, 256] case -> [1, 256, 256]
            input = input.unsqueeze(0)  # [1, 256, 256]
        elif len(input.shape) == 3:  # [N, H, W] case
            pass  # Already in correct format [N, H, W]
        elif len(input.shape) == 4:  # [N, C, H, W] case -> [N*C, H, W]
            N, C, H, W = input.shape
            input = input.view(N * C, H, W)  # [N*C, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

        # Now input is guaranteed to be [N*C, H, W] or [N, H, W]
        # Chunk along the W dimension (dim=2) for [N*C, H, W] format
        input_chunk = torch.chunk(input=input, chunks=self.num_chunks, dim=2)
        quantized_inputs = []
        mse_loss = 0
        encoding_inds = []
        rate_u = []
        prior_vq_dist = []
        prior_vq_rate = []
        numel = input.numel()
        for i, chunk in enumerate(input_chunk):
            vq = self.vq_modules[i]
            quantized, loss, inds, bit_u, prior_dist, param_bit = vq(chunk)
            quantized_inputs.append(quantized)
            mse_loss += loss
            encoding_inds.append(inds)
            rate_u.append(bit_u / numel)
            prior_vq_dist.append(prior_dist / numel)
            prior_vq_rate.append(param_bit / numel)
        # Concatenate quantized inputs along dimension 2 (W dimension)
        quantized_inputs = torch.cat(quantized_inputs, dim=2)

        # Reshape back to original format if needed
        if len(original_shape) == 2:  # [256, 256] case
            quantized_inputs = quantized_inputs.squeeze(0)  # [256, 256]
        elif len(original_shape) == 4:  # [N, C, H, W] case
            N, C, H, W = original_shape
            quantized_inputs = quantized_inputs.view(N, C, H, W)  # [N, C, H, W]

        # Ensure the shape of quantized_inputs matches original input
        assert quantized_inputs.shape == original_shape, (
            f"The shape of quantized_inputs {quantized_inputs.shape} does not match the shape of input {original_shape}"
        )

        # Calculate the average loss
        rate = sum(rate_u)
        mse_loss /= self.num_chunks
        rd_loss = rate + self.lmbda * mse_loss
        return [quantized_inputs, mse_loss, rd_loss, rate, encoding_inds]

    def compress(self, input: Tensor, **kwargs):
        # Handle different input shapes and convert to [N*C, H, W] format
        original_shape = input.shape
        if len(input.shape) == 2:  # [256, 256] case -> [1, 256, 256]
            input = input.unsqueeze(0)  # [1, 256, 256]
        elif len(input.shape) == 3:  # [N, H, W] case
            pass  # Already in correct format [N, H, W]
        elif len(input.shape) == 4:  # [N, C, H, W] case -> [N*C, H, W]
            N, C, H, W = input.shape
            input = input.view(N * C, H, W)  # [N*C, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

        # Now input is guaranteed to be [N*C, H, W] or [N, H, W]
        # Chunk along the W dimension (dim=2) for [N*C, H, W] format
        input_chunk = torch.chunk(input=input, chunks=self.num_chunks, dim=2)
        quantized_inputs = []
        mse_loss = 0
        encoding_inds = []
        strings = []
        for i, chunk in enumerate(input_chunk):
            quantized, string, loss, inds = self.vq_modules[i].compress(chunk)

            quantized_inputs.append(quantized)
            mse_loss += loss
            encoding_inds.append(inds)
            strings.append(string)
        quantized_inputs = torch.cat(quantized_inputs, dim=2)

        # Reshape back to original format if needed
        if len(original_shape) == 2:  # [256, 256] case
            quantized_inputs = quantized_inputs.squeeze(0)  # [256, 256]
        elif len(original_shape) == 4:  # [N, C, H, W] case
            N, C, H, W = original_shape
            quantized_inputs = quantized_inputs.view(N, C, H, W)  # [N, C, H, W]

        # Ensure the shape of quantized_inputs matches original input
        assert quantized_inputs.shape == original_shape, (
            f"The shape of quantized_inputs {quantized_inputs.shape} does not match the shape of input {original_shape}"
        )

        # Calculate the average loss
        mse_loss /= self.num_chunks

        return quantized_inputs, mse_loss, strings, encoding_inds

    def decompress(self, strings, vq_shape):
        quantized_inputs = []
        empt = torch.zeros(vq_shape)
        input_empt = torch.chunk(input=empt, chunks=self.num_chunks, dim=2)
        for i, chunk in enumerate(input_empt):
            # +++bug+++

            chunk = self.vq_modules[0].decompress(strings[i], chunk.shape)
            quantized_inputs.append(chunk)
        quantized_inputs = torch.cat(
            quantized_inputs, dim=2
        )  # Concatenate along W dimension
        return quantized_inputs

    def sample(self, num_samples: int, device) -> Tensor:
        raise Warning

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class RESVQ(BaseVAE):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, num_chunks: int = 2, **kwargs
    ) -> None:
        super(RESVQ, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_chunks = num_chunks
        self.vq_modules = nn.ModuleList(
            [
                VectorQuantizer(self.num_embeddings, self.embedding_dim)
                for _ in range(self.num_chunks)
            ]
        )
        self.res_modules = nn.ModuleList(
            [
                VectorQuantizer(self.num_embeddings, self.embedding_dim)
                for _ in range(self.num_chunks)
            ]
        )

    # @autocast()
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # encoding = self.encode(input)[0]

        input_chunk = torch.chunk(input=input, chunks=self.num_chunks, dim=1)

        quantized_inputs = []
        mse_loss = 0
        encoding_inds = []
        for i, chunk in enumerate(input_chunk):
            vq = self.vq_modules[i]
            quantized, loss, inds = vq(chunk)
            quantized_inputs.append(quantized)
            mse_loss += loss
            encoding_inds.append(inds)

        # Concatenate quantized inputs along dimension 1
        quantized_inputs = torch.cat(quantized_inputs, dim=1)
        # Ensure the shape of quantized_inputs is the same as input
        assert quantized_inputs.shape == input.shape, (
            "The shape of quantized_inputs does not match the shape of input"
        )

        # Calculate the average loss
        mse_loss /= self.num_chunks

        return [quantized_inputs, mse_loss, encoding_inds]

    def sample(self, num_samples: int, device) -> Tensor:
        raise Warning

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

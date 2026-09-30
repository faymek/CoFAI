"""VQ-UFC quantizer and token transforms, separate from its codec adapter."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cofai.entropy_models.vqfc_entropy import DiscreteEntropyModel, SoftmaxPrior


class SequenceAnalysisTransform(nn.Module):
    """Blockwise linear token analysis: B x L x C -> B x (ceil(L/K) * N) x C."""

    def __init__(self, block_tokens=256, latent_tokens=128):
        super().__init__()
        self.block_tokens = int(block_tokens)
        self.latent_tokens = int(latent_tokens)
        self.proj = nn.Linear(self.block_tokens, self.latent_tokens)

    def forward(self, x):
        b, l, c = x.shape
        pad_len = (-l) % self.block_tokens
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        padded_l = x.shape[1]
        num_blocks = padded_l // self.block_tokens
        x = x.view(b, num_blocks, self.block_tokens, c)
        x = x.permute(0, 1, 3, 2).reshape(b * num_blocks, c, self.block_tokens)
        y = self.proj(x)
        y = y.reshape(b, num_blocks, c, self.latent_tokens).permute(0, 1, 3, 2)
        return y.reshape(b, num_blocks * self.latent_tokens, c).contiguous()


class SequenceSynthesisTransform(nn.Module):
    """Blockwise linear token synthesis: B x (M * N) x C -> B x L x C."""

    def __init__(self, latent_tokens=128, block_tokens=256):
        super().__init__()
        self.latent_tokens = int(latent_tokens)
        self.block_tokens = int(block_tokens)
        self.proj = nn.Linear(self.latent_tokens, self.block_tokens)

    def forward(self, x, target_length):
        b, l, c = x.shape
        if l % self.latent_tokens != 0:
            raise ValueError(
                f"Blockwise token synthesis expects latent length to be divisible by N={self.latent_tokens}, "
                f"got input shape {tuple(x.shape)}"
            )
        num_blocks = l // self.latent_tokens
        y = x.view(b, num_blocks, self.latent_tokens, c)
        y = y.permute(0, 1, 3, 2).reshape(b * num_blocks, c, self.latent_tokens)
        y = self.proj(y)
        y = y.reshape(b, num_blocks, c, self.block_tokens).permute(0, 1, 3, 2)
        y = y.reshape(b, num_blocks * self.block_tokens, c)
        return y[:, :target_length, :].contiguous()


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        lmbda,
        entropy_model,
        use_soft_assignment=False,
        temperature=1.0,
    ):
        super().__init__()
        self.K = int(num_embeddings)
        self.D = int(embedding_dim)
        self.lmbda = float(lmbda)
        self.entropy_model = entropy_model
        self.use_soft_assignment = bool(use_soft_assignment)
        self.temperature = float(temperature)
        self.embedding = nn.Embedding(self.K, self.D)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)

    def adapted_codebook(self):
        return self.embedding.weight

    def _costs(self, flat_latents):
        # Feature extraction/evaluation may run under autocast.  Keep the VQ
        # distance calculation in float32 so un-clipped normalized tails do not
        # overflow fp16 and change the selected codeword.
        flat_latents = flat_latents.float()
        codebook = self.adapted_codebook().float()
        log2_pmf = self.entropy_model.log_pmf().to(device=flat_latents.device, dtype=torch.float32).reshape(-1) / (
            -math.log(2)
        )
        distance = (
            flat_latents.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2 * flat_latents @ codebook.t()
        )
        cost = distance + log2_pmf.unsqueeze(0) / self.lmbda
        return cost, log2_pmf

    def _indices(self, flat_latents):
        cost, log2_pmf = self._costs(flat_latents)
        return cost.argmin(dim=1, keepdim=True).long(), log2_pmf

    def _soft_quantize(self, flat_latents):
        cost, log2_pmf = self._costs(flat_latents)
        soft_assignment = F.softmax(-cost / self.temperature, dim=-1)
        avg_probs = soft_assignment.mean(dim=0)
        hard_indices = soft_assignment.argmax(dim=-1)
        hard_assignment = F.one_hot(hard_indices, num_classes=self.K).to(soft_assignment.dtype)

        # Forward uses one codeword; backward follows the soft probabilities.
        assignment = hard_assignment + soft_assignment - soft_assignment.detach()
        quantized = assignment @ self.adapted_codebook()
        soft_rate_bits = (soft_assignment * log2_pmf.unsqueeze(0)).sum()
        hard_rate_bits = log2_pmf[hard_indices].sum()
        return (
            quantized,
            hard_indices.unsqueeze(1),
            avg_probs,
            soft_rate_bits,
            hard_rate_bits,
        )

    def forward(self, latents):
        if latents.shape[-1] != self.D:
            raise ValueError(f"Expected VQ vectors with D={self.D}, got {tuple(latents.shape)}")
        original_shape = latents.shape
        flat_latents = latents.reshape(-1, self.D)
        avg_probs = None
        if self.use_soft_assignment:
            quantized_flat, indices, avg_probs, rate_bits, hard_rate_bits = self._soft_quantize(flat_latents)
            quantized_raw = quantized_flat.view(original_shape)
            quantized = quantized_raw
        else:
            indices, log2_pmf = self._indices(flat_latents)
            quantized_raw = F.embedding(indices.squeeze(1), self.adapted_codebook()).view(original_shape)
            # Keep STE only for training.  Evaluation uses the exact codebook
            # values so forward() matches the real rANS decode path bit-for-bit.
            quantized = latents + (quantized_raw - latents).detach() if self.training else quantized_raw
            rate_bits = log2_pmf[indices.squeeze(1)].sum()
            hard_rate_bits = rate_bits
        mse = F.mse_loss(quantized_raw, latents)
        commitment_loss = F.mse_loss(latents, quantized_raw.detach())
        return quantized, mse, commitment_loss, indices, rate_bits, hard_rate_bits, avg_probs

    def compress(self, latents):
        original_shape = latents.shape
        flat_latents = latents.reshape(-1, self.D)
        indices, _ = self._indices(flat_latents)
        quantized = F.embedding(indices.squeeze(1), self.adapted_codebook()).view(original_shape)
        mse = F.mse_loss(quantized, latents)
        string = self.entropy_model.compress(indices)
        return quantized, string, mse, indices

    def decompress(self, string, vector_shape):
        vector_shape = torch.Size(vector_shape)
        num_vectors = int(math.prod(vector_shape[:-1]))
        indices = self.entropy_model.decompress(string, torch.Size([num_vectors, 1]))
        indices = indices.to(self.embedding.weight.device).long().reshape(-1)
        quantized = F.embedding(indices, self.adapted_codebook()).view(vector_shape)
        return quantized


class VQUFCModel(nn.Module):
    """Single-level FCVQ with optional channel transform.

    For B x L x C input, L is padded to a multiple of embedding_dim and
    consecutive values along L form each VQ vector.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        num_chunks=1,
        lmbda=1.0,
        vector_mode="legacy_sequence",
        **kwargs,
    ):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.num_chunks = int(num_chunks)
        self.lmbda = float(lmbda)
        self.commit_weight = float(kwargs.get("commit_weight", 0.25))
        self.use_soft_assignment = bool(kwargs.get("use_soft_assignment", False))
        self.soft_rate_weight = float(kwargs.get("soft_rate_weight", 1.0))
        self.usage_weight = float(kwargs.get("usage_weight", 0.0))
        self.soft_temperature_initial = float(kwargs.get("soft_temperature", 1.0))
        self.soft_temperature_min = float(kwargs.get("soft_temperature_min", 0.1))
        self.soft_temperature_decay = float(kwargs.get("soft_temperature_decay", 0.95))
        self.soft_temperature = self.soft_temperature_initial
        self.vector_mode = vector_mode
        self.use_transform = bool(kwargs.get("use_transform", False))
        self.transform_input_tokens = int(kwargs.get("transform_input_tokens", 256))
        requested_tokens = int(kwargs.get("transform_tokens", 0))
        self.transform_tokens = requested_tokens or ((self.transform_input_tokens + 1) // 2)

        if self.vector_mode != "legacy_sequence":
            raise ValueError("VQ-UFC only supports legacy_sequence vector layout")
        self.analysis_transform = (
            SequenceAnalysisTransform(self.transform_input_tokens, self.transform_tokens)
            if self.use_transform
            else nn.Identity()
        )
        self.synthesis_transform = (
            SequenceSynthesisTransform(self.transform_tokens, self.transform_input_tokens)
            if self.use_transform
            else nn.Identity()
        )

        self.base_logits = nn.Parameter(torch.zeros(1, self.num_embeddings))
        self.entropy_model = DiscreteEntropyModel(prior=SoftmaxPrior(self.base_logits))
        self.vq = VectorQuantizer(
            self.num_embeddings,
            self.embedding_dim,
            self.lmbda,
            self.entropy_model,
            use_soft_assignment=self.use_soft_assignment,
            temperature=self.soft_temperature,
        )
    @staticmethod
    def _pad_by_repeating_tail(x, dim, pad_len):
        if pad_len == 0:
            return x
        size = x.shape[dim]
        start = -min(size, pad_len)
        indices = torch.arange(start, start + pad_len, device=x.device) % size
        return torch.cat((x, x.index_select(dim, indices)), dim=dim)

    def _split_vectors(self, x):
        b, l, c = x.shape
        d = self.embedding_dim
        pad_len = (-l) % d
        padded = self._pad_by_repeating_tail(x, dim=1, pad_len=pad_len)
        padded_l = padded.shape[1]
        vectors = padded.permute(0, 2, 1).contiguous().view(b, c, padded_l // d, d).reshape(-1, d)
        return vectors, {
            "mode": self.vector_mode,
            "b": b,
            "l": l,
            "c": c,
            "padded_l": padded_l,
        }

    def _merge_vectors(self, vectors, ctx):
        b, l, c = ctx["b"], ctx["l"], ctx["c"]
        padded_l = ctx["padded_l"]
        return (
            vectors.view(b, c, padded_l // self.embedding_dim, self.embedding_dim)
            .reshape(b, c, padded_l)[:, :, :l]
            .permute(0, 2, 1)
            .contiguous()
        )

    def _prepare(self, input_tensor):
        if input_tensor.ndim != 3:
            raise ValueError(f"Expected B x L x C input, got {tuple(input_tensor.shape)}")
        original_shape = input_tensor.shape
        transformed = self.analysis_transform(input_tensor.contiguous())
        vectors, ctx = self._split_vectors(transformed)
        chunks = [chunk for chunk in torch.chunk(vectors, chunks=self.num_chunks, dim=0) if chunk.numel() > 0]
        return chunks, ctx, original_shape

    def _restore(self, blocks, ctx, original_shape):
        transformed = self._merge_vectors(torch.cat(blocks, dim=0), ctx)
        output = (
            self.synthesis_transform(transformed, target_length=original_shape[1])
            if self.use_transform else transformed
        )
        return output.view(original_shape)

    def load_compatible_checkpoint(self, checkpoint_path, map_location="cpu", checkpoint=None):
        if checkpoint is None:
            try:
                # Training artifacts may include NumPy metadata (PyTorch 2.6+).
                checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state = checkpoint.get("vqvae_state_dict", checkpoint.get("state_dict", checkpoint))
        missing, unexpected = self.load_state_dict(state, strict=False)
        checkpoint_use_transform = bool((checkpoint.get("model_config") or {}).get("use_transform", False))
        allowed_missing = {
            key
            for key in missing
            if (
                key.startswith("entropy_model.")
                or key.startswith("vq.entropy_model.")
                or (
                    not checkpoint_use_transform
                    and (key.startswith("analysis_transform.") or key.startswith("synthesis_transform."))
                )
            )
        }
        real_missing = [key for key in missing if key not in allowed_missing]
        if real_missing:
            raise RuntimeError(f"Missing checkpoint parameters: {real_missing}")
        print(f"Loaded current FCVQ checkpoint; ignored keys: {unexpected}")
        return checkpoint

    def set_codec_trainability(self, freeze_codebook=True, freeze_entropy=True):
        """Configure whether the pretrained codebook and entropy prior are trainable."""
        self.freeze_codebook = bool(freeze_codebook)
        self.freeze_entropy = bool(freeze_entropy)
        self.vq.embedding.weight.requires_grad_(not self.freeze_codebook)
        self.base_logits.requires_grad_(not self.freeze_entropy)
        for parameter in self.entropy_model.parameters():
            parameter.requires_grad_(not self.freeze_entropy)
        print(
            "Codec trainability: "
            f"codebook={'frozen' if self.freeze_codebook else 'trainable'}, "
            f"entropy/logits={'frozen' if self.freeze_entropy else 'trainable'}."
        )

    def prepare_entropy_model_for_compression(self):
        self.entropy_model.get_ready_for_compression()

    def update_soft_temperature(self, epoch):
        if not self.use_soft_assignment:
            return self.soft_temperature
        self.soft_temperature = max(
            self.soft_temperature_min,
            self.soft_temperature_initial * (self.soft_temperature_decay ** max(int(epoch) - 1, 0)),
        )
        self.vq.temperature = self.soft_temperature
        return self.soft_temperature

    def forward(self, input_tensor, **kwargs):
        chunks, ctx, original_shape = self._prepare(input_tensor)
        blocks, indices = [], []
        mse = input_tensor.new_zeros(())
        commitment_loss = input_tensor.new_zeros(())
        rate = input_tensor.new_zeros(())
        hard_rate = input_tensor.new_zeros(())
        avg_probs_sum = None
        total_vectors = sum(chunk.shape[0] for chunk in chunks)
        total_values = sum(chunk.numel() for chunk in chunks)
        for chunk in chunks:
            (
                quantized,
                chunk_mse,
                chunk_commitment,
                chunk_indices,
                chunk_bits,
                chunk_hard_bits,
                chunk_avg_probs,
            ) = self.vq(chunk)
            blocks.append(quantized)
            indices.append(chunk_indices)
            chunk_weight = chunk.shape[0] / total_vectors
            mse = mse + chunk_mse * chunk_weight
            commitment_loss = commitment_loss + chunk_commitment * chunk_weight
            rate = rate + chunk_bits / total_values
            hard_rate = hard_rate + chunk_hard_bits / total_values
            if chunk_avg_probs is not None:
                weighted_probs = chunk_avg_probs * chunk_weight
                avg_probs_sum = weighted_probs if avg_probs_sum is None else avg_probs_sum + weighted_probs
        output = self._restore(blocks, ctx, original_shape)
        recon_mse = F.mse_loss(output, input_tensor)
        if avg_probs_sum is None:
            usage_entropy = input_tensor.new_zeros(())
            usage_loss = input_tensor.new_zeros(())
            usage_ppl = input_tensor.new_ones(())
        else:
            eps = torch.finfo(avg_probs_sum.dtype).eps
            usage_entropy = -(avg_probs_sum * (avg_probs_sum + eps).log()).sum()
            # Minimize this negative entropy to discourage single-codeword collapse.
            usage_loss = -usage_entropy
            usage_ppl = usage_entropy.exp()
        self.last_usage_loss = usage_loss.detach()
        self.last_usage_entropy = usage_entropy.detach()
        self.last_usage_ppl = usage_ppl.detach()
        rd_loss = (
            self.soft_rate_weight * rate
            + self.lmbda * recon_mse
            + self.commit_weight * commitment_loss
            + self.usage_weight * usage_loss
        )
        return (
            output,
            recon_mse,
            mse,
            commitment_loss,
            rd_loss,
            rate,
            hard_rate,
            indices,
        )

    def compress(self, input_tensor, **kwargs):
        chunks, ctx, original_shape = self._prepare(input_tensor)
        blocks, strings, indices = [], [], []
        mse = input_tensor.new_zeros(())
        total_vectors = sum(chunk.shape[0] for chunk in chunks)
        for chunk in chunks:
            quantized, string, chunk_mse, chunk_indices = self.vq.compress(chunk)
            blocks.append(quantized)
            strings.append(string)
            indices.append(chunk_indices)
            mse = mse + chunk_mse * (chunk.shape[0] / total_vectors)
        output = self._restore(blocks, ctx, original_shape)
        shape_info = {
            "original_shape": tuple(original_shape),
            "ctx": ctx,
            "chunk_shapes": [tuple(chunk.shape) for chunk in chunks],
        }
        return output, mse, strings, indices, shape_info

    def decompress(self, strings, shape_info, **kwargs):
        blocks = [
            self.vq.decompress(string, chunk_shape)
            for string, chunk_shape in zip(strings, shape_info["chunk_shapes"])
        ]
        return self._restore(blocks, shape_info["ctx"], torch.Size(shape_info["original_shape"]))

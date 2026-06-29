import torch
import torch.nn as nn

from cofai.entropy_models.vqfc_model import VQFC


class VQFeatureCodec(nn.Module):
    """Adapt :class:`~cofai.entropy_models.vqfc_model.VQFC` to the latent-codec API.

    ``DinoFeatureCodecModel`` / ``DinoSlideFeatureCodecModel`` expect a codec that
    exposes ``forward(h, token_res, qp) -> {"h_hat", ...}``,
    ``compress(h, token_res, qp) -> {"strings", "pstate"}`` and
    ``decompress(strings, pstate) -> {"h_hat"}`` operating on encoder token tensors
    of shape ``(B, N, C)``. The raw ``VQFC`` model instead works on patch-token
    tensors and returns ``(quantized, mse, strings, inds)``.

    This thin wrapper bridges the two. The number of prefix (cls / register)
    tokens is declared explicitly via ``num_prefix_tokens`` instead of being
    inferred from ``token_res``.

    When ``drop_prefix=True`` the leading ``num_prefix_tokens`` tokens are split
    off and excluded from compression; VQFC runs on the patch tokens only and the
    prefix is re-attached so the result can flow back through
    ``backbone.decode_seg``. Because segmentation decoding drops prefix tokens,
    the reconstructed prefix values are irrelevant; the ``forward`` path keeps the
    original prefix tokens while ``decompress`` (which has no access to them) uses
    zeros, matching ``Dinov2OrigSlideSegVQFC``.

    When ``drop_prefix=False`` (default) the whole token sequence (prefix
    included) is compressed and reconstructed, so the cls token survives the
    round-trip. This is required for classification (``backbone.decode_cls``) and
    matches the rate of the legacy ``Dinov2OrigClsVQFC`` (which compresses the
    full ``h_dino``).

    Args:
        num_prefix_tokens (int): Number of leading prefix tokens (cls + register).
            Only used when ``drop_prefix=True``. Defaults to ``0``.
        drop_prefix (bool): If ``True``, exclude the leading ``num_prefix_tokens``
            tokens from compression (segmentation behavior). If ``False``
            (default), compress every token including the prefix.
        **vqfc_cfg: Keyword arguments forwarded to :class:`VQFC` (e.g.
            ``num_embeddings``, ``embedding_dim``, ``num_chunks``, ``lmbda``,
            ``ckpt_path``).
    """

    def __init__(
        self,
        num_prefix_tokens: int = 0,
        drop_prefix: bool = False,
        **vqfc_cfg,
    ):
        super().__init__()
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.drop_prefix = bool(drop_prefix)
        self.vqfc = VQFC(**vqfc_cfg)
        if hasattr(self.vqfc, "uncondi_entropy_model"):
            self.vqfc.uncondi_entropy_model.get_ready_for_compression()

    def _num_prefix(self) -> int:
        return self.num_prefix_tokens if self.drop_prefix else 0

    def forward(self, h, token_res=None, qp=0, **kwargs):
        n_prefix = self._num_prefix()
        prefix, patch = h[:, :n_prefix, :], h[:, n_prefix:, :]
        patch_hat, _mse, strings, _inds = self.vqfc.compress(patch)
        h_hat = torch.cat([prefix, patch_hat], dim=1)
        return {
            "h_hat": h_hat,
            "strings": {"vqfc": [[s] for s in strings]},
        }

    def compress(self, h, token_res=None, qp=0, **kwargs):
        n_prefix = self._num_prefix()
        patch = h[:, n_prefix:, :]
        patch_hat, _mse, strings, _inds = self.vqfc.compress(patch)
        return {
            "strings": {"vqfc": [[s] for s in strings]},
            "pstate": {
                "patch_shape": tuple(patch.shape),
                "n_prefix": int(n_prefix),
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        chunk_strings = [s[0] for s in strings["vqfc"]]
        patch_shape = tuple(pstate["patch_shape"])
        patch_hat = self.vqfc.decompress(chunk_strings, patch_shape)
        device = next(self.parameters()).device
        patch_hat = patch_hat.to(device)

        n_prefix = int(pstate["n_prefix"])
        prefix = torch.zeros(
            patch_shape[0],
            n_prefix,
            patch_shape[2],
            device=device,
            dtype=patch_hat.dtype,
        )
        h_hat = torch.cat([prefix, patch_hat], dim=1)
        return {"h_hat": h_hat}

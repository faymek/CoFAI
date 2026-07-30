import torch
from compressai.models.base import CompressionModel

from cofai.utils.coder import encode_uniform_to_bits, decode_uniform_from_bits


class UniformIndexCodec(CompressionModel):
    """Encode discrete indices under a uniform alphabet model.

    This codec assumes a uniform distribution over the alphabet and encodes
    integer indices using the repository's uniform arithmetic-coding helpers.
    """

    def __init__(self, alphabet_size, **kwargs):
        """Initialize the uniform index codec.

        Args:
            alphabet_size (int): Size of the token alphabet (number of possible values).
            **kwargs (dict): Additional keyword arguments passed to parent class.
        """
        super().__init__()
        self.alphabet_size = alphabet_size

    def forward(self, tokens):
        """Forward pass to compute uniform likelihoods.

        Args:
            tokens (torch.Tensor): Input tokens of any shape.

        Returns:
            output (dict): Dictionary containing:
                - "likelihoods" (dict): Dictionary with key "t" containing uniform
                  likelihoods of shape matching tokens.
                - "tokens" (torch.Tensor): Original input tokens.
        """
        return {
            "likelihoods": {"t": self._uniform_likelihood(tokens)},
            "tokens": tokens,
        }

    def _uniform_likelihood(self, tokens):
        """Compute uniform likelihoods for tokens.

        Args:
            tokens (torch.Tensor): Input tokens of any shape.

        Returns:
            likelihoods (torch.Tensor): Uniform likelihoods of shape matching tokens,
                where each element is 1.0 / alphabet_size.
        """
        likelihoods = torch.ones(tokens.shape) * (1.0 / self.alphabet_size)
        likelihoods = likelihoods.to(tokens.device)
        return likelihoods

    def compress(self, tokens):
        """Compress tokens to bitstring.

        Note: tokens should not have batch dimension.

        Args:
            tokens (torch.Tensor): Input tokens to compress. Shape should be
                (H, W, ...) without batch dimension.

        Returns:
            coded_unit (dict): Dictionary containing:

                - "strings" (dict): Dictionary with key "t" containing nested list
                  with encoded bitstring. Nested structure is for consistent API.
                - "pstate" (dict): Dictionary with key "t_shape" containing the
                  original shape of tokens as a tuple.
        """
        alphabet_size = self.alphabet_size
        string = encode_uniform_to_bits(tokens.flatten(), alphabet_size)
        return {
            "strings": {"t": [[string]]},  # Nested structure for consistent API
            "pstate": {"t_shape": tuple(tokens.shape)},
        }

    def decompress(self, strings, pstate, **kwargs):
        """Decompress bitstring to tokens.

        Args:
            strings (dict): Dictionary with key "t" containing nested list with
                encoded bitstring. Nested structure is for consistent API.
            pstate (dict): Dictionary with key "t_shape" containing the original
                shape of tokens as a tuple.
            **kwargs (dict): Additional keyword arguments (unused).

        Returns:
            task_feats (dict): Dictionary with key "tokens" containing decompressed
                tokens of shape specified in pstate["t_shape"]. Note: tokens
                do not have batch dimension.
        """
        # Extract strings from nested structure for consistent API
        _strings = strings["t"][0][0]
        _shape = pstate["t_shape"]
        symbols_len = 1
        # Compute total number of symbols (no batch dimension)
        for dim in _shape:
            symbols_len *= dim
        alphabet_size = self.alphabet_size
        tokens = decode_uniform_from_bits(_strings, symbols_len, alphabet_size)
        tokens = tokens.reshape(_shape).long().cuda()
        return {"tokens": tokens}

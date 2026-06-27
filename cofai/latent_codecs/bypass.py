import torch.nn as nn


class BypassLatentCodec(nn.Module):
    """Pass-through codec used as the no-compression anchor.

    The reconstructed features equal the input features and the reported bitstream is
    empty, so the measured rate is zero. Plug it into ``DinoFeatureCodecModel`` to
    benchmark task quality on uncompressed backbone features.

    Args:
        **kwargs (dict): Additional keyword arguments (currently unused).
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, h, token_res=None, qp=0, **kwargs):
        return {"h_hat": h, "likelihoods": {"bypass": h.new_ones(1)}}

    def compress(self, h, token_res=None, qp=0, **kwargs):
        return {
            "strings": {"bypass": [[b""]]},
            "pstate": {"token_res": token_res, "h_hat": h},
        }

    def decompress(self, strings, pstate, **kwargs):
        return {"h_hat": pstate["h_hat"]}

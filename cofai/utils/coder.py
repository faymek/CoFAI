import torch
import torchac


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device="cpu")
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.0)
    return cdf_with_0


def get_uniform_pmf(batch_size, alphabet_size):
    p = 1.0 / alphabet_size
    return torch.full((batch_size, alphabet_size), p, dtype=torch.float32)



def encode_uniform_to_bits(indices, alphabet_size):
    """
    将均匀分布的索引编码为比特字符串
    
    Args:
        indices (torch.Tensor): 要编码的索引张量
        alphabet_size (int): 符号表大小
        
    Returns:
        bytes: 编码后的比特字符串
    """
    # 创建均匀分布的CDF
    batch_size = indices.shape[0]
    pmf = get_uniform_pmf(batch_size, alphabet_size)
    cdf = pmf_to_cdf(pmf)
    
    # 使用算术编码
    bitstream = torchac.encode_float_cdf(
        cdf_float=cdf,
        sym=indices.to(dtype=torch.int16).cpu(),
        check_input_bounds=True,
    )
    
    return bitstream


def decode_uniform_from_bits(bitstream, batch_size, alphabet_size):
    """
    从比特字符串解码均匀分布的索引
    
    Args:
        bitstream (bytes): 编码后的比特字符串
        alphabet_size (int): 符号表大小
        num_symbols (int): 要解码的符号数量
        
    Returns:
        torch.Tensor: 解码后的索引张量
    """
    # 创建均匀分布的CDF
    pmf = get_uniform_pmf(batch_size, alphabet_size)
    cdf = pmf_to_cdf(pmf)
    
    # 使用算术解码
    indices = torchac.decode_float_cdf(cdf, bitstream)
    
    return indices


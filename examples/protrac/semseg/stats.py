import torch


def load_stats(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }

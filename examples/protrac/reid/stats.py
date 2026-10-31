import torch
from torch.nn import functional as F


def load_stats(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    prototypes = F.normalize(payload["semantic_prototypes"].to(device).float(), dim=1, eps=1e-5)
    cells = {
        view: {key: value.to(device).float() for key, value in data.items()} for view, data in payload["cells"].items()
    }
    return prototypes, cells

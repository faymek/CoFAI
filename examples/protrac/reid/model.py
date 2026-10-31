import torch

from cofai.backbone.vdt import VDTBackbone
from examples.protrac.calibration import (
    ProTRACStats,
    clip_delta,
    projection_delta,
    semantic_weights,
)

from .stats import load_stats


class ProTRACReIDBackbone(VDTBackbone):
    def __init__(self, cfg, head, device):
        super().__init__(image_size=tuple(cfg.image_size), slot=cfg.slot)
        self.head = head
        self.cfg = cfg.calibration
        self.prototypes, self.cells = load_stats(cfg.stats, device)
        self.trackers = {view: ProTRACStats.create(data["hidden_mean"]) for view, data in self.cells.items()}
        self.pending = None

    def reset(self):
        for tracker in self.trackers.values():
            tracker.reset()
        self.pending = None

    @torch.no_grad()
    def encode(self, image, view=None, calibrate=True, **kwargs):
        tokens = super().encode(image)
        if not calibrate:
            return tokens
        cfg = self.cfg
        raw = self.head(self.final_tokens(tokens))
        assignments = semantic_weights(raw, self.prototypes, cfg.topm, cfg.temperature, eps=1e-5)
        cells = self.cells[view]
        assignments = assignments * (cells["support"] > 0).float()
        assignments = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-5)
        hidden = tokens[:, 1].float()
        global_shift = self.trackers[view].shift(assignments)
        projection = projection_delta(
            (hidden + global_shift).unsqueeze(1), cells["hidden_mean"], cells["hidden_basis"][:, :, : cfg.rank]
        )
        projection = (assignments.unsqueeze(-1) * projection).sum(dim=1)
        delta = global_shift + cfg.projection_weight * projection
        edited = tokens.clone()
        edited[:, 1] = (hidden + clip_delta(delta, hidden, cfg.clip, eps=1e-5)).to(tokens.dtype)
        self.pending = (view, hidden, assignments)
        return edited

    @torch.no_grad()
    def update(self):
        view, hidden, assignments = self.pending
        self.trackers[view].update(hidden, assignments, eps=1e-5)
        self.pending = None

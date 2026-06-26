"""Depth estimation meters."""

from __future__ import annotations

import numpy as np
import torch

from cofai.backbone.dinov3.eval.depth.datasets.datasets_utils import (
    _EvalCropType,
    make_valid_mask,
)
from cofai.backbone.dinov3.eval.depth.metrics import (
    DEPTH_METRICS,
    calculate_depth_metrics,
)


class DepthEstimationMeter:
    def __init__(self, max_depth: float | None = None, min_depth: float | None = None):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.abs_rel = 0.0
        self.sq_rel = 0.0

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred, gt = pred.squeeze(), gt.squeeze()
        if self.max_depth is None or self.min_depth is None:
            raise ValueError("DepthEstimationMeter requires max_depth and min_depth")

        mask = torch.logical_and(gt < self.max_depth, gt > self.min_depth)
        self.n_valid += float(mask.float().sum().item())

        gt = gt.clone()
        pred = pred.clone()
        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        log_rmse_tmp = torch.pow(torch.log(gt[mask]) - torch.log(pred[mask]), 2)
        self.total_log_rmses += float(log_rmse_tmp.sum().item())

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += float(rmse_tmp.sum().item())

        self.abs_rel += float((torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item())
        self.sq_rel += float((((gt[mask] - pred[mask]) ** 2) / gt[mask]).sum().item())

    def compute(self) -> dict[str, float]:
        if self.n_valid <= 0:
            return {"rmse": 0.0, "log_rmse": 0.0, "abs_rel": 0.0, "sq_rel": 0.0}
        return {
            "rmse": float(np.sqrt(self.total_rmses / self.n_valid)),
            "log_rmse": float(np.sqrt(self.total_log_rmses / self.n_valid)),
            "abs_rel": float(self.abs_rel / self.n_valid),
            "sq_rel": float(self.sq_rel / self.n_valid),
        }


class DepthEstimationMeterLegacy:
    def __init__(self, ignore_index: int = 255):
        self.total_rmses = 0.0
        self.total_log_rmses = 0.0
        self.n_valid = 0.0
        self.ignore_index = int(ignore_index)
        self.abs_rel = 0.0
        self.sq_rel = 0.0

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred, gt = pred.squeeze(), gt.squeeze()
        mask = (gt != self.ignore_index).bool()
        self.n_valid += float(mask.float().sum().item())

        gt = gt.clone()
        pred = pred.clone()
        gt[gt <= 0] = 1e-9
        pred[pred <= 0] = 1e-9

        log_rmse_tmp = torch.pow(torch.log(gt[mask]) - torch.log(pred[mask]), 2)
        self.total_log_rmses += float(log_rmse_tmp.sum().item())

        rmse_tmp = torch.pow(gt[mask] - pred[mask], 2)
        self.total_rmses += float(rmse_tmp.sum().item())

        self.abs_rel += float((torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item())
        self.sq_rel += float((((gt[mask] - pred[mask]) ** 2) / gt[mask]).sum().item())

    def compute(self) -> dict[str, float]:
        if self.n_valid <= 0:
            return {"rmse": 0.0, "log_rmse": 0.0, "abs_rel": 0.0, "sq_rel": 0.0}
        return {
            "rmse": float(np.sqrt(self.total_rmses / self.n_valid)),
            "log_rmse": float(np.sqrt(self.total_log_rmses / self.n_valid)),
            "abs_rel": float(self.abs_rel / self.n_valid),
            "sq_rel": float(self.sq_rel / self.n_valid),
        }

class Dinov3DepthEstimationMeter:
    """DINOv3 depth metrics with crop mask and configurable metric names."""

    def __init__(
        self,
        names=None,
        min_depth: float = 0.001,
        max_depth: float = 10.0,
        ignored_value: float = 0.0,
        eval_mask: str = "FULL",
        normalization_constant: float = 1000.0,
        **kwargs,
    ):
        self.names = list(names) if names is not None else ["rmse", "abs_rel", "a1"]
        valid_names = {metric.name for metric in DEPTH_METRICS}
        unknown = [name for name in self.names if name not in valid_names]
        if unknown:
            raise ValueError(
                f"Unknown depth metric names: {unknown}. "
                f"Available names: {sorted(valid_names)}"
            )

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.ignored_value = float(ignored_value)
        self.eval_mask = str(eval_mask)
        self.normalization_constant = float(normalization_constant)
        self.reset()

    def reset(self) -> None:
        self.totals = {name: 0.0 for name in self.names}
        self.count = 0

    @torch.no_grad()
    def update(self, pred, gt) -> None:
        pred = torch.as_tensor(pred).float()
        gt = torch.as_tensor(gt, device=pred.device).float()

        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(0)

        if gt.dim() == 2:
            gt = gt.unsqueeze(0).unsqueeze(0)
        elif gt.dim() == 3:
            gt = gt.unsqueeze(0)

        gt = gt / self.normalization_constant
        pred = pred.clamp(min=self.min_depth, max=self.max_depth)

        ignored = torch.tensor(
            self.ignored_value,
            device=gt.device,
            dtype=gt.dtype,
        )
        gt_for_eval = torch.where(
            torch.logical_or(gt >= self.max_depth, gt <= self.min_depth),
            ignored,
            gt,
        )
        valid_mask = make_valid_mask(
            gt_for_eval,
            eval_crop=_EvalCropType(self.eval_mask),
            ignored_value=self.ignored_value,
        )
        metrics_spec = [metric for metric in DEPTH_METRICS if metric.name in self.names]
        depth_metrics = calculate_depth_metrics(
            gt_for_eval,
            pred,
            valid_mask,
            list_metrics=metrics_spec,
        )
        for name in self.names:
            self.totals[name] += float(getattr(depth_metrics, name))
        self.count += 1

    def compute(self) -> dict[str, float]:
        if self.count <= 0:
            return {name: 0.0 for name in self.names}
        return {name: float(value / self.count) for name, value in self.totals.items()}


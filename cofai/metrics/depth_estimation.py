"""Depth estimation meters."""

from __future__ import annotations

import math
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

# The following depth-evaluation helpers are copied from DINOv3
# (eval/depth/datasets/datasets_utils.py and eval/depth/metrics.py)
# Copyright (c) Meta Platforms, Inc. and affiliates. Used under the DINOv3 License Agreement.


class _EvalCropType(Enum):
    NYU_EIGEN = "NYU_EIGEN"
    FULL = "FULL"


def make_valid_mask(
    input, eval_crop: _EvalCropType = _EvalCropType.FULL, ignored_value: float = 0.0
):
    """Following Adabins, do garg_crop or eigen_crop for testing.

    Args:
        input: input tensor in BxCxHxW format
        eval_crop (_EvalCropType): evaluation crop used for evaluation
        ignored_value (float): value from input to be ignored during evaluation
    """
    B, _, h, w = input.shape
    eval_mask = torch.zeros(input.shape, device=input.device)
    if eval_crop == _EvalCropType.NYU_EIGEN:
        y1, y2, x1, x2 = 45, 471, 41, 601
        orig_h, orig_w = 480, 640
        y1_new = int((y1 / orig_h) * h)
        y2_new = int((y2 / orig_h) * h)
        x1_new = int((x1 / orig_w) * w)
        x2_new = int((x2 / orig_w) * w)
        eval_mask[:, :, y1_new:y2_new, x1_new:x2_new] = 1
    else:
        eval_mask.fill_(1)

    ignored_value_mask = torch.ones((B, 1, h, w), device=eval_mask.device)
    ignored_value_mask[(input == ignored_value).all(dim=1, keepdims=True)] = 0

    eval_mask = eval_mask * ignored_value_mask
    return eval_mask.bool()


@dataclass(frozen=True)
class _DepthMetric:
    name: str
    is_lower_better: bool

    @property
    def worst_value(self) -> float:
        return math.inf if self.is_lower_better else -math.inf

    def is_better(self, value1, value2) -> bool:
        sign = 1 if self.is_lower_better else -1
        return sign * value1 < sign * value2


DEPTH_METRICS = (
    _DepthMetric(name="a1", is_lower_better=False),
    _DepthMetric(name="a2", is_lower_better=False),
    _DepthMetric(name="a3", is_lower_better=False),
    _DepthMetric(name="abs_rel", is_lower_better=True),
    _DepthMetric(name="rmse", is_lower_better=True),
    _DepthMetric(name="log_10", is_lower_better=True),
    _DepthMetric(name="rmse_log", is_lower_better=True),
    _DepthMetric(name="silog", is_lower_better=True),
    _DepthMetric(name="sq_rel", is_lower_better=True),
    _DepthMetric(name="mae", is_lower_better=True),
)

_DepthMetricValues = namedtuple(
    "DepthMetricValues", [metric.name for metric in DEPTH_METRICS]
)  # type: ignore


def calculate_depth_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    list_metrics: list[_DepthMetric] = list(DEPTH_METRICS),
):
    if gt.shape[0] == 0:
        return [torch.nan] * len(DEPTH_METRICS)

    if valid_mask is not None:
        valid_mask = torch.logical_and(valid_mask, gt > 0)

    gt = gt[valid_mask]
    pred = pred[valid_mask]

    metrics_dict = {}

    metric_names = [metric.name for metric in list_metrics]

    thresh = torch.maximum((gt / pred), (pred / gt))
    metrics_dict["a1"] = (
        (thresh < 1.25).float().mean() if "a1" in metric_names else torch.nan
    )
    metrics_dict["a2"] = (
        (thresh < 1.25**2).float().mean() if "a2" in metric_names else torch.nan
    )
    metrics_dict["a3"] = (
        (thresh < 1.25**3).float().mean() if "a3" in metric_names else torch.nan
    )

    error = gt - pred
    sq_error = error**2
    metrics_dict["mae"] = (
        torch.mean(torch.abs(error)) if "mae" in metric_names else torch.nan
    )
    metrics_dict["abs_rel"] = (
        torch.mean(torch.abs(error) / gt) if "abs_rel" in metric_names else torch.nan
    )
    metrics_dict["sq_rel"] = (
        torch.mean(sq_error / gt) if "sq_rel" in metric_names else torch.nan
    )

    metrics_dict["rmse"] = (
        torch.sqrt(sq_error.mean()) if "rmse" in metric_names else torch.nan
    )

    error_log = torch.log(gt) - torch.log(pred)
    sq_error_log = error_log**2
    metrics_dict["rmse_log"] = (
        torch.sqrt(sq_error_log.mean()) if "rmse_log" in metric_names else torch.nan
    )
    if "silog" in metric_names:
        silog = torch.sqrt(torch.mean(sq_error_log) - torch.mean(error_log) ** 2) * 100
        if torch.isnan(silog):
            silog = torch.tensor(0)
        metrics_dict["silog"] = silog
    else:
        metrics_dict["silog"] = torch.nan
    metrics_dict["log_10"] = (
        (torch.abs(torch.log10(gt) - torch.log10(pred))).mean()
        if "log_10" in metric_names
        else math.inf
    )

    return _DepthMetricValues(**metrics_dict)


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

        self.abs_rel += float(
            (torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item()
        )
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

        self.abs_rel += float(
            (torch.abs(gt[mask] - pred[mask]) / gt[mask]).sum().item()
        )
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

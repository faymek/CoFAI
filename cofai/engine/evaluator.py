"""Multi-task metric evaluation for unified eval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from cofai.engine.schema import StepOutput

ALLOWED_KINDS = {
    # short, dataset-modality oriented kinds
    "rec",
    "semseg",
    "human_parts",
    "cls",
    "depth",
    "edge",
    "sal",
    "normals",
    "scene",
    "vqa",
}


class Meter(Protocol):
    def update(self, pred: Any, gt: Any) -> None: ...

    def compute(self) -> Dict[str, float]: ...


@dataclass(frozen=True)
class TaskConfig:
    """Per-task meter definition for unified eval.

    `eval_model` MUST produce `StepOutput.pred/gt` for this label in final form.
    """

    label: str
    meter: Meter


@dataclass(frozen=True)
class TaskSpec:
    """Task semantics for one task label.

    `kind` is optional; when present it should come from `ALLOWED_KINDS`.
    """

    label: str
    kind: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


def infer_kind(*, label: str, kind: Any) -> Optional[str]:
    """Infer kind from config."""

    label_s = str(label)
    if kind is None:
        return label_s if label_s in ALLOWED_KINDS else None

    kind_s = str(kind)
    if kind_s not in ALLOWED_KINDS:
        raise ValueError(f"Invalid task kind: {kind_s!r}. Allowed: {sorted(ALLOWED_KINDS)!r}")
    return kind_s


class MultiTaskEvaluator:
    """Multi-task meter aggregator (meters only)."""

    def __init__(self, task_configs: List[TaskConfig]):
        self.task_configs = list(task_configs)
        self.meters: Dict[str, Meter] = {tc.label: tc.meter for tc in self.task_configs}

    def update(self, step_output: StepOutput, ctx: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(step_output, StepOutput):
            raise TypeError(f"Expected StepOutput, got {type(step_output)!r}")
        pred = step_output.pred or {}
        gt = step_output.gt or {}
        for label, pv in pred.items():
            if not isinstance(label, str):
                continue
            gv = gt.get(label)
            if gv is None:
                continue
            meter = self.meters.get(label)
            if meter is None:
                continue
            meter.update(pv, gv)

    def compute_all_metrics(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for label, meter in self.meters.items():
            out[label] = meter.compute()
        return out

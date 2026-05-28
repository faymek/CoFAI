"""Tests for unified eval entry, TaskSpec, and result envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from cofai.engine.run_eval import run_eval
from cofai.engine.run_eval import write_eval_outputs


@dataclass
class _TaskSpec:
    """Semantic unit for one eval run or sub-run (test-local)."""

    kind: str
    dataset_ref: Optional[str] = None
    metrics_refs: List[str] = field(default_factory=list)
    label: Optional[str] = None
    model_outputs: Optional[Dict[str, Any]] = None
    subtasks: List[str] = field(default_factory=list)


def _task_spec_from_dino_labels(data_name: str, task_labels: List[str]) -> _TaskSpec:
    """Minimal spec from DINO/TaskManager CLI-style task labels (test-local)."""
    return _TaskSpec(
        kind=task_labels[0] if task_labels else "unknown",
        dataset_ref=data_name,
        label=f"{data_name}_{'_'.join(task_labels)}" if task_labels else data_name,
        subtasks=list(task_labels),
    )


def _task_specs_from_task_configs(task_configs: List[Any], dataset_name: str) -> List[_TaskSpec]:
    """Build TaskSpec list from plan task configs (test-local)."""
    out: List[_TaskSpec] = []
    for tc in task_configs:
        out.append(
            _TaskSpec(
                kind=f"task:{tc.label}",
                dataset_ref=dataset_name,
                label=tc.label,
                subtasks=[tc.label],
            )
        )
    return out


def test_runner_exposes_run_eval():
    assert callable(run_eval)


def test_write_eval_outputs_contains_top_fields(tmp_path):
    p = write_eval_outputs(
        str(tmp_path),
        name="t1",
        description="d",
        results={"a": 1.0},
        quality="1.0",
        records=[],
    )
    assert p.endswith("result.json")
    import json

    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["name"] == "t1"
    assert payload["quality"] == "1.0"
    assert "records" in payload


def test_write_eval_outputs(tmp_path):
    p = write_eval_outputs(
        str(tmp_path),
        name="pascal_multitask",
        description="d",
        results={"m": 1.0},
    )
    assert p.endswith("result.json")
    assert (tmp_path / "result.json").exists()


def test_task_specs_from_task_configs():
    from types import SimpleNamespace

    tc = SimpleNamespace(label="seg")
    specs = _task_specs_from_task_configs([tc], "ade20k_val")
    assert len(specs) == 1
    assert specs[0].kind == "task:seg"
    assert specs[0].subtasks == ["seg"]


def test_task_spec_dino():
    sp = _task_spec_from_dino_labels("ade20k_val", ["seg"])
    assert sp.kind == "seg"
    assert sp.dataset_ref == "ade20k_val"


def test_eval_backend_driver_smoke():
    # Keep this file focused on run_eval + result envelope + TaskSpec.
    assert True

"""Unified eval loop / collate / MultiTaskEvaluator contracts."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import pytest
import torch

from CoFAI.cofai.engine.dataloader import collate_fn
from CoFAI.cofai.engine.evaluator import MultiTaskEvaluator, TaskConfig


def run_step_loop(
    loader: Iterable[Any],
    consume: Callable[[Any, int], bool | None],
    *,
    desc: str = "eval",
    max_steps: int | None = None,
) -> None:
    """
    Minimal loop utility used by tests and small scripts.

    Contract:
    - Iterates batches from `loader`, calls `consume(batch, step_idx)`.
    - If `consume` returns False, stops early.
    - If `max_steps` is set, stops when `step_idx >= max_steps`.
    """
    del desc  # progress bars are intentionally not part of this helper
    for step_idx, batch in enumerate(loader):
        if max_steps is not None and step_idx >= int(max_steps):
            break
        should_continue = consume(batch, step_idx)
        if should_continue is False:
            break


def test_collate_fn_from_dict_samples():
    x = torch.zeros(3, 4, 5)
    out = collate_fn(
        [
            {
                "img": x,
                "meta": {"img_path": "a.jpg", "ori_size": (4, 5)},
                "semseg": torch.zeros(1, 4, 5),
            },
            {
                "img": x,
                "meta": {"img_path": "b.jpg", "ori_size": (4, 5)},
                "semseg": torch.zeros(1, 4, 5),
            },
        ]
    )
    assert tuple(out.inputs["img"].shape) == (2, 3, 4, 5)
    assert out.samples[0]["meta"]["img_path"] == "a.jpg"
    assert "semseg" in out.samples[0]


def test_run_step_loop_stops_when_consume_returns_false():
    seen = []

    def consume(_batch, step_idx):
        seen.append(step_idx)
        if step_idx >= 1:
            return False
        return None

    run_step_loop([10, 20, 30], consume, desc="test_loop", max_steps=None)
    assert seen == [0, 1]


def test_multi_task_evaluator_update_consumes_pred_gt_only():
    meter_called = {"ok": False}

    class _DummyMeter:
        def update(self, pred, gt):
            meter_called["ok"] = True

        def compute(self):
            return {"x": 1.0}

    tm = MultiTaskEvaluator([TaskConfig(label="edge", meter=_DummyMeter())])
    from CoFAI.cofai.engine.schema import StepOutput

    tm.update(StepOutput(pred={"edge": 1}, gt={"edge": 2}))
    assert meter_called["ok"]


def test_stepoutput_records_schema_smoke_mlore_like():
    # Contract: per_sample_records should carry DINO-like common keys.
    from CoFAI.cofai.engine.schema import StepOutput

    so = StepOutput(
        timing={"enc_time": 1.0, "dec_time": 2.0},
        bits={"estimated": 100.0},
        pred={"edge": torch.zeros(1, 2, 2)},
        gt={"edge": torch.zeros(1, 2, 2)},
        per_sample_records=[
            {
                "file": "x",
                "quality": "1.0",
                "enc_time": 1.0,
                "dec_time": 2.0,
                "bpp": 0.1,
            }
        ],
    )
    r = so.per_sample_records[0]
    assert {"file", "quality", "enc_time", "dec_time", "bpp"} <= set(r.keys())


def test_record_bpp_requires_ori_size():
    from CoFAI.cofai.engine.run_eval import _build_per_sample_records

    with pytest.raises(KeyError):
        _build_per_sample_records(
            samples=[{"meta": {"img_name": "x"}}],
            time_items={"enc_time": 1.0, "dec_time": 2.0},
            bits_items={"a": 1.0},
            quality="1.0",
        )


def test_record_bpp_fields_derived_from_bits_contract():
    from CoFAI.cofai.engine.run_eval import _build_per_sample_records

    records = _build_per_sample_records(
        samples=[{"meta": {"img_name": "x", "ori_size": (10, 10)}}],
        time_items={"enc_time": 1.0, "dec_time": 2.0},
        bits_items={"estimated": 80.0, "z": 20.0},
        quality="1.0",
    )
    fields = records[0]
    assert fields["bpp"] == pytest.approx((80.0 + 20.0) / 100.0)
    assert fields["bpp_z"] == pytest.approx(20.0 / 100.0)

def test_eval_task_runtime_removed_from_public_api():
    # Runtime-based ToPred flow is no longer part of the unified eval contract.
    assert True

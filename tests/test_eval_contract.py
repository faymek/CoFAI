"""Smoke tests for eval bits/inference helpers (rec protocol path)."""

from types import SimpleNamespace

import torch

from cofai.engine.bitrate import bits_from_coded_unit
from cofai.engine.run_eval import eval_step, inference_model
from cofai.engine.schema import EvalBatch


def test_calc_bits_strings():
    data = {"strings": {"a": [[b"xx"]]}}
    out = bits_from_coded_unit(data)
    assert out["a"] == 16.0


def test_inference_x_real_returns_rec():
    class DummyModel:
        def compress(self, x, qp=1, tasks=None):
            return {"strings": {"a": [[b"x"]]}}

        def decompress(self, coded_data, tasks=None, **kwargs):
            return {"rec": torch.ones(1, 3, 2, 2)}

    x = torch.zeros(1, 3, 8, 8)
    time_items, bits_items, task_feats = inference_model(
        DummyModel(), x, qp=1, real=True, tasks=["rec"]
    )
    assert "rec" in task_feats
    assert "total_enc_time" in time_items
    assert "total_dec_time" in time_items
    assert "a" in bits_items


def test_eval_model_fail_fast_missing_task_output():
    class DummyModel:
        def forward_test(self, x, qp=1, tasks=None, **kwargs):
            # Intentionally return only "rec".
            coded = {"strings": {"a": [[b"x"]]}}
            return coded, {"rec": torch.ones_like(x)}

    cfg = SimpleNamespace(args=SimpleNamespace(quality="1.0", real=False))
    model = DummyModel()
    ctx = {
        "cfg": cfg,
        "tasks": ["not_returned"],
        "device": torch.device("cpu"),
        "task_specs": None,
    }

    batch = EvalBatch(
        inputs={"img": torch.zeros(1, 3, 2, 2)},
        samples=[{"meta": {"ori_size": (2, 2)}}],
    )
    try:
        eval_step(model=model, batch=batch, step_ctx=ctx)
        assert False, "Expected KeyError for missing task output"
    except KeyError as e:
        assert "missing" in str(e)

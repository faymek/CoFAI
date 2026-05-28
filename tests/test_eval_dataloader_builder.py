"""Ensure run_eval does not require plan.dataloader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

from omegaconf import OmegaConf


@dataclass
class DummyBatch:
    samples: List[Any]


class DummyDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Any:
        return {"x": idx}

class DummyMeter:
    def update(self, pred: Any, gt: Any) -> None:
        return None

    def compute(self) -> dict:
        return {}


class DummyModel:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def to(self, device: Any):
        return self

    def eval(self):
        return self


class DummyPost:
    def __call__(self, *args, **kwargs):
        return None


class DummyWriter:
    def add(self, step_out: Any) -> None:
        return None

    def finalize(self) -> list:
        return []


class DummyDataLoaderBuilder:
    def __init__(self):
        self.called = False

    def build(self, cfg: Any, dataset: Any) -> Iterable[DummyBatch]:
        self.called = True
        # minimal iterable compatible with `for batch in loader`
        return [DummyBatch(samples=[{"id": 0}])]


def test_run_eval_uses_default_dataloader_when_plan_has_no_dataloader_key(monkeypatch):
    from CoFAI.cofai.engine import run_eval as runner_mod
    from CoFAI.cofai.engine import builder as builder_mod

    called = {"n": 0}

    def fake_build_dataloader(cfg: Any, dataset: Any):
        called["n"] += 1

        class _FakeLoader:
            def __init__(self, ds: Any) -> None:
                self.dataset = ds

            def __iter__(self) -> Any:
                return iter([])

        return _FakeLoader(dataset)

    monkeypatch.setattr(runner_mod, "build_dataloader", fake_build_dataloader)
    monkeypatch.setattr(builder_mod, "DummyDataset", DummyDataset, raising=False)

    def fake_instantiate_class(cfg: Any, **kwargs):
        cfg = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else dict(cfg)
        t = cfg.get("type")
        if t == "DummyDataset":
            return DummyDataset()
        if t == "DummyMeter":
            return DummyMeter()
        if t == "DummyModel":
            return DummyModel(**{k: v for k, v in cfg.items() if k != "type"}, **kwargs)
        if t == "DummyPost":
            return DummyPost()
        raise KeyError(f"Unexpected type in test: {t!r}")

    monkeypatch.setattr(builder_mod, "instantiate_class", fake_instantiate_class)

    cfg = OmegaConf.create(
        {
            "args": {
                "plan": "t",
                "device": "cpu",
                "output_dir": "",
                "max_samples": 1,
                "real": False,
                "quality": 1.0,
            },
            # cfg=plan (root is plan)
            "name": "t",
            "description": "d",
            "dataset": {"type": "DummyDataset"},
            "task_configs": [{"label": "x", "meter": {"type": "DummyMeter"}}],
            "model": {"type": "DummyModel"},
        }
    )

    summary, records = runner_mod.run_eval(cfg)
    assert summary == {}
    assert records == []
    assert called["n"] == 1


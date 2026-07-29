"""Hydra compose sanity (no GPU / no dataset IO)."""

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

_CONF = Path(__file__).resolve().parents[1] / "conf"


def _compose_plan_cfg(conf_path: str, overrides: list[str]):
    p = Path(conf_path).resolve()
    rel = p.relative_to(_CONF.resolve())
    name = str(rel.with_suffix("")).replace("\\", "/")
    with initialize_config_dir(version_base=None, config_dir=str(_CONF.resolve())):
        cfg = compose(config_name=name, overrides=list(overrides))
    if not cfg.get("PROJECT_ROOT"):
        OmegaConf.set_struct(cfg, False)
        cfg.PROJECT_ROOT = os.environ.get("PROJECT_ROOT", str(Path.cwd()))
        OmegaConf.set_struct(cfg, True)
    try:
        HydraConfig.instance().set_config(
            OmegaConf.create({"hydra": {"runtime": {"cwd": str(Path.cwd())}}})
        )
    except Exception:
        pass
    return cfg


@pytest.fixture(scope="module")
def conf_dir():
    return str(_CONF.resolve())


def test_compose_default_dinov2(conf_dir):
    cfg = _compose_plan_cfg(
        str(
            Path(conf_dir)
            / "plan"
            / "dinov2"
            / "ade20k-val__dinov2-vitb16-reg4-slot09__Bypass__semseg-last4.yaml"
        ),
        [],
    )
    assert cfg.name == "ade20k-val__dinov2-vitb16-reg4-slot09__Bypass__semseg-last4"
    assert cfg.model


def test_compose_override_plan(conf_dir):
    cfg = _compose_plan_cfg(
        str(
            Path(conf_dir)
            / "plan"
            / "dinov2"
            / "ade20k-val__dinov2-vitb16-slot09__MPC-vbr__semseg-last4.yaml"
        ),
        ["+args.max_samples=1"],
    )
    assert cfg.name == "ade20k-val__dinov2-vitb16-slot09__MPC-vbr__semseg-last4"
    assert cfg.model

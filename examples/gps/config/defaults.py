"""Load GPS YAML files with repository-relative path expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from omegaconf import DictConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - used only in minimal environments
    DictConfig = dict
    OmegaConf = None


class _ConfigNode(dict):
    """Small attribute-access wrapper used when OmegaConf is unavailable."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _wrap(value):
    if isinstance(value, dict):
        return _ConfigNode({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def load_config(path: str | Path, override: str | Path | None = None) -> DictConfig:
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    if OmegaConf is None:
        import yaml

        def load_yaml(config_path):
            value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
            text = str(value).replace("${PROJECT_ROOT}", str(project_root))
            return yaml.safe_load(text)

        cfg = load_yaml(path)
        if override is not None:
            extra = load_yaml(override)
            cfg.update(extra)
        return _wrap(cfg)
    OmegaConf.register_new_resolver("project_root", lambda: str(project_root), replace=True)
    cfg = OmegaConf.load(path)
    if override is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(override))
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def set_override(cfg: DictConfig, dotted_key: str, value: Any) -> None:
    """Set one dotted configuration key for command-line overrides."""
    OmegaConf.update(cfg, dotted_key, value, merge=False)

"""Load GPS YAML files with repository-relative path expansion."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, override: str | Path | None = None) -> DictConfig:
    load_dotenv(Path.cwd() / ".env", override=False)
    project_root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    OmegaConf.register_new_resolver(
        "project_root", lambda: str(project_root), replace=True
    )
    cfg = OmegaConf.load(path)
    if override is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(override))
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

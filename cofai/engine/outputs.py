"""Write evaluation results and resolved configuration files."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from cofai.engine.schema import EvalResultPayload


def write_eval_outputs(
    output_dir: str,
    *,
    name: str,
    description: str,
    results: Dict[str, Any],
    quality: Optional[str] = None,
    records: Optional[List[Dict[str, Any]]] = None,
    cfg: Any | None = None,
) -> str:
    """Write `result.json` and return its path."""

    os.makedirs(output_dir, exist_ok=True)

    payload: EvalResultPayload = {
        "name": name,
        "description": description,
        "results": results,
        "quality": quality,
        "records": records or [],
    }
    canonical_path = os.path.join(output_dir, "result.json")
    with open(canonical_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    if cfg is not None:
        with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    return canonical_path

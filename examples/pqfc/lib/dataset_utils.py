"""Re-export VTM dataset helpers for PQFC DINOv3 replay."""

from __future__ import annotations

import sys
from pathlib import Path

# examples/pqfc/lib -> examples/vtm/lib
_VTM_LIB = Path(__file__).resolve().parents[2] / "vtm" / "lib"
if not _VTM_LIB.is_dir():
    _VTM_LIB = Path(__file__).resolve().parents[3] / "examples" / "vtm" / "lib"
if str(_VTM_LIB) not in sys.path:
    sys.path.insert(0, str(_VTM_LIB))

from dataset_utils import (  # noqa: E402,F401
    build_backbone,
    build_dataset,
    build_head,
    build_meter,
    load_subset,
    sample_stem,
)

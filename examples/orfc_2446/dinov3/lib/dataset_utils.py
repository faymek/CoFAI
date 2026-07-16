"""Re-export VTM dataset helpers for ORFC replay."""

from __future__ import annotations

import sys
from pathlib import Path

# examples/orfc_2446/dinov3/lib -> examples/vtm/lib
_VTM_LIB = Path(__file__).resolve().parents[3] / "vtm" / "lib"
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

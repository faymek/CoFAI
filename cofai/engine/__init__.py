"""Engine-level skeleton (v0: eval-only).

`cofai.engine` is the long-term home for runner/config/builder contracts.
CLI 入口迁移到项目根目录 `test.py`。
"""

from __future__ import annotations

from cofai.engine.bitrate import bits_from_coded_unit
from cofai.engine.checkpoint import load_checkpoint_into_model

__all__ = ["bits_from_coded_unit", "load_checkpoint_into_model"]

from pathlib import Path

import yaml

from cofai.utils.download import DEFAULT_BASE_URL, parse_manifest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/vtc"


def test_checkpoint_manifest_covers_vtc_light_plan():
    entries = parse_manifest(EXAMPLE / "vtc-light.manifest.txt")
    plan = yaml.safe_load((EXAMPLE / "plan/ade20k-val--vtc-light.yaml").read_text())
    paths = {f"${{PROJECT_ROOT}}/{entry.dest}" for entry in entries}

    assert len(entries) == len(paths) == 2
    assert paths == {
        "${PROJECT_ROOT}/weights/VTC/dinov3-vitl16-slot24--VTC-light.pth.tar",
        "${PROJECT_ROOT}/weights/VTC/dinov3-vitl16-slot24--VTC-light-finetuned.pth.tar",
    }
    assert plan["load"]["path"] == "${PROJECT_ROOT}/weights/VTC/dinov3-vitl16-slot24--VTC-light-finetuned.pth.tar"
    defaults = yaml.safe_load((ROOT / "conf/args/default.yaml").read_text())["args"]
    assert defaults["quality"] == 0
    assert defaults["multi_run"] is False
    assert "multi_run" not in plan
    assert "rules" not in plan["load"]
    for entry in entries:
        assert entry.url == f"{DEFAULT_BASE_URL}/{entry.dest}"
        assert entry.algo == "sha256"
        assert entry.value

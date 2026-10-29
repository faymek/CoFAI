from pathlib import Path

import pytest
import yaml

from cofai.utils.download import DEFAULT_BASE_URL, parse_manifest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/vq_ufc"


@pytest.mark.parametrize(
    "plan_name",
    [
        "ade20k-val__dinov3-vitl16-slot24__VQUFC__semseg",
        "nyuv2-val__dinov3-vitl16-slot24__VQUFC__depth",
    ],
)
def test_checkpoint_manifest_matches_plan_paths(plan_name):
    entries = parse_manifest(EXAMPLE / "dinov3-vqufc.manifest.txt")
    plan = yaml.safe_load((EXAMPLE / "dinov3/plan" / f"{plan_name}.yaml").read_text())
    expected = {f"${{PROJECT_ROOT}}/{entry.dest}" for entry in entries}
    actual = {
        config["model"]["dino_codec"]["checkpoint_path"]
        for config in [plan, *plan["multi_run"].values()]
    }

    assert len(entries) == len(expected) == 16
    assert actual == expected
    for entry in entries:
        assert entry.url == f"{DEFAULT_BASE_URL}/{entry.dest}"
        assert entry.algo == "sha256"
        assert entry.value

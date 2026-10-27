"""PQFC entrypoints use native plans and explicit resource paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import zipfile

import pytest
import yaml

from cofai.utils.download import parse_manifest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/pqfc"


def _run(script, project_root, **overrides):
    env = {
        **os.environ,
        "PROJECT_ROOT": str(project_root),
        "PYTHON": f"{sys.executable} -u",
        "GPU_IDS": "0,1",
        "TASK": "both",
        "VARIANT": "both",
        "OUTPUT_DIR": str(project_root / "output"),
        "DRY_RUN": "0",
        **overrides,
    }
    return subprocess.run(
        ["bash", str(EXAMPLE / "scripts" / script)],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


@pytest.mark.parametrize("variant", ["transform", "noR", "both"])
@pytest.mark.parametrize("task", ["semseg", "depth", "both"])
def test_eval_dry_run_selects_native_plans(tmp_path, variant, task):
    result = _run("run_eval_ctc.sh", tmp_path, DRY_RUN="1", VARIANT=variant, TASK=task)
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")]
    expected = (2 if variant == "both" else 1) * (2 if task == "both" else 1)
    assert len(commands) == expected
    for command in commands:
        assert "-m cofai.engine.run_eval" in command
        assert "args.multi_run=true" in command
        assert "orfc_weights_path=" not in command
        if variant != "both":
            assert ("__PQFC-noR__" in command) == (variant == "noR")
        if task != "both":
            assert f"__{task}.yaml" in command
    assert f"PROJECT_ROOT={tmp_path}" in result.stdout
    assert not (tmp_path / "output").exists()


def test_eval_limits_gpu_concurrency_and_propagates_failures(tmp_path):
    runner = tmp_path / "fake_eval.py"
    runner.write_text(textwrap.dedent("""\
        import json
        import os
        from pathlib import Path
        import sys
        import time

        root = Path(os.environ['PROJECT_ROOT'])
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        lock = root / ('gpu-' + gpu)
        with lock.open('x'):
            pass
        plan = Path(sys.argv[3]).stem
        with (root / 'calls.jsonl').open('a') as stream:
            stream.write(json.dumps({'gpu': gpu, 'plan': plan}) + '\\n')
        time.sleep(0.15)
        lock.unlink()
        sys.exit(1 if os.environ.get('FAIL_PLAN') == plan else 0)
    """))
    interpreter = f"{sys.executable} {runner}"
    result = _run("run_eval_ctc.sh", tmp_path, PYTHON=interpreter)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 4
    assert len({call["plan"] for call in calls}) == 4
    assert [call["gpu"] for call in calls].count("0") == 2
    assert [call["gpu"] for call in calls].count("1") == 2

    (tmp_path / "calls.jsonl").unlink()
    first_plan = sorted((EXAMPLE / "plan/dinov3").glob("*__PQFC*.yaml"))[0].stem
    result = _run("run_eval_ctc.sh", tmp_path, PYTHON=interpreter, FAIL_PLAN=first_plan)
    assert result.returncode != 0
    assert first_plan in result.stderr
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 2


def test_manifest_covers_nor_plans():
    entries = {entry.dest: entry for entry in parse_manifest(EXAMPLE / "dinov3-pqfc.manifest.txt")}
    for plan_path in (EXAMPLE / "plan/dinov3").glob("*__PQFC-noR__*.yaml"):
        plan = yaml.safe_load(plan_path.read_text())
        for config in [plan, *plan["multi_run"].values()]:
            path = config["model"]["dino_codec"]["orfc_weights_path"].removeprefix("${PROJECT_ROOT}/")
            assert path in entries
            assert entries[path].algo == "sha256"
            assert len(entries[path].value) == 8


def test_prepare_unpacks_release_and_checks_all_qualities(tmp_path):
    for directory in ["data/ADEChallengeData2016", "data/NYU"]:
        (tmp_path / directory).mkdir(parents=True)
    artifacts = set()
    for plan_path in (EXAMPLE / "plan/dinov3").glob("*__PQFC*.yaml"):
        plan = yaml.safe_load(plan_path.read_text())
        artifacts.add(plan["model"]["dino_backbone"]["ckpt_path"])
        for head in plan["model"]["heads"].values():
            artifacts.add(head["checkpoint"])
        for config in [plan, *plan["multi_run"].values()]:
            artifacts.add(config["model"]["dino_codec"]["orfc_weights_path"])
    archive = tmp_path / "weights/orfc_2446/dinov3_vitl16_ori.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as release:
        for artifact in artifacts:
            path = tmp_path / artifact.removeprefix("${PROJECT_ROOT}/")
            if "dinov3_vitl16_ori/" in artifact:
                release.writestr(path.name, b"fixture")
                release.writestr(f"__MACOSX/._{path.name}", b"metadata")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
    result = _run("prepare_dinov3.sh", tmp_path, SKIP_DOWNLOAD="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "all 15 codec artifacts" in result.stdout
    assert not list((tmp_path / "weights/orfc_2446/dinov3_vitl16_ori").glob("._*"))
    assert not list((tmp_path / "weights/dinov3/backbone").glob("*.pth"))
    preserved = tmp_path / "weights/orfc_2446/dinov3_vitl16_ori/blk23_K16_e32.npz"
    preserved.write_bytes(b"existing")
    missing = tmp_path / "weights/pqfc/dinov3_vitl16_noR/blk23_K256_e8_no_transform.npz"
    missing.unlink()
    result = _run("prepare_dinov3.sh", tmp_path, SKIP_DOWNLOAD="1")
    assert result.returncode != 0
    assert str(missing) in result.stdout
    assert preserved.read_bytes() == b"existing"


def test_replay_uses_pqfc_local_helpers():
    from examples.pqfc.lib import dataset_utils, rate_eval
    from examples.pqfc import run_pqfc_dinov3 as replay
    from examples.pqfc.offline import extract_val_features as extract

    assert replay.build_dataset is dataset_utils.build_dataset
    assert extract.build_backbone is dataset_utils.build_backbone
    assert replay.evaluate_rate is rate_eval.evaluate_rate


def test_pqfc_reuses_existing_softpq_training_tools():
    from examples.orfc_2446.offline import artifacts, soft_pq
    from examples.pqfc.lib import config_utils
    from examples.pqfc.offline import train_pqfc_dinov3 as train

    assert train.load_config is config_utils.load_config
    assert train.train_soft_pq is soft_pq.train_soft_pq
    assert train.compute_histogram_pmf is artifacts.compute_histogram_pmf


def test_native_multi_run_fails_on_missing_quality(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    engine = importlib.import_module("cofai.engine.run_eval")
    cfg = OmegaConf.create({
        "args": {"multi_run": True, "output_dir": str(tmp_path), "quality": 0},
        "multi_run": {1: {"artifact": "present.npz"}, 2: {"artifact": "missing.npz"}},
    })
    monkeypatch.setattr(engine, "resolve_plan", lambda _: (
        None, SimpleNamespace(name="pqfc-plan", description="test"),
    ))
    attempted = []

    def evaluate(config, **kwargs):
        attempted.append(config.args.quality)
        if config.artifact == "missing.npz":
            raise FileNotFoundError(config.artifact)
        return {"bpp": 0.5}, []

    monkeypatch.setattr(engine, "run_eval", evaluate)
    with pytest.raises(FileNotFoundError, match="missing.npz"):
        engine.multi_run(cfg)
    assert attempted == [1, 2]
    assert not (tmp_path / "pqfc-plan/summary.json").exists()

"""Published DINOv2 ORFC plan and result discovery."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


SOURCE_ROOT = Path(__file__).resolve().parents[4]
PLAN_ROOT = SOURCE_ROOT / "examples" / "orfc_2446" / "plan" / "dinov2"
_PLAN_PATTERN = "*__ORFC__*.yaml"
_PLAN_NAME = re.compile(r"dinov2-(vit[lg]14).*__ORFC__(?:__)?(cls|semseg)$")
_CODEC_NAME = re.compile(r"^(blk\d+)_K(\d+)_emb(\d+)_bt(\d+)_ws" r"(?:_lmbda([\d.]+))?_tau([\d.]+)_lr([\d.]+)_ep(\d+)")


@dataclass(frozen=True)
class PlanResult:
    plan_name: str
    quality: str
    backbone: str
    layer: str
    task: str
    codec_path: str
    result_path: Path


def get_project_root() -> Path:
    """Return the shared artifact root configured for this checkout."""
    load_dotenv(SOURCE_ROOT / ".env", override=False, encoding="utf-8")
    value = os.environ.get("PROJECT_ROOT")
    if not value:
        raise RuntimeError(f"PROJECT_ROOT is required; set it or add it to {SOURCE_ROOT / '.env'}")

    project_root = Path(value).expanduser().resolve()
    if not project_root.is_dir():
        raise RuntimeError(f"PROJECT_ROOT is not a directory: {project_root}")
    return project_root


def parse_codec_params(codec_path: str) -> dict[str, int | float | str]:
    """Parse the release checkpoint naming convention into comparison fields."""
    match = _CODEC_NAME.match(Path(codec_path).name)
    if not match:
        raise ValueError(f"Unsupported DINOv2 SoftPQ checkpoint name: {codec_path}")
    return {
        "layer": match.group(1),
        "K": int(match.group(2)),
        "emb": int(match.group(3)),
        "bt": int(match.group(4)),
        "lambda": float(match.group(5)) if match.group(5) else 0.0,
        "tau": float(match.group(6)),
        "lr": float(match.group(7)),
        "epochs": int(match.group(8)),
    }


def format_codec_config(codec_path: str) -> str:
    """Return a stable human-readable label derived from a checkpoint name."""
    params = parse_codec_params(codec_path)
    return (
        f"K={params['K']}, e{params['emb']}, lambda={params['lambda']}, " f"lr={params['lr']:g}, ep={params['epochs']}"
    )


def discover_plan_results(results_dir: Path) -> list[PlanResult]:
    """Expand every published plan quality into its expected result path."""
    entries: list[PlanResult] = []
    for plan_path in sorted(PLAN_ROOT.glob(_PLAN_PATTERN)):
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        plan_name = str(plan["name"])
        plan_match = _PLAN_NAME.search(plan_name)
        if not plan_match:
            raise ValueError(f"Unsupported DINOv2 ORFC plan name: {plan_name}")

        backbone, task_name = plan_match.groups()
        task = "cls" if task_name == "cls" else "seg"
        multi_run = plan.get("multi_run")
        if not isinstance(multi_run, dict) or not multi_run:
            raise ValueError(f"Published plan has no multi_run entries: {plan_path}")

        for quality, overrides in multi_run.items():
            codec_path = str(overrides["model"]["dino_codec"]["orfc_weights_path"])
            params = parse_codec_params(codec_path)
            quality_name = str(quality)
            entries.append(
                PlanResult(
                    plan_name=plan_name,
                    quality=quality_name,
                    backbone=backbone,
                    layer=str(params["layer"]),
                    task=task,
                    codec_path=codec_path,
                    result_path=results_dir / plan_name / f"q{quality_name}" / "result.json",
                )
            )
    return entries


def load_metrics(entry: PlanResult) -> tuple[float, float]:
    """Load BPFP and the task metric from one engine result."""
    results = json.loads(entry.result_path.read_text(encoding="utf-8"))["results"]
    bpfp = float(results["bpfp"])
    if entry.task == "cls":
        return bpfp, float(results["cls_top-1"])

    miou = float(results["semseg_mIoU"])
    return bpfp, miou * 100.0 if miou <= 1.0 else miou

"""Regression coverage for shared-checkpoint generated-box detection."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from torch import nn

from cofai.datasets.odinw13 import ODinW13Dataset
from cofai.engine.run_eval import eval_step
from cofai.engine.schema import EvalBatch
from cofai.metrics.odinw13 import ODinWDetectionMeter, parse_bbox_answer
from cofai.models import base as model_base


@pytest.fixture
def dataset(tmp_path):
    subsets = []
    for name, ids, category in [("alpha", [1, 2], 11), ("beta", [1], 99)]:
        folder = tmp_path / name / "valid"
        folder.mkdir(parents=True)
        for image_id in ids:
            Image.new("RGB", (100, 100)).save(folder / f"{image_id}.png")
        annotations = {
            "info": {},
            "images": [
                {"id": i, "width": 100, "height": 100, "file_name": f"{i}.png"}
                for i in ids
            ],
            "categories": [{"id": category, "name": "object"}],
            "annotations": [
                {
                    "id": i,
                    "image_id": i,
                    "category_id": category,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
                for i in ids
            ],
        }
        (folder / "annotations.json").write_text(json.dumps(annotations))
        subsets.append(
            {
                "data_root": f"data/odinw/{name}/",
                "ann_file": "valid/annotations.json",
                "data_prefix": {"img": "valid/"},
                "metainfo": {"classes": ["object"]},
            }
        )
    (tmp_path / "odinw13_config.py").write_text(
        f"dataset_prefixes = ['alpha', 'beta']\ndatasets = {subsets!r}\n"
    )
    return ODinW13Dataset(tmp_path)


ANSWER = '[{"bbox_2d": [100, 100, 300, 300], "label": "OBJECT"}]'


def test_local_ids_and_missed_images_are_scored_per_dataset(dataset):
    assert dataset.per_dataset_counts == {"alpha": 2, "beta": 1}
    assert dataset[0]["img"].shape == (3, 100, 100)
    assert dataset[0]["det"]["image_id"] == dataset[2]["det"]["image_id"] == 1
    meter = ODinWDetectionMeter()
    for index in range(len(dataset)):
        target = dataset[index]["det"]
        meter.update(ANSWER if target["dataset_name"] == "alpha" else "[]", target)
    metrics = meter.compute()
    assert metrics["dataset/alpha/mAP"] == pytest.approx(1.0)
    assert metrics["dataset/beta/mAP"] == 0.0
    assert metrics["mAP"] == pytest.approx(0.5)


def test_absolute_coordinates_and_unknown_labels(dataset):
    target = {**dataset[0]["det"], "resized_size": (300, 200)}
    meter = ODinWDetectionMeter(coord_mode="abs")
    answer = '[{"bbox_2d": [20, 30, 60, 90], "label": "object"}, {"bbox_2d": [0, 0, 99, 99], "label": "unknown"}]'
    meter.update(answer, target)
    assert meter.compute()["mAP"] == pytest.approx(1.0)


@pytest.mark.parametrize("answer", [None, "broken json", '[{"bbox_2d": [1,2,3]}]'])
def test_malformed_answers_are_empty(answer):
    assert parse_bbox_answer(answer) == ([], [])


def test_fenced_answer_with_reasoning():
    assert parse_bbox_answer("reason</think>```json\n" + ANSWER + "\n```") == (
        [[100, 100, 300, 300]],
        ["OBJECT"],
    )


class TextBackbone(nn.Module):
    def encode_image(self, image):
        return torch.ones(1, 4, 3), (2, 2)

    def decode_text(self, features, prompt, token_res):
        assert features.shape == (1, 4, 3) and token_res == (2, 2)
        return ANSWER if prompt.startswith("Locate") else "A"


@pytest.fixture
def model(monkeypatch):
    original = model_base.instantiate_class
    monkeypatch.setattr(
        model_base,
        "instantiate_class",
        lambda config: (
            TextBackbone()
            if config["type"] == "test.TextBackbone"
            else original(config)
        ),
    )
    return model_base.Qwen3vlFeatureCodecModel(
        qwen_backbone={"type": "test.TextBackbone"}
    )


def test_vqa_and_detection_use_the_same_codec_in_both_paths(model, dataset):
    tasks = ["vqa", "det"]
    task_data = {"vqa": {"prompt": "Choose an answer"}, "det": dataset[0]["det"]}
    image = torch.zeros(1, 3, 100, 100)
    _, estimated = model.forward_test(image, tasks=tasks, task_data=task_data)
    actual = model.decompress(model.compress(image), tasks=tasks, task_data=task_data)
    assert actual == estimated == {"vqa": "A", "det": ANSWER}


def test_engine_records_generated_boxes(model, dataset):
    batch = EvalBatch(
        inputs={"img": dataset[0]["img"].unsqueeze(0)}, samples=[dataset[0]]
    )
    context = {
        "cfg": SimpleNamespace(
            args=SimpleNamespace(quality=0, real=True, profile=False)
        ),
        "tasks": ["det"],
        "task_specs": [{"label": "det", "kind": "det"}],
        "device": torch.device("cpu"),
    }
    result = eval_step(model=model, batch=batch, step_ctx=context)
    assert result.pred["det"] == ANSWER
    assert result.per_sample_records[0]["task_predictions"] == {"det": ANSWER}


def test_mmstar_and_odinw_share_every_checkpoint():
    repo = Path(__file__).resolve().parents[1]
    folder = repo / "examples/dsvfc/plan"
    plans = []
    for dataset, task in [("mmstar", "vqa"), ("odinw13", "det")]:
        with initialize_config_dir(version_base=None, config_dir=str(folder)):
            plans.append(
                compose(
                    config_name=f"{dataset}__qwen3vl-8b-ins-slot9__DSVFC__{task}",
                    overrides=[
                        f"hydra.searchpath=[file://{repo / 'conf'}]",
                        "+PROJECT_ROOT=/resources",
                    ],
                )
            )
    mmstar, odinw = plans
    assert mmstar.load == odinw.load
    assert mmstar.multi_run == odinw.multi_run
    assert mmstar.model.qwen_codec == odinw.model.qwen_codec
    assert mmstar.args.max_samples is odinw.args.max_samples is None

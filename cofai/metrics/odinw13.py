"""Generated-box parsing and macro-averaged COCO metrics for ODinW-13."""

import ast
import json
from typing import ClassVar

import numpy as np

from cofai.metrics.coco_detection import CocoDetectionMeter


def parse_bbox_answer(answer):
    """Preserve the proposal parser's fences, reasoning tags, and empty answers."""
    if answer is None:
        return [], []
    answer = str(answer).split("</think>")[-1].strip()
    answer = answer.replace("```json", "").replace("```", "")
    try:
        boxes, labels = [], []
        for item in ast.literal_eval(answer):
            if len(item.get("bbox_2d", [])) != 4:
                continue
            boxes.append(item["bbox_2d"])
            labels.append(item["label"])
        return boxes, labels
    except (SyntaxError, ValueError, TypeError, AttributeError, KeyError):
        return [], []


class ODinWDetectionMeter:
    """Use the common COCO meter per subset, then average the 13 results.

    The proposal uses normalized 0..1000 coordinates and unit confidence.
    Its COCO ``maxDets`` and per-subset three-decimal rounding are retained.
    ``abs`` coordinates use the resize bookkeeping provided by the dataset.
    """

    _METRICS: ClassVar[dict[str, str]] = {
        "mAP": "AP",
        "mAP_50": "AP50",
        "mAP_75": "AP75",
        "mAP_s": "APs",
        "mAP_m": "APm",
        "mAP_l": "APl",
    }

    def __init__(self, coord_mode="rel", proposal_nums=(100, 300, 1000)):
        if coord_mode not in {"rel", "abs"}:
            raise ValueError(f"Unknown coordinate mode: {coord_mode}")
        self.coord_mode = coord_mode
        self.proposal_nums = tuple(proposal_nums)
        self._meters = {}
        self._category_ids = {}
        self.per_dataset_results = {}

    def update(self, answer, target):
        name = target["dataset_name"]
        if name not in self._meters:
            self._meters[name] = CocoDetectionMeter(
                target["anno_path"], max_dets=self.proposal_nums
            )
            with open(target["anno_path"]) as stream:
                self._category_ids[name] = [
                    category["id"] for category in json.load(stream)["categories"]
                ]
        boxes, labels = parse_bbox_answer(answer)
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        height, width = target["orig_size"]
        if self.coord_mode == "rel":
            boxes = boxes / 1000.0 * np.array([width, height, width, height])
        else:
            resized_h, resized_w = target["resized_size"]
            boxes = boxes / np.array([resized_w, resized_h, resized_w, resized_h])
            boxes = boxes * np.array([width, height, width, height])
        classes = [str(label).lower() for label in target["classes"]]
        detections = []
        for box, label in zip(boxes.tolist(), labels):
            label = str(label).lower()
            if label not in classes:
                continue
            detections.append(
                {
                    "category_id": self._category_ids[name][classes.index(label)],
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": 1.0,
                }
            )
        self._meters[name].update(detections, target)

    def compute(self):
        self.per_dataset_results = {}
        for name, meter in sorted(self._meters.items()):
            metrics = meter.compute()
            self.per_dataset_results[name] = {
                key: float(round(metrics[value], 3))
                for key, value in self._METRICS.items()
            }
        if not self.per_dataset_results:
            return {key: 0.0 for key in self._METRICS}
        result = {
            key: float(np.mean([row[key] for row in self.per_dataset_results.values()]))
            for key in self._METRICS
        }
        for name, metrics in self.per_dataset_results.items():
            result.update(
                {f"dataset/{name}/{key}": value for key, value in metrics.items()}
            )
        return result

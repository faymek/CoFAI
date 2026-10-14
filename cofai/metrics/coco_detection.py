from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List


class CocoDetectionMeter:
    """Accumulate COCO-format detections and report bbox AP on a fixed json."""

    def __init__(self, ann_file: str, score_thr: float = 0.001):
        self.ann_file = str(ann_file)
        self.score_thr = float(score_thr)
        self.results: List[Dict[str, Any]] = []
        self.image_ids: set[int] = set()
        if not Path(self.ann_file).is_file():
            raise FileNotFoundError(f"COCO ann_file not found: {self.ann_file}")

    def update(self, pred: Any, gt: Any) -> None:
        image_id = int(gt["image_id"])
        self.image_ids.add(image_id)
        dets = pred
        if dets is None:
            return
        if isinstance(dets, dict):
            dets = [dets]
        for det in dets:
            score = float(det.get("score", 0.0))
            if score < self.score_thr:
                continue
            item = dict(det)
            item.setdefault("image_id", image_id)
            self.results.append(item)

    def compute(self) -> Dict[str, float]:
        empty = {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APs": 0.0,
            "APm": 0.0,
            "APl": 0.0,
        }
        if not self.results:
            return empty

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt = COCO(self.ann_file)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(self.results, tmp)
            tmp_path = tmp.name
        try:
            coco_dt = coco_gt.loadRes(tmp_path)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = sorted(self.image_ids)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = coco_eval.stats.tolist()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "APs": float(stats[3]),
            "APm": float(stats[4]),
            "APl": float(stats[5]),
        }

"""Regression tests for fixed-subset detection metrics and image geometry."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cofai.datasets.coco import CocoDetectionDataset
from cofai.heads.dinov3_det_head import Dinov3PlainDETRHead
from cofai.metrics.coco_detection import CocoDetectionMeter
from cofai.transforms import PadToMultiple, ResizeShortSide, ToTensor


@pytest.fixture
def annotations(tmp_path):
    ids = [3, 21, 44]
    data = {
        "info": {},
        "images": [{"id": i, "width": 100, "height": 100} for i in ids],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
            for i in ids
        ],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(data))
    return path


def detection(image_id=3, score=0.9):
    return {"image_id": image_id, "category_id": 1, "bbox": [10, 10, 20, 20], "score": score}


@pytest.mark.parametrize("missed", [[], None, [detection(21, 0.0001)]])
def test_metric_includes_images_without_retained_predictions(annotations, missed):
    meter = CocoDetectionMeter(str(annotations))
    meter.update([detection()], {"image_id": 3})
    meter.update(missed, {"image_id": 21})
    meter.update([], {"image_id": 21})
    assert meter.image_ids == {3, 21}
    assert meter.compute()["AP"] == pytest.approx(0.5049504950495048)


def test_metric_uses_evaluated_subset_not_all_annotations(annotations):
    meter = CocoDetectionMeter(str(annotations))
    meter.update([detection()], {"image_id": 3})
    assert meter.compute()["AP"] == pytest.approx(1.0)


def test_metric_all_empty_and_missing_prediction_id(annotations):
    meter = CocoDetectionMeter(str(annotations))
    meter.update([], {"image_id": 21})
    assert meter.compute()["AP"] == 0.0
    pred = detection()
    del pred["image_id"]
    meter.update(pred, {"image_id": 3})
    assert meter.results[0]["image_id"] == 3
    assert meter.compute()["AP"] == pytest.approx(0.5049504950495048)


def vendor_modules():
    source = Path(__file__).resolve().parents[1] / "vendor" / "dinov3"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from dinov3.eval.detection.models import detr
    from dinov3.eval.detection.util import misc
    return detr, misc


def test_default_source_and_strict_checkpoint_loading(tmp_path, monkeypatch):
    detr, _ = vendor_modules()
    monkeypatch.setattr(detr, "build_model", lambda *args: nn.Linear(2, 2))
    state = nn.Linear(2, 2).state_dict()
    state["backbone.0.backbone.unused"] = torch.zeros(1)
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": state}, path)
    head = Dinov3PlainDETRHead(str(path))
    assert torch.equal(head.model.weight, state["weight"])
    del state["bias"]
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        Dinov3PlainDETRHead(str(path))
    state["bias"] = torch.zeros(2)
    state["unexpected"] = torch.zeros(1)
    torch.save({"model": state}, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        Dinov3PlainDETRHead(str(path))
    with pytest.raises(FileNotFoundError, match="detection source"):
        Dinov3PlainDETRHead(str(path), dinov3_src=str(tmp_path / "missing"))


class FixedTransformer(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        self.two_stage_num_proposals = 2

    def forward(self, srcs, masks, pos, queries, attention_mask):
        hidden = torch.zeros(1, 1, 1, 4)
        return hidden, self.reference, [], torch.zeros(1, 1, 2), self.reference, None, None, None


def prediction_head(reference):
    detr, misc = vendor_modules()
    head = Dinov3PlainDETRHead.__new__(Dinov3PlainDETRHead)
    nn.Module.__init__(head)
    head.score_thr = 0.001
    head._NestedTensor = misc.NestedTensor
    head._inverse_sigmoid = misc.inverse_sigmoid
    head.postprocess = detr.PostProcess(topk=1, reparam=False)
    classifier = nn.Linear(4, 2)
    regressor = nn.Linear(4, 4)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.bias.copy_(torch.tensor([-10.0, 10.0]))
        regressor.weight.zero_()
        regressor.bias.zero_()
    head.model = SimpleNamespace(
        backbone=[None, [lambda nested: torch.zeros_like(nested.tensors)]],
        input_proj=[nn.Identity()],
        transformer=FixedTransformer(reference),
        num_queries=2,
        num_queries_one2one=1,
        two_stage=True,
        mixed_selection=True,
        query_embed=nn.Embedding(2, 4),
        class_embed=[classifier],
        bbox_embed=[regressor],
    )
    return head


@pytest.mark.parametrize("original_size", [(480, 640), (640, 480), (321, 1600), (1024, 1024)])
def test_dataset_propagates_resize_and_padding_geometry(tmp_path, original_size):
    from PIL import Image
    image_dir = tmp_path / "val2017"
    image_dir.mkdir()
    h, w = original_size
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(image_dir / "000000000003.jpg")
    (tmp_path / "coco_val_1k.txt").write_text("000000000003\n")
    resize = ResizeShortSide(size=1024, max_size=1536)
    pad = PadToMultiple(multiple=16, keys=["img"])
    tensor = ToTensor(keys=["img"])
    dataset = CocoDetectionDataset(tmp_path, transform=lambda sample: tensor(pad(resize(sample))))
    sample = dataset[0]
    assert sample["det"]["orig_size"] == original_size
    roi = sample["det"]["valid_roi"]
    resized_w, resized_h = resize._new_wh(w, h)
    assert (roi["height"], roi["width"]) == (resized_h, resized_w)
    canvas_h, canvas_w = sample["img"].shape[-2:]
    assert (roi["top"], roi["left"]) == ((canvas_h - resized_h) // 2, (canvas_w - resized_w) // 2)
    expected = torch.tensor([w * 0.2, h * 0.2, w * 0.4, h * 0.4])
    normalized = torch.tensor([[
        (roi["left"] + resized_w * 0.4) / canvas_w,
        (roi["top"] + resized_h * 0.4) / canvas_h,
        resized_w * 0.4 / canvas_w,
        resized_h * 0.4 / canvas_h,
    ]]).unsqueeze(0)
    head = prediction_head(normalized)
    features = [torch.zeros(1, 1, canvas_h // 16, canvas_w // 16)] * 4
    result = head.predict(features, **sample["det"])
    assert result[0]["bbox"] == pytest.approx(expected.tolist(), abs=0.0002)
    assert result[0]["image_id"] == 3
    assert head.model.num_queries == 2
    assert head.model.transformer.two_stage_num_proposals == 2


def test_unpadded_geometry_preserves_predictions_exactly():
    head = prediction_head(torch.tensor([[[0.4, 0.4, 0.4, 0.4]]]))
    features = [torch.zeros(1, 1, 4, 5)] * 4
    plain = head.predict(features, orig_size=(48, 60), image_id=3)
    roi = {"top": 0, "left": 0, "height": 64, "width": 80}
    assert head.predict(features, orig_size=(48, 60), image_id=3, valid_roi=roi) == plain


def test_forward_and_real_decode_pass_detection_geometry():
    from cofai.models.base import DinoFeatureCodecModel

    class Backbone(nn.Module):
        def encode(self, image):
            return torch.zeros(1, 1, 4)

        def decode_det(self, features, token_res):
            return [features] * 4

    class Codec(nn.Module):
        def forward(self, features, token_res, qp=0):
            return {"h_hat": features}

        def decompress(self, **coded_unit):
            return {"h_hat": torch.zeros(1, 1, 4)}

    class Head(nn.Module):
        def predict(self, features, **kwargs):
            return kwargs

    model = DinoFeatureCodecModel.__new__(DinoFeatureCodecModel)
    nn.Module.__init__(model)
    model.patch_size = 16
    model.dino = Backbone()
    model.dino_codec = Codec()
    model.heads = nn.ModuleDict({"det": Head()})
    data = {"image_id": 3, "orig_size": (10, 11), "valid_roi": {"top": 3, "left": 2, "height": 10, "width": 11}}
    _, estimated = model.forward_test(torch.zeros(1, 3, 16, 16), tasks=["det"], task_data={"det": data})
    actual = model.decompress({"strings": {}, "pstate": {"token_res": (1, 1)}}, tasks=["det"], task_data={"det": data})
    assert estimated["det"] == actual["det"] == data

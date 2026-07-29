"""Evaluation-only GPS ReID loop."""

from __future__ import annotations

import time

import numpy as np
import torch

from cofai.engine.run_eval import _bits_from_coded_unit
from cofai.models import CommonFeatureCodecModel

from .metrics import R1_mAP_eval


def _output_order(batch_meta, dataset_name: str) -> np.ndarray:
    """Map grouped GPS embeddings back to dataset-level batch metadata."""
    pids = np.asarray(batch_meta["pid"])
    views_per_group = 3 if dataset_name == "query" else 1
    if pids.size % views_per_group:
        raise ValueError(
            f"{dataset_name} batch size must be divisible by {views_per_group}"
        )
    groups = np.arange(pids.size).reshape(-1, views_per_group)
    if views_per_group > 1 and not np.all(
        pids[groups] == pids[groups][:, :1]
    ):
        raise ValueError("each GPS query group must contain one vehicle identity")
    return groups[:, 0]


def evaluate_model(
    cfg,
    model,
    query_loader,
    gallery_loader,
    num_query,
    *,
    real_codec=False,
):
    """Run multi-view retrieval and return task quality plus coded query rate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    rate_bits: dict[str, float] = {}
    coded_groups = 0

    start = time.perf_counter()
    with torch.inference_mode():
        for loader, dataset_name in (
            (query_loader, "query"),
            (gallery_loader, "gallery"),
        ):
            for img, batch_meta in loader:
                img = img.to(device, non_blocking=True)
                pid = batch_meta["pid"]
                camid = batch_meta["camid"]
                model_kwargs = {
                    "cam_label": batch_meta["camera_label"].to(
                        device,
                        non_blocking=True,
                    ),
                    "view_label": batch_meta["view_label"].to(
                        device,
                        non_blocking=True,
                    ),
                    "label": torch.as_tensor(pid, dtype=torch.long, device=device),
                    "multi_view": True,
                    "flip_view": False,
                    "extra_token": False,
                    "dataset_name": dataset_name,
                }
                if isinstance(model, CommonFeatureCodecModel):
                    if real_codec:
                        coded = model.compress(img, tasks=["reid"], **model_kwargs)
                        task_outputs = model.decompress(coded, tasks=["reid"])
                    else:
                        coded, task_outputs = model.forward_test(
                            img,
                            tasks=["reid"],
                            **model_kwargs,
                        )
                    feature = task_outputs["reid"]
                    if dataset_name == "query":
                        for name, value in _bits_from_coded_unit(coded).items():
                            rate_bits[name] = rate_bits.get(name, 0.0) + float(value)
                        coded_groups += int(feature.shape[0])
                else:
                    feature, _model_order = model(img, **model_kwargs)

                order = _output_order(batch_meta, dataset_name)
                evaluator.update(
                    (
                        feature,
                        np.asarray(pid)[order],
                        np.asarray(camid)[order],
                    )
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    cmc, mAP, *_ = evaluator.compute()
    result = {
        "mAP": float(mAP),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "num_query": int(num_query),
        "elapsed_seconds": float(elapsed),
    }
    if coded_groups:
        result["bits"] = rate_bits
        result["coded_groups"] = coded_groups
    return result

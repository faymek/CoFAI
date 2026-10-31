import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from tqdm import tqdm

from cofai.metrics.cv_metrics import MeanIoUMetric
from examples.protrac.config import load_config
from examples.protrac.evaluation import infer, save_results
from examples.protrac.model import build_codec_model
from examples.protrac.semseg.dataset import build_dataset


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Evaluate ProTRAC on ADE20K-C.")
    parser.add_argument("--config", default="examples/protrac/config/ade20k_c.yaml")
    parser.add_argument("--corruptions", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--real-codec", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.corruptions:
        cfg.corruptions = args.corruptions
    if args.output:
        cfg.output = args.output
    torch.set_num_threads(4)
    device = torch.device(cfg.device)
    model = build_codec_model(cfg, device)
    results = {}
    for corruption in cfg.corruptions:
        dataset = build_dataset(cfg.data_root, cfg.corruption_root, corruption, cfg.severity)
        model.backbone.reset()
        metric = MeanIoUMetric(150)
        for sample in tqdm(dataset, total=len(dataset), desc=corruption):
            image = sample["img"].contiguous().unsqueeze(0).to(device)
            target = sample["semseg"].squeeze(0).long()
            valid = sample["meta"]["valid_roi"]
            roi = (
                slice(valid["top"], valid["top"] + valid["height"]),
                slice(valid["left"], valid["left"] + valid["width"]),
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = infer(model, image, "semseg", args.real_codec, valid_roi=roi)
                logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
            prediction = logits[0, :, roi[0], roi[1]].argmax(dim=0)
            metric.update(prediction, target)
            model.backbone.update()
        results[corruption] = {"mIoU": 100 * float(metric.compute()["mIoU"]), "images": len(dataset)}
        print(f"{corruption}: {results[corruption]}", flush=True)
        save_results(cfg, results)
    print(f"Mean mIoU: {sum(value['mIoU'] for value in results.values()) / len(results):.2f}")


if __name__ == "__main__":
    main()

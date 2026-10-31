import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.protrac.config import load_config
from examples.protrac.evaluation import infer, save_results
from examples.protrac.model import build_codec_model
from examples.protrac.reid.dataset import AGReID
from examples.protrac.reid.metrics import retrieval_metrics


@torch.inference_mode()
def extract(model, dataset, cfg, view=None, real_codec=False):
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    features = []
    for images in tqdm(loader, desc=view or "gallery"):
        embedding = infer(model, images.to(cfg.device), "reid", real_codec, view=view, calibrate=view is not None)
        features.append(embedding.cpu())
        if view is not None:
            model.backbone.update()
    return torch.cat(features)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Evaluate ProTRAC on AG-ReID-C.")
    parser.add_argument("--config", default="examples/protrac/config/agreid.yaml")
    parser.add_argument("--corruptions", nargs="+")
    parser.add_argument("--protocols", nargs="+", choices=["AG", "GA"])
    parser.add_argument("--output")
    parser.add_argument("--real-codec", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.corruptions:
        cfg.corruptions = args.corruptions
    if args.protocols:
        cfg.protocols = args.protocols
    if args.output:
        cfg.output = args.output
    torch.set_num_threads(4)
    model = build_codec_model(cfg, torch.device(cfg.device))
    root = Path(cfg.data_root)
    results = {}
    for protocol in cfg.protocols:
        query_camera, gallery_camera, view = (0, 3, "Aerial") if protocol == "AG" else (3, 0, "Ground")
        gallery_data = AGReID(root / f"bounding_box_test_all_c{gallery_camera}", cfg.image_size)
        gallery = extract(model, gallery_data, cfg, real_codec=args.real_codec)
        results[protocol] = {}
        for corruption in cfg.corruptions:
            query_data = AGReID(
                root / "query_corruption" / corruption / str(cfg.severity) / f"query_all_c{query_camera}",
                cfg.image_size,
            )
            model.backbone.reset()
            query = extract(model, query_data, cfg, view, args.real_codec)
            metrics = retrieval_metrics(query, gallery, query_data, gallery_data)
            metrics.update(queries=len(query_data), gallery=len(gallery_data))
            results[protocol][corruption] = metrics
            print(f"{protocol} {corruption}: {metrics}", flush=True)
            save_results(cfg, results)
        domains = results[protocol].values()
        print(
            f"{protocol} mean: mAP={sum(value['mAP'] for value in domains) / len(results[protocol]):.2f}, "
            f"Rank-1={sum(value['Rank-1'] for value in domains) / len(results[protocol]):.2f}"
        )


if __name__ == "__main__":
    main()

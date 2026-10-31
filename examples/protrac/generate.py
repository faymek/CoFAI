import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import ROOT, load_config


def initialize_corruptions():
    import cv2
    import imagecorruptions
    from imagecorruptions import corruptions
    from skimage.filters import gaussian
    from skimage.util import random_noise

    def gaussian_filter(image, sigma, multichannel=False, **kwargs):
        return gaussian(image, sigma, channel_axis=-1 if multichannel else None, **kwargs)

    def impulse_noise(image, severity=1):
        amount = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
        rng = np.random.default_rng(np.random.randint(2**32))
        return random_noise(np.asarray(image) / 255, mode="s&p", amount=amount, rng=rng) * 255

    # imagecorruptions 1.1.2 predates scikit-image's channel_axis and rng APIs.
    corruptions.gaussian = gaussian_filter
    imagecorruptions.corruption_dict["impulse_noise"] = impulse_noise
    cv2.setNumThreads(1)


def generate_image(task):
    from imagecorruptions import corrupt

    path, destination, corruption, severity, seed, quality = task
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with Image.open(path) as source:
        image = source.convert("RGB")
    if min(image.size) < 32:
        scale = 32 / min(image.size)
        image = image.resize(tuple(ceil(size * scale) for size in image.size), Image.Resampling.BICUBIC)
    payload = f"{seed}|{path.parent.name}|{path.name}|{corruption}|{severity}".encode()
    np.random.seed(int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest()[:4], "little"))
    pixels = corrupt(np.asarray(image), corruption_name=corruption, severity=severity)
    Image.fromarray(pixels).save(destination, quality=quality)


def main(task):
    segmentation = task == "semseg"
    config = ROOT / "examples/protrac/config" / ("ade20k_c.yaml" if segmentation else "agreid.yaml")
    cfg = load_config(config)
    parser = argparse.ArgumentParser(description=f"Generate {'ADE20K-C' if segmentation else 'AG-ReID-C'}.")
    parser.add_argument("--input-root", type=Path, default=Path(cfg.data_root))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(cfg.corruption_root) if segmentation else Path(cfg.data_root) / "query_corruption",
    )
    parser.add_argument("--corruptions", nargs="+", choices=list(cfg.corruptions), default=list(cfg.corruptions))
    parser.add_argument("--severity", type=int, choices=range(1, 6), default=cfg.severity)
    parser.add_argument("--seed", type=int, default=20260313)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    source_root = args.input_root / "images" if segmentation else args.input_root
    splits = ["validation"] if segmentation else ["query_all_c0", "query_all_c3"]
    sources = []
    for split in splits:
        images = sorted((source_root / split).glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No images in {source_root / split}")
        sources.extend(images)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_corruptions) as pool:
        for corruption in args.corruptions:
            jobs = [
                (
                    path,
                    args.output_root / corruption / str(args.severity) / path.parent.name / path.name,
                    corruption,
                    args.severity,
                    args.seed,
                    75 if segmentation else 95,
                )
                for path in sources
            ]
            for _ in tqdm(pool.map(generate_image, jobs, chunksize=8), total=len(jobs), desc=corruption):
                pass

"""ODinW-13 validation images and prompts for generated-box detection."""

import json
import math
import runpy
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


def _resized_size(height, width, min_pixels, max_pixels, factor=32):
    """Match the proposal's ODinW resize bookkeeping for absolute boxes."""
    if height < 2 or width < 2 or max(height, width) / min(height, width) > 200:
        raise ValueError(f"Unsupported ODinW image dimensions: {(height, width)}")
    if max(height, width) > 50000:
        scale = max(height, width) / 50000
        height, width = int(height / scale), int(width / scale)
    resized_h = round(height / factor) * factor
    resized_w = round(width / factor) * factor
    if resized_h * resized_w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resized_h = math.floor(height / scale / factor) * factor
        resized_w = math.floor(width / scale / factor) * factor
    elif resized_h * resized_w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * scale / factor) * factor
        resized_w = math.ceil(width * scale / factor) * factor
    return resized_h, resized_w


class ODinW13Dataset(Dataset):
    """Load all validation images from the bundled ``odinw13_config.py``.

    Images retain their original pixels; the Qwen processor performs resizing.
    ``det`` contains the prompt and original geometry, using the common
    detection ``image_id`` and ``orig_size`` fields. Image IDs are local to
    each named sub-dataset, so metrics must group by ``dataset_name``.
    """

    def __init__(
        self,
        data_dir,
        config_filename="odinw13_config.py",
        min_pixels=768 * 32 * 32,
        max_pixels=12800 * 32 * 32,
        dataset_filter=None,
    ):
        self.data_dir = Path(data_dir)
        config = runpy.run_path(str(self.data_dir / config_filename))
        names = config["dataset_prefixes"]
        subsets = config["datasets"]
        if len(names) != len(subsets):
            raise ValueError("ODinW names and dataset configurations must align.")
        if dataset_filter is not None and set(dataset_filter) - set(names):
            raise ValueError(
                f"Unknown ODinW subsets: {set(dataset_filter) - set(names)}"
            )
        self.items = []
        self.per_dataset_counts = {}
        for name, subset in zip(names, subsets):
            if dataset_filter is not None and name not in dataset_filter:
                continue
            sub_root = self.data_dir / subset["data_root"].removeprefix("data/odinw/")
            annotation_path = sub_root / subset["ann_file"]
            annotations = json.loads(annotation_path.read_text())
            classes = list(subset["metainfo"]["classes"])
            prompt = (
                "Locate every instance that belongs to the following categories: "
                f"'{', '.join(classes)}'. Report bbox coordinates in JSON format."
            )
            images = sorted(annotations["images"], key=lambda image: image["id"])
            self.per_dataset_counts[name] = len(images)
            for image in images:
                image_path = (
                    sub_root / subset["data_prefix"]["img"] / image["file_name"]
                )
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                orig_size = (int(image["height"]), int(image["width"]))
                self.items.append(
                    {
                        "img_path": str(image_path),
                        "det": {
                            "dataset_name": name,
                            "image_id": int(image["id"]),
                            "anno_path": str(annotation_path),
                            "classes": classes,
                            "orig_size": orig_size,
                            "resized_size": _resized_size(
                                *orig_size, min_pixels, max_pixels
                            ),
                            "prompt": prompt,
                        },
                    }
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        with Image.open(item["img_path"]) as image:
            pixels = to_tensor(image.convert("RGB"))
        return {
            "img": pixels,
            "meta": {
                "img_path": item["img_path"],
                "ori_size": item["det"]["orig_size"],
            },
            "det": dict(item["det"]),
        }

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    """COCO detection subset for CTC eval (image-id file list)."""

    def __init__(
        self,
        root,
        img_path="val2017",
        file_list="coco_val_1k.txt",
        *,
        tasks: Optional[List[str]] = None,
        transform=None,
        **kwargs,
    ):
        self.root = Path(root)
        self.img_path = img_path
        self.file_list = file_list
        self.transform = transform
        self.tasks = tasks if tasks is not None else ["det"]
        from cofai.engine.evaluator import ALLOWED_KINDS

        bad = [t for t in self.tasks if str(t) not in ALLOWED_KINDS]
        if bad:
            raise ValueError(
                f"Invalid tasks for CocoDetectionDataset: {bad}. Allowed kinds: {sorted(ALLOWED_KINDS)}"
            )

        img_dir = self.root / self.img_path
        if not img_dir.is_dir():
            raise RuntimeError(f'Missing directory "{img_dir}"')

        list_path = self.root / self.file_list
        if not list_path.is_file():
            raise FileNotFoundError(f"file_list not found: {list_path}")
        names = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
        self.samples = []
        for name in names:
            stem = Path(name).stem
            path = img_dir / f"{stem}.jpg"
            if not path.is_file():
                raise FileNotFoundError(f"Missing COCO image: {path}")
            self.samples.append((int(stem), str(path)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_id, img_path = self.samples[index]
        img_pil = Image.open(img_path).convert("RGB")
        orig_h, orig_w = img_pil.size[1], img_pil.size[0]
        img = np.array(img_pil).astype(np.float32) / 255.0
        meta = {
            "img_path": img_path,
            "img_name": Path(img_path).stem,
            "img_size": (img.shape[0], img.shape[1]),
            "ori_size": (orig_h, orig_w),
            "image_id": image_id,
        }
        sample = {
            "img": img,
            "meta": meta,
        }
        if "det" in self.tasks:
            sample["det"] = {
                "image_id": image_id,
                "orig_size": (orig_h, orig_w),
            }
        if self.transform:
            sample = self.transform(sample)
        return sample

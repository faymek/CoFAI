import re
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor


class AGReID(Dataset):
    def __init__(self, directory, image_size):
        self.images = sorted(Path(directory).glob("*.jpg"))
        if not self.images:
            raise FileNotFoundError(f"No images in {directory}")
        pattern = re.compile(r"P(\d+)T(\d+)A(\d+)C(\d+)F(\d+)")
        records = [pattern.match(path.stem).groups() for path in self.images]
        self.pids = np.array([int("".join(record[:3])) for record in records])
        self.cameras = np.array([int(record[3]) for record in records])
        self.transform = Compose([Resize(tuple(image_size), interpolation=InterpolationMode.BICUBIC), ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as image:
            return self.transform(image.convert("RGB"))

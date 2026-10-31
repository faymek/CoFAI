from pathlib import Path

from torchvision.transforms import Compose

from cofai.datasets import SegmentationDataset
from cofai.transforms import PadToMultiple, ToTensor


def build_dataset(data_root, corruption_root, corruption, severity):
    directory = Path(corruption_root) / corruption / str(severity) / "validation"
    dataset = SegmentationDataset(
        root=data_root,
        img_path=str(directory),
        seg_map_path="annotations/validation",
        reduce_zero_label=True,
        transform=Compose([PadToMultiple(16), ToTensor(keys=["img", "semseg"])]),
    )
    if not len(dataset):
        raise FileNotFoundError(f"No images in {directory}")
    return dataset

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional


class ImageFolder(Dataset):
    """Load an image folder database.

    Training and testing image samples are respectively stored in separate
    directories:

        rootdir/
            train/
                img000.png
                img001.png
            test/
                img000.png
                img001.png

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): A function or transform that takes in a
            PIL image and returns a transformed version. Defaults to None.
        split (str): Split mode ('train' or 'val'). Defaults to "train".
    """

    def __init__(
        self,
        root,
        transform=None,
        split="train",
        *,
        return_dict: bool = False,
        tasks: Optional[List[str]] = None,
    ):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.rglob("*") if f.is_file())

        self.transform = transform
        self.return_dict = bool(return_dict)
        self.tasks = tasks
        if self.tasks is not None:
            from cofai.engine.evaluator import ALLOWED_KINDS

            bad = [t for t in self.tasks if str(t) not in ALLOWED_KINDS]
            if bad:
                raise ValueError(
                    f"Invalid tasks for ImageFolder: {bad}. Allowed kinds: {sorted(ALLOWED_KINDS)}"
                )

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            length (int): Number of image files in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """Get an image sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            img (PIL.Image.Image or torch.Tensor): The image. If transform is
                provided, returns the transformed version (typically a torch.Tensor).
                Otherwise, returns a PIL Image in RGB format.
        """
        img_path = self.samples[index]
        img_name = Path(img_path).stem
        img_pil = Image.open(img_path).convert("RGB")
        img_hw = (img_pil.size[1], img_pil.size[0])
        img_meta = {
            "img_path": str(img_path),
            "img_name": img_name,
            "img_size": img_hw,
            "ori_size": img_hw,
        }
        if self.return_dict:
            # transforms (PadToMultiple/ToTensor/...) expect ndarray HWC float32.
            img = (np.array(img_pil).astype(np.float32) / 255.0)
            out = {"img": img, "meta": img_meta}
            # Optionally expose rec GT for reconstruction eval.
            if self.tasks and "rec" in self.tasks:
                out["rec"] = img.copy()
            if self.transform:
                out = self.transform(out)
            return out
        img = img_pil
        if self.transform:
            img = self.transform(img)
        return img, img_meta


class ClassificationDataset(Dataset):
    """Image-folder classification in the dict shape as MMEngine style.

    With ``return_dict=True``: ``img`` is HWC ``float32``
    in ``[0, 1]``, ``meta`` carries ``ori_size`` / ``img_size`` before
    ``self.transform``, then ``sample = self.transform(sample)`` when set —
    same control flow as segmentation.

    (Deprecated) With ``return_dict=False``: returns ``(PIL RGB, meta)`` and applies
    ``transform`` to the PIL image only (legacy single-input ``torchvision``).
    """

    def __init__(
        self,
        root,
        transform=None,
        split="",
        file_list=None,
        labels_file=None,
        *,
        return_dict: bool = False,
        label_key: str = "scene",
        tasks: Optional[List[str]] = None,
        **kwargs,
    ):
        self.transform = transform
        self.root = Path(root)
        self.split = split
        self.file_list = file_list
        self.labels_file = labels_file
        self.return_dict = bool(return_dict)
        self.label_key = str(label_key)
        self.tasks = tasks
        if self.tasks is not None:
            from cofai.engine.evaluator import ALLOWED_KINDS

            bad = [t for t in self.tasks if str(t) not in ALLOWED_KINDS]
            if bad:
                raise ValueError(
                    f"Invalid tasks for ClassificationDataset: {bad}. Allowed kinds: {sorted(ALLOWED_KINDS)}"
                )

        # Determine data directory
        if self.split:
            self.data_dir = self.root / self.split
        else:
            self.data_dir = self.root

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f'Missing directory "{self.data_dir}"')

        # Load file list
        if self.file_list and (self.root / self.file_list).exists():
            with open(self.root / self.file_list, "r") as f:
                self.samples = [line.strip() for line in f.readlines()]
            # Ensure file paths are relative to data_dir
            self.samples = [str(self.data_dir / sample) for sample in self.samples]
            self.samples = sorted(self.samples)
        else:
            # If file_list is not specified, scan the directory
            self.samples = sorted(
                f
                for f in self.data_dir.rglob("*")
                if f.is_file()
                and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            )
            self.samples = [str(f) for f in self.samples]

        # Load label information
        if not self.labels_file:
            raise ValueError("labels_file is required for classification dataset")
        if not (self.root / self.labels_file).exists():
            raise FileNotFoundError(f"labels_file {self.labels_file} not found")
        self.labels_dict = {}
        with open(self.root / self.labels_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])
                    self.labels_dict[img_name] = label

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            length (int): Number of image files in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """See class docstring; ``meta`` matches segmentation conventions (H, W)."""
        img_path = self.samples[index]
        img_name = Path(img_path).stem

        img_pil = Image.open(img_path).convert("RGB")
        img = (np.array(img_pil).astype(np.float32) / 255.0)
        label = self.labels_dict.get(img_name, None)
        img_meta = {
            "img_path": img_path,
            "img_name": img_name,
            "img_size": (img.shape[0], img.shape[1]),
            "ori_size": (img.shape[0], img.shape[1]),
            "cls_label": label,
        }

        if self.return_dict:
            sample: dict = {"img": img, "meta": img_meta}
            if label is not None:
                if self.tasks is None:
                    sample[self.label_key] = int(label)
                else:
                    if "cls" in self.tasks:
                        sample["cls"] = int(label)
                    if "rec" in self.tasks:
                        sample["rec"] = img.copy()
            if self.transform:
                sample = self.transform(sample)
            return sample

        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, img_meta


class SegmentationDataset(Dataset):
    """
    Segmentation dataset in MMEngine style dict sample format.

    Returns a dict sample:
      - sample["img"]: HWC float32
      - sample["semseg"]: HW1 float32 (padding uses 255 ignore)
      - sample["meta"]:  metadata dict
    """

    def __init__(
        self,
        root,
        transform=None,
        img_path="JPEGImages",
        seg_map_path="SegmentationClass",
        file_list=None,
        reduce_zero_label=False,
        tasks: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.img_path = img_path
        self.seg_map_path = seg_map_path
        self.file_list = file_list
        self.reduce_zero_label = reduce_zero_label
        self.tasks = tasks if tasks is not None else ["semseg"]
        from cofai.engine.evaluator import ALLOWED_KINDS

        bad = [t for t in self.tasks if str(t) not in ALLOWED_KINDS]
        if bad:
            raise ValueError(
                f"Invalid tasks for SegmentationDataset: {bad}. Allowed kinds: {sorted(ALLOWED_KINDS)}"
            )

        img_dir = self.root / self.img_path
        if not img_dir.is_dir():
            raise RuntimeError(f'Missing directory "{img_dir}"')

        if self.file_list and (self.root / self.file_list).exists():
            with open(self.root / self.file_list, "r") as f:
                names = [line.strip() for line in f.readlines()]
            self.samples = [str(img_dir / f"{img_name}.jpg") for img_name in names]
        else:
            self.samples = sorted(str(f) for f in img_dir.rglob("*.jpg"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img_name = Path(img_path).stem

        img_pil = Image.open(img_path).convert("RGB")
        img = (np.array(img_pil).astype(np.float32) / 255.0)

        seg_label_path = str(self.root / self.seg_map_path / f"{img_name}.png")
        seg = np.array(Image.open(seg_label_path))
        if self.reduce_zero_label:
            seg = seg.astype(np.int64) - 1
            seg[seg == -1] = 255
        seg = seg.astype(np.float32)
        seg = np.expand_dims(seg, -1)  # HW -> HW1

        meta = {
            "img_path": img_path,
            "img_name": img_name,
            "img_size": (img.shape[0], img.shape[1]),
            "ori_size": (img.shape[0], img.shape[1]),
            "seg_label_path": seg_label_path,
        }

        sample = {"img": img, "meta": meta}
        if "semseg" in self.tasks:
            sample["semseg"] = seg
        if "rec" in self.tasks:
            sample["rec"] = img
        if self.transform:
            sample = self.transform(sample)
        return sample


class NYUDepthDataset(Dataset):
    """NYUD depth dataset adapter for the MPC evaluation script.

    Supports both the RFC/MT layout and the local NYU layout with ``nyu_test.txt``
    plus ``train``/``test`` folders containing ``rgb_*.jpg`` and
    ``sync_depth_*.png`` pairs.
        ├─ NYU/
│       ├─ train/
│       └─ test/
│       ├─ nyu_train.txt/
│       └─ nyu_test.txt/
    """

    def __init__(self, root, split="val", split_file=None, download=False, **kwargs):
        self.root = Path(root)
        self.dataset = None
        self.samples = None

        if (self.root / "gt_sets").is_dir() and (self.root / "scene_names.npy").is_file():
            from cofai.datasets.rfcdata.nyud import NYUD_MT

            self.dataset = NYUD_MT(
                root=root,
                split=split,
                download=download,
                transform=None,
                do_depth=True,
            )
            return

        split_name = "test" if split in ("val", "test") else split
        split_path = self.root / split_name
        list_path = self.root / split_file if split_file else self.root / f"nyu_{split_name}.txt"
        if not list_path.is_file() and split_file:
            list_path = self.root / Path(split_file).name

        samples = []
        if list_path.is_file():
            with open(list_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    img_rel, depth_rel = parts[0], parts[1]
                    img_path = split_path / img_rel
                    depth_path = split_path / depth_rel
                    if not img_path.is_file():
                        img_path = self.root / img_rel
                    if not depth_path.is_file():
                        depth_path = self.root / depth_rel
                    samples.append((img_path, depth_path))
        elif split_path.is_dir():
            for img_path in sorted(split_path.rglob("rgb_*.jpg")):
                depth_path = img_path.with_name(img_path.name.replace("rgb_", "sync_depth_").replace(".jpg", ".png"))
                if depth_path.is_file():
                    samples.append((img_path, depth_path))
        else:
            raise FileNotFoundError(
                f"Cannot find NYU split file or directory under {self.root}: "
                f"{list_path} / {split_path}"
            )

        if not samples:
            raise RuntimeError(f"No NYU RGB-depth pairs found under {self.root}")
        self.samples = samples

    def __len__(self):
        return len(self.dataset) if self.dataset is not None else len(self.samples)

    def __getitem__(self, index):
        if self.dataset is not None:
            sample = self.dataset[index]
            image_np = sample["image"]
            if image_np.dtype != np.uint8:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_np).convert("RGB")
            depth = sample["depth"]
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            meta = sample.get("meta", {})
            img_name = meta.get("img_name", str(index))
            img_path = str(self.dataset.images[index])
        else:
            img_path_obj, depth_path = self.samples[index]
            image = Image.open(img_path_obj).convert("RGB")
            depth = np.array(Image.open(depth_path), dtype=np.float32)
            img_name = img_path_obj.stem
            img_path = str(img_path_obj)

        return image, {
            "img_path": img_path,
            "img_name": img_name,
            "ori_size": image.size,
            "depth_label": depth.astype(np.float32, copy=False),
        }
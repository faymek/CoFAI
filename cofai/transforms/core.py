# same transform as ATRC

from typing import Any, Optional, Sequence

import numpy as np
import random
import cv2
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import CenterCrop as TvCenterCrop
from torchvision.transforms.v2 import Resize as TvResize


class RandomScaling:
    """Random scale the input.
    Args:
      min_scale_factor: Minimum scale value.
      max_scale_factor: Maximum scale value.
      step_size: The step size from minimum to maximum value.
    Returns:
        sample: The input sample scaled
    """

    def __init__(self, scale_factors=(0.5, 2.0), discrete=False, keys: Optional[Sequence[str]] = None):
        self.scale_factors = scale_factors
        self.discrete = discrete
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)
        self.mode = {
            'semseg': cv2.INTER_NEAREST,
            'seg': cv2.INTER_NEAREST,
            'depth': cv2.INTER_NEAREST,
            'normals': cv2.INTER_NEAREST,
            'edge': cv2.INTER_NEAREST,
            'sal': cv2.INTER_NEAREST,
            'human_parts': cv2.INTER_NEAREST,
            'rec': cv2.INTER_LINEAR,
            'rae': cv2.INTER_LINEAR,
            'img': cv2.INTER_LINEAR,
        }

    def get_scale_factor(self):
        if self.discrete:
            # choose one option out of the list
            random_scale = random.choice(self.scale_factors)
        else:
            assert len(self.scale_factors) == 2
            random_scale = random.uniform(*self.scale_factors)
        return random_scale

    def scale(self, key, unscaled, scale=1.0):
        """Randomly scales image and label.
        Args:
            key: Key indicating the uscaled input origin
            unscaled: Image or target to be scaled.
            scale: The value to scale image and label.
        Returns:
            scaled: The scaled image or target
        """
        # No random scaling if scale == 1.
        if scale == 1.0:
            return unscaled
        image_shape = np.shape(unscaled)[0:2]
        new_dim = tuple([int(x * scale) for x in image_shape])

        unscaled = np.squeeze(unscaled)
        scaled = cv2.resize(unscaled, new_dim[::-1], interpolation=self.mode[key])
        if scaled.ndim == 2:
            scaled = np.expand_dims(scaled, axis=2)

        if key == 'depth':
            # ignore regions for depth are 0
            scaled /= scale

        return scaled

    def __call__(self, sample):
        random_scale = self.get_scale_factor()
        for key in self._keys:
            if key not in sample:
                continue
            sample[key] = self.scale(key, sample[key], scale=random_scale)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self._keys!r})'


class ResizeToFit:
    """Resize image/targets to fit within a max canvas size.

    - Preserves aspect ratio.
    - Only resizes when input is larger than ``size`` in either dimension.
    - Uses bilinear for image and nearest for discrete targets.
    """

    def __init__(self, size, keys: Optional[Sequence[str]] = None):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            self.size = (int(size[0]), int(size[1]))
        else:
            raise ValueError("size must be int or (H, W)")
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)

        self.mode = {
            "semseg": cv2.INTER_NEAREST,
            "seg": cv2.INTER_NEAREST,
            "depth": cv2.INTER_NEAREST,
            "normals": cv2.INTER_NEAREST,
            "edge": cv2.INTER_NEAREST,
            "sal": cv2.INTER_NEAREST,
            "human_parts": cv2.INTER_NEAREST,
            "rec": cv2.INTER_LINEAR,
            "rae": cv2.INTER_LINEAR,
            "img": cv2.INTER_LINEAR,
        }

    def _resize(self, key, arr, new_hw):
        h2, w2 = int(new_hw[0]), int(new_hw[1])
        if h2 <= 0 or w2 <= 0:
            return arr
        a = np.squeeze(arr)
        out = cv2.resize(
            a,
            (w2, h2),
            interpolation=self.mode[key],
        )
        if out.ndim == 2:
            out = np.expand_dims(out, axis=2)
        return out

    def __call__(self, sample):
        img = sample.get("img", None)
        if img is None:
            return sample
        h, w = int(np.shape(img)[0]), int(np.shape(img)[1])
        max_h, max_w = int(self.size[0]), int(self.size[1])
        if h <= max_h and w <= max_w:
            return sample

        scale = min(max_h / float(h), max_w / float(w))
        new_h = max(int(round(h * scale)), 1)
        new_w = max(int(round(w * scale)), 1)

        for key in self._keys:
            if key not in sample:
                continue
            sample[key] = self._resize(key, sample[key], (new_h, new_w))
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, keys={self._keys!r})"


class ResizeImage:
    """Thin adapter over ``torchvision.transforms.v2.Resize``.

    Constructor arguments match TorchVision (``size``, ``interpolation``, ``max_size``,
    ``antialias``). The only CoFAI extra is ``keys``: the same resize is applied to
    each ``sample[key]``.

    Sample entries must be **HWC** ``numpy`` arrays (use ``float32`` for images).
    TorchVision operates on **CHW** tensors, so this class only converts layout and
    dtype; all scaling rules and errors come from ``TvResize``.
    """

    def __init__(
        self,
        size: Any,
        keys: Optional[Sequence[str]] = None,
        *,
        interpolation: Any = InterpolationMode.BILINEAR,
        max_size: Optional[int] = None,
        antialias: Optional[bool] = True,
    ):
        self._size = size
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)
        if isinstance(interpolation, str):
            interpolation = InterpolationMode[interpolation.upper()]
        self._resize = TvResize(
            size=size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=antialias,
        )

    def __call__(self, sample):
        for key in self._keys:
            if key not in sample:
                continue
            a = np.asarray(sample[key])
            if a.ndim == 2:
                a = a[..., np.newaxis]
            t = torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1)
            t = self._resize(t)
            sample[key] = (
                t.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32, copy=False)
            )
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self._size!r}, keys={self._keys!r})"


class CenterCropImage:
    """Thin adapter over ``torchvision.transforms.v2.CenterCrop``.

    Applies a center crop of the given ``size`` to each ``sample[key]``.
    Sample entries must be **HWC** ``numpy`` arrays.
    """

    def __init__(self, size: Any, keys: Optional[Sequence[str]] = None):
        self._size = size
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)
        self._crop = TvCenterCrop(size=size)

    def __call__(self, sample):
        for key in self._keys:
            if key not in sample:
                continue
            a = np.asarray(sample[key])
            if a.ndim == 2:
                a = a[..., np.newaxis]
            t = torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1)
            t = self._crop(t)
            sample[key] = (
                t.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32, copy=False)
            )
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self._size!r}, keys={self._keys!r})"


class PadImage:
    """Pad image and label to have dimensions >= [size_height, size_width]
    Args:
        size: Desired size
    Returns:
        sample: The input sample padded
    """

    def __init__(self, size, keys: Optional[Sequence[str]] = None):
        if isinstance(size, int):
            self.size = tuple([size, size])
        elif isinstance(size, (list, tuple)):
            self.size = size
        else:
            raise ValueError('Crop size must be an int, tuple or list')
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)
        self.fill_index = {'edge': 255,
                           'human_parts': 255,
                           'semseg': 255,
                           'seg': 255,  # alias for semseg (used by some DINO-style datasets)
                           'depth': 0,
                           'normals': [0, 0, 0],
                           'sal': 255,
                           'img': [0, 0, 0],
                           # reconstruction targets are image-like RGB arrays
                           'rec': [0, 0, 0],
                           'rae': [0, 0, 0]}

    def pad(self, key, unpadded):
        unpadded_shape = np.shape(unpadded)
        delta_height = max(self.size[0] - unpadded_shape[0], 0)
        delta_width = max(self.size[1] - unpadded_shape[1], 0)

        if delta_height == 0 and delta_width == 0:
            return unpadded

        # Location to place image
        height_location = [delta_height // 2,
                           (delta_height // 2) + unpadded_shape[0]]
        width_location = [delta_width // 2,
                          (delta_width // 2) + unpadded_shape[1]]

        pad_value = self.fill_index[key]
        max_height = max(self.size[0], unpadded_shape[0])
        max_width = max(self.size[1], unpadded_shape[1])

        padded = np.full((max_height, max_width, unpadded_shape[2]),
                        pad_value, dtype=np.float32)
        padded[height_location[0]:height_location[1],
            width_location[0]:width_location[1], :] = unpadded
        # else:
        #     padded = np.full((max_height, max_width),
        #                     pad_value, dtype=np.float32)
        #     padded[height_location[0]:height_location[1],
        #         width_location[0]:width_location[1]] = unpadded

        return padded

    def __call__(self, sample):
        for key in self._keys:
            if key not in sample:
                continue
            sample[key] = self.pad(key, sample[key])
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self._keys!r})"


class SetImageAsOriginal:
    """Set ``meta[\"ori_size\"]`` to current ``img`` spatial size ``(H, W)``."""

    def __call__(self, sample):
        if "img" in sample and "meta" in sample:
            h, w = sample["img"].shape[:2]
            sample["meta"]["ori_size"] = (int(h), int(w))
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PadToMultiple:
    """Center-pad so H and W become multiples of ``multiple`` (see ``fill_index`` per key).

    If ``keys`` is set, only those sample keys are padded (must exist). Otherwise only
    ``img`` is padded.
    """

    def __init__(self, multiple: int, keys: Optional[Sequence[str]] = None):
        self.multiple = int(multiple)
        if self.multiple <= 0:
            raise ValueError(f"multiple must be positive, got {multiple!r}")
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)
        self.fill_index = {'edge': 255,
                           'human_parts': 255,
                           'semseg': 255,
                           'seg': 255,  # alias for semseg (used by some DINO-style datasets)
                           'depth': 0,
                           'normals': [0, 0, 0],
                           'sal': 255,
                           'img': [0, 0, 0],
                           # reconstruction targets are image-like RGB arrays
                           'rec': [0, 0, 0],
                           'rae': [0, 0, 0]}

    def pad(self, key, unpadded):
        unpadded_shape = np.shape(unpadded)
        h, w = int(unpadded_shape[0]), int(unpadded_shape[1])
        new_h = (h + self.multiple - 1) // self.multiple * self.multiple
        new_w = (w + self.multiple - 1) // self.multiple * self.multiple
        delta_height = max(new_h - h, 0)
        delta_width = max(new_w - w, 0)
        if delta_height == 0 and delta_width == 0:
            return unpadded

        height_location = [delta_height // 2, (delta_height // 2) + h]
        width_location = [delta_width // 2, (delta_width // 2) + w]

        pad_value = self.fill_index[key]
        padded = np.full((new_h, new_w, unpadded_shape[2]), pad_value, dtype=np.float32)
        padded[height_location[0]:height_location[1],
               width_location[0]:width_location[1], :] = unpadded
        return padded

    def __call__(self, sample):
        ref = next((k for k in self._keys if k in sample), None)
        h0 = w0 = 0
        if ref is not None:
            h0, w0 = int(sample[ref].shape[0]), int(sample[ref].shape[1])
        for key in self._keys:
            if key not in sample:
                continue
            sample[key] = self.pad(key, sample[key])
        if ref is not None:
            meta = sample.setdefault("meta", {})
            h1, w1 = int(sample[ref].shape[0]), int(sample[ref].shape[1])
            dh, dw = h1 - h0, w1 - w0
            meta["valid_roi"] = {
                "top": dh // 2,
                "left": dw // 2,
                "height": h0,
                "width": w0,
            }
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(multiple={self.multiple}, keys={self._keys!r})"


class RandomCrop:
    """Random crop image if it exceeds desired size
    Args:
        size: Desired size
    Returns:
        sample: The input sample randomly cropped
    """

    def __init__(self, size, cat_max_ratio=1, keys: Optional[Sequence[str]] = None):
        if isinstance(size, int):
            self.size = tuple([size, size])
        elif isinstance(size, (list, tuple)):
            self.size = size
        else:
            raise ValueError('Crop size must be an int, tuple or list')
        self.cat_max_ratio = cat_max_ratio  # need semantic labels for this
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)

    def get_random_crop_loc(self, uncropped):
        """Gets a random crop location.
        Args:
            key: Key indicating the uncropped input origin
            uncropped: Image or target to be cropped.
        Returns:
            Cropping region.
        """
        uncropped_shape = np.shape(uncropped)
        img_height = uncropped_shape[0]
        img_width = uncropped_shape[1]

        crop_height = self.size[0]
        crop_width = self.size[1]
        if img_height == crop_height and img_width == crop_width:
            return None
        # Get random offset uniformly from [0, max_offset]
        max_offset_height = max(img_height - crop_height, 0)
        max_offset_width = max(img_width - crop_width, 0)

        offset_height = random.randint(0, max_offset_height)
        offset_width = random.randint(0, max_offset_width)
        crop_loc = [offset_height, offset_height + crop_height,
                    offset_width, offset_width + crop_width]

        return crop_loc

    def random_crop(self, key, uncropped, crop_loc):
        if crop_loc is None:
            return uncropped

        cropped = uncropped[crop_loc[0]:crop_loc[1],
                            crop_loc[2]:crop_loc[3], :]
        return cropped

    def __call__(self, sample):
        crop_location = self.get_random_crop_loc(sample["img"])
        if self.cat_max_ratio < 1.0 and "semseg" in sample and "semseg" in self._keys:
            for _ in range(10):
                seg_tmp = self.random_crop("semseg", sample["semseg"], crop_location)
                labels, cnt = np.unique(seg_tmp, return_counts=True)
                cnt = cnt[labels != 255]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_location = self.get_random_crop_loc(sample["img"])

        for key in self._keys:
            if key not in sample:
                continue
            sample[key] = self.random_crop(key, sample[key], crop_location)
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self._keys!r})"


class RandomHorizontalFlip:
    """Horizontally flip the given image and ground truth randomly."""

    def __init__(self, p=0.5, keys: Optional[Sequence[str]] = None):
        self.p = p
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)

    def __call__(self, sample):
        if random.random() < self.p:
            for key in self._keys:
                if key not in sample:
                    continue
                sample[key] = np.fliplr(sample[key]).copy()
                if key == "normals":
                    sample[key][:, :, 0] *= -1
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self._keys!r})"


class Normalize:
    """ Normalize image values by first mapping from [0, 255] to [0, 1] and then
    applying standardization.
    """

    def __init__(self, mean, std, keys: Optional[Sequence[str]] = None):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)

    def normalize_img(self, img):
        assert img.dtype == np.float32
        scaled = img.copy() / 255.
        scaled -= self.mean
        scaled /= self.std
        return scaled

    def __call__(self, sample):
        for key in self._keys:
            if key not in sample:
                continue
            sample[key] = self.normalize_img(sample[key])
        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, keys: Optional[Sequence[str]] = None):
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)

    def __call__(self, sample):
        for key in self._keys:
            if key not in sample:
                continue
            val = sample[key]
            sample[key] = torch.from_numpy(val.transpose((2, 0, 1))).float()
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self._keys!r})"


class AddIgnoreRegions:
    """Add Ignore Regions"""

    def __init__(self, keys: Optional[Sequence[str]] = None):
        self._keys: tuple[str, ...] = (
            ("normals", "human_parts") if keys is None else tuple(keys)
        )

    def __call__(self, sample):
        for elem in self._keys:
            if elem not in sample:
                continue
            tmp = sample[elem]
            if elem == "normals":
                norm = np.sqrt(tmp[:, :, 0] ** 2 + tmp[:, :, 1] ** 2 + tmp[:, :, 2] ** 2)
                tmp[norm == 0, :] = 255
                sample[elem] = tmp
            elif elem == "human_parts":
                if ((tmp == 0) | (tmp == 255)).all():
                    tmp = np.full(tmp.shape, 255, dtype=tmp.dtype)
                    sample[elem] = tmp
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(keys={self._keys!r})"


class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
        keys: Optional[Sequence[str]] = None,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self._keys: tuple[str, ...] = ("img",) if keys is None else tuple(keys)

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.random() < 0.5:
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.random() < 0.5:
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.random() < 0.5:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.random() < 0.5:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta - 1)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, sample):
        for key in self._keys:
            if key not in sample:
                continue
            img = sample[key].astype(np.uint8)
            img = self.brightness(img)
            f_mode = random.random() < 0.5
            if f_mode:
                img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
            if not f_mode:
                img = self.contrast(img)
            sample[key] = img.astype(np.float32)
        return sample

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, {self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, {self.saturation_upper}), "
            f"hue_delta={self.hue_delta}, keys={self._keys!r})"
        )

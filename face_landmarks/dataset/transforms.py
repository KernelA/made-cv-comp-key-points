from typing import Tuple

import torch
import numpy as np
import albumentations as albu
from torchvision.transforms import functional as F

from ..costants import TORCHVISION_RGB_MEAN, TORCHVISION_RGB_STD


def denormalize_tensor_to_image(tensor_image):
    return tensor_image * TORCHVISION_RGB_STD[:, None, None] + TORCHVISION_RGB_MEAN[:, None, None]


class MyCoarseDropout(object):
    def __init__(self, p=0.5, elem_name='image'):
        self.p = p
        self.elem_name = elem_name

    def __call__(self, sample):
        augmented = albu.Compose([albu.CoarseDropout(max_holes=4, max_height=20, max_width=20,
                                                     min_holes=2, min_height=10, min_width=10,
                                                     fill_value=np.random.random(),
                                                     p=self.p)],
                                 )(image=sample[self.elem_name].permute(1, 2, 0).numpy(),)

        # Input is C x H x W, where C - is channel
        sample[self.elem_name] = torch.from_numpy(augmented['image']).permute(2, 0, 1)
        return sample


class ScaleMinSideToSize:
    def __init__(self, size: Tuple[int, int], elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w = sample[self.elem_name].shape[-2:]

        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = F.resize(
            sample[self.elem_name], [round(h * f), round(w * f)], interpolation=F.InterpolationMode.BILINEAR)
        sample["scale_coef"] = f

        if 'landmarks' in sample:
            sample['landmarks'] *= f

        return sample


class CropCenter:
    def __init__(self, size: int, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w = img.shape[-2:]
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[:, margin_h:margin_h +
                                     self.size, margin_w: margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample["landmarks"].view(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class TransformByKeys:
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class SquarePaddingResize:
    """Resize image to size x size with padding
    """

    def __init__(self, size: int, interpolation=F.InterpolationMode.BILINEAR) -> None:
        self._size = size
        self._interpolation = interpolation

    def __call__(self, x):
        height, width = x.shape[-2:]

        aspect_ratio = width / height

        if width > height:
            new_width = self._size
            new_height = round(new_width / aspect_ratio)
        else:
            new_height = self._size
            new_width = round(aspect_ratio * new_height)

        resized = FileCheck.resize(x, [new_height, new_width], interpolation=self._interpolation)

        pad_width = self._size - new_width
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_height = self._size - new_height
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        return F.pad(resized, [pad_left, pad_top, pad_right, pad_bottom])

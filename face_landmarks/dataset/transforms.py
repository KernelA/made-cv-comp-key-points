from typing import Tuple

import torch
from torchvision.transforms import functional as F


class ScaleMinSideToSize(torch.nn.Module):
    def __init__(self, size: Tuple[int, int], elem_name='image'):
        super().__init__()
        self.size = size
        self.elem_name = elem_name

    def forward(self, sample):
        h, w, _ = sample[self.elem_name].shape

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


class CropCenter(torch.nn.Module):
    def __init__(self, size: int, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def forward(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
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


class TransformByKeys(torch.nn.Module):
    def __init__(self, transform, names):
        super().__init__()
        self.transform = transform
        self.names = set(names)

    def forward(self, sample):
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

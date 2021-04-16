
from torchvision.transforms import functional


class SquarePaddingResize:
    """Resize image to size x size with padding
    """

    def __init__(self, size: int, interpolation=functional.InterpolationMode.BILINEAR) -> None:
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

        resized = functional.resize(x, [new_height, new_width], interpolation=self._interpolation)

        pad_width = self._size - new_width
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_height = self._size - new_height
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        return functional.pad(resized, [pad_left, pad_top, pad_right, pad_bottom])

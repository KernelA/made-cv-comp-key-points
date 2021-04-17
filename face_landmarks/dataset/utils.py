import torch
from torchvision import io
from torchvision.io.image import ImageReadMode
from matplotlib import pyplot as plt


def read_image(image_path: str, read_mode=ImageReadMode.RGB):
    return io.read_image(image_path, read_mode)


def normalize_landmarks(landmarks_positions, image_width: int, image_height: int):
    """Return normalized landmark positions [0; 1]
    """
    norm_positions = landmarks_positions.to(torch.float32)
    norm_positions[:, 0] /= image_width
    norm_positions[:, 1] /= image_height
    return norm_positions


def denormalize_landmarks(norm_landmarks_positions, image_width: int, image_height: int):
    """Return landmark positions in image pixel coordinates
    """
    landmarks_positions = norm_landmarks_positions.clone()
    landmarks_positions[:, 0] *= image_width
    landmarks_positions[:, 1] *= image_height

    return torch.round(landmarks_positions)


def plot_landmarks(image_tensor, landmark_positions, figsize=(10, 10), markersize: int = 2):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.imshow(image_tensor)
    ax.plot(landmark_positions[:, 0], landmark_positions[:, 1], "g+", markersize=markersize)

    return fig

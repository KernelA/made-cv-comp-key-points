import torch
from torch import nn
from torch.nn import functional as F


class LandmarkPredictor(nn.Module):

    def __init__(self, *, backbone, emb_dim: int, num_landmarks: int, dropout_prob: float = 0.5, train_backbone: bool = True):
        super().__init__()

        self.regressor = backbone
        self.regressor.requires_grad_(train_backbone)

        linear_output = 1024

        self.regressor.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(emb_dim, linear_output),
            nn.Dropout(dropout_prob),
            nn.Linear(linear_output, num_landmarks * 2)
        )

    def forward(self, x):
        """Return predcited positions: batch_size x num_landmarks * 2

        Each row is: x0, y0, x1, y1, x2, y2, ...
        """
        images = x["image"]
        return self.regressor(images)

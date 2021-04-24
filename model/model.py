import torch
from torch import nn
from torch.nn import functional as F


class LandmarkPredictor(nn.Module):

    def __init__(self, *, backbone, emb_dim: int, num_landmarks: int, dropout_prob: float = 0.5,
                 train_backbone: bool = True):
        super().__init__()

        self.regressor = backbone
        self.regressor.requires_grad_(train_backbone)

        self.regressor.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.regressor.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout_prob),
            nn.Linear(2 * 2 * emb_dim, 2 * num_landmarks, bias=True)
        )

    def forward(self, x):
        """Return predcited positions: batch_size x num_landmarks * 2

        Each row is: x0, y0, x1, y1, x2, y2, ...
        """
        images = x["image"]
        return self.regressor(images)

import torch
from torch import nn
from torch.nn import functional as F


class LandmarkPredictor(nn.Module):

    def __init__(self, *, backbone, emb_dim: int, num_landmarks: int, dropout_prob: float = 0.5):
        super().__init__()

        self.backbone = backbone

        for param in self.backbone.parameters():
            param.required_grad = True

        linear_output = 1024

        self.regressor = nn.Sequential(
            nn.Linear(emb_dim, linear_output),
            nn.Dropout(dropout_prob),
            nn.Linear(linear_output, num_landmarks * 2)
        )

    def forward(self, x):
        """Return predcited positions: batch_size x num_landmarks * 2

        Each row is: x0, y0, x1, y1, x2, y2, ...
        """
        images = x["image"]
        conv_res = self.backbone(images)
        embeddings = F.adaptive_avg_pool2d(conv_res, 1).view(images.shape[0], -1)
        return torch.relu(self.regressor(embeddings))

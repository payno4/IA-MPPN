"""
This implementation is inspired by the approach described in:

Peter Pfeiffer, Johannes Lahann, Peter Fettke,
"Multivariate Business Process Representation Learning utilizing
Gramian Angular Fields and Convolutional Neural Networks",
arXiv:2106.08027, 2021.

URL: https://github.com/joLahann/mppn

"""


import torch
import torch.nn as nn
from config.model_configs.MPPN_configs import MPPNConfig

class BaseMPPN(nn.Module):

    def __init__(self, config: MPPNConfig, num_perspectives: int):
        super().__init__()
        self.num_perspectives = num_perspectives
        self.feature_size = config.feature_size
        representation_dim = config.representation_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=self.feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2) # original from Paper
            nn.AdaptiveAvgPool2d((1, 1)) # modified to adapt to different input sizes
        )

        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(num_perspectives * self.feature_size, num_perspectives * self.feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(num_perspectives * self.feature_size, num_perspectives * self.feature_size),
            nn.ReLU(inplace=True),
            nn.Linear(num_perspectives * self.feature_size, representation_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, P, H, W)
        x = x.transpose(0, 1)  # (P, B, H, W)

        per_view_feats = []
        for v in x:
            v = v.unsqueeze(1)          # (B, 1, H, W)
            v = self.cnn(v)                  # (B, feature_size, _, _ )
            v = v.flatten(1)            # (B, feature_size)
            per_view_feats.append(v)

        pooled = torch.cat(per_view_feats, dim=1)      # (B, P*feature_size)
        rep = self.mlp(pooled)                         # (B, representation_dim)
        return rep


class MPPNRegressor(nn.Module):

    def __init__(self, config: MPPNConfig, num_perspectives: int):
        super().__init__()
        feature_size = config.feature_size
        representation_dim = config.representation_dim

        self.backbone = BaseMPPN(
            num_perspectives=num_perspectives,
            config=config,
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(representation_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.backbone(x)
        y_hat = self.head(rep)
        return y_hat
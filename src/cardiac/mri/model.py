from __future__ import annotations

import torch
from torch import nn


class ConvBlock3D(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, pool: bool = True):
        layers = [
            nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2)))
        super().__init__(*layers)


class CardiacCNN3D(nn.Module):
    def __init__(self, num_classes: int = 5, in_channels: int = 2,
                 widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.5):
        super().__init__()
        blocks, c = [], in_channels
        for i, w in enumerate(widths):
            blocks.append(ConvBlock3D(c, w, pool=i < len(widths) - 1))
            c = w
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))

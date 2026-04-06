"""Baseline CNN for image straightening.

Lightweight custom conv stack — no pretrained weights, trains in minutes.
Outputs (sin θ, cos θ) for circular angle regression.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Simple conv-bn-relu stack → global avg pool → regression head.

    Args:
        image_size: Input spatial resolution (default 224).
    """

    def __init__(self, image_size: int = 480) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 → 32, stride 2 → 112x112
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            # Block 2: 32 → 64, stride 2 → 56x56
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            # Block 3: 64 → 128, stride 2 → 28x28
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            # Block 4: 128 → 256, stride 2 → 14x14
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            # Block 5: 256 → 256, stride 2 → 7x7
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            # Block 6: 256 → 512, stride 2 → 8x8 (for 480x480 input)
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 + 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),   # outputs: (sin θ, cos θ)
        )

    def forward(self, x: torch.Tensor, aspect: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        combined = torch.cat([x, aspect], dim=1)
        return self.head(combined)


if __name__ == "__main__":
    model = BaselineCNN()
    dummy = torch.randn(4, 3, 480, 480)
    aspects = torch.zeros(4, 1)
    out = model(dummy, aspects)
    print(f"Output shape: {out.shape}")   # (4, 2)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

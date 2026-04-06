"""EfficientNet-B0 backbone for image straightening.

Pretrained on ImageNet via timm. Replace BaselineCNN with this once
baseline MAE is established. Two-phase training: head-only first, then full.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class EfficientNetModel(nn.Module):
    """EfficientNet-B0 with a custom (sin θ, cos θ) regression head.

    Args:
        pretrained: Load ImageNet weights (default True).
        dropout: Dropout rate before final linear (default 0.3).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,       # remove classifier, keep feature extractor
        )
        in_features = self.backbone.num_features  # 1280 for B0

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features + 1, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 2),   # outputs: (sin θ, cos θ)
        )

    def forward(self, x: torch.Tensor, aspect: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, 1280)
        combined = torch.cat([features, aspect], dim=1)
        return self.head(combined)    # (B, 2)

    def freeze_backbone(self) -> None:
        """Phase 1: freeze backbone, train head only."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Phase 2: unfreeze all for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, head_lr: float, backbone_lr_scale: float = 0.1) -> list[dict]:
        """Return param groups with differential LR: backbone gets 10x lower LR."""
        return [
            {"params": self.backbone.parameters(), "lr": head_lr * backbone_lr_scale},
            {"params": self.head.parameters(),     "lr": head_lr},
        ]


if __name__ == "__main__":
    model = EfficientNetModel(pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

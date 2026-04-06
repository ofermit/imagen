"""Circular angle utilities: loss, metric, angle recovery, TTA."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def circular_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAE in degrees space — handles 0°/360° wrap-around correctly.
    
    Args:
        pred: (B, 2) tensor of predicted (sin θ, cos θ).
        target: (B, 2) tensor of ground-truth (sin θ, cos θ).

    Returns:
        Scalar loss in degrees.
    """
    pred_deg = sincos_to_deg(pred)
    target_deg = sincos_to_deg(target)
    diff = torch.abs(pred_deg - target_deg) % 360.0
    diff = torch.min(diff, 360.0 - diff)
    return diff.mean()


def circular_mse_loss(pred: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    """MSE in (sin, cos) space — handles 0°/360° wrap-around correctly.

    Args:
        pred: (B, 2) tensor of predicted (sin θ, cos θ).
        target_deg: (B, 2) tensor of ground-truth (sin θ, cos θ).

    Returns:
        Scalar loss.
    """
    return nn.functional.mse_loss(pred, target_deg)


def sincos_to_deg(sincos: torch.Tensor) -> torch.Tensor:
    """Convert (sin θ, cos θ) predictions back to degrees via atan2.

    Args:
        sincos: (B, 2) tensor [sin, cos].

    Returns:
        (B,) tensor of angles in degrees in [-180, 180].
    """
    return torch.rad2deg(torch.atan2(sincos[:, 0], sincos[:, 1]))


def circular_mae(pred_deg: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    """Mean absolute angular error, handling wrap-around.

    Example: |179° - (-179°)| = 2°, not 358°.
    """
    diff = torch.abs(pred_deg - target_deg) % 360
    diff = torch.min(diff, 360 - diff)
    return diff.mean()


# ── TTA ──────────────────────────────────────────────────────────────────────

_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
_TO_TENSOR = transforms.ToTensor()


from dataset import pad_to_square

def _pil_to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    img = pad_to_square(img)
    img = img.resize((size, size), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
    return _NORMALIZE(_TO_TENSOR(img))


@torch.no_grad()
def predict_with_tta(
    model: nn.Module,
    image_path,
    device: torch.device,
    image_size: int = 480,
) -> float:
    """Run TTA: original + horizontal flip, average sin/cos, return degrees.

    Horizontal flip negates the angle, so we negate the recovered sin before averaging.
    """
    model.eval()
    img = Image.open(image_path).convert("RGB")

    w, h = img.size
    aspect = torch.tensor([[np.log(float(w) / float(h))]], dtype=torch.float32, device=device)

    # Original
    t_orig = _pil_to_tensor(img, image_size).unsqueeze(0).to(device)
    pred_orig = model(t_orig, aspect)[0].cpu()   # (2,)

    # Horizontal flip — sin negates, cos stays
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    t_flip = _pil_to_tensor(img_flip, image_size).unsqueeze(0).to(device)
    pred_flip = model(t_flip, aspect)[0].cpu()
    pred_flip[0] = -pred_flip[0]         # undo the sign flip

    avg = (pred_orig + pred_flip) / 2.0
    angle = float(torch.rad2deg(torch.atan2(avg[0], avg[1])))
    return angle

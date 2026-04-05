"""Evaluation script for the Image Straightening task.

Usage:
    python eval_test.py --images_dir <path_to_images_folder>

Loads the best available checkpoint automatically:
    1. checkpoints/efficientnet_best.pth   (EfficientNet-B0, preferred)
    2. checkpoints/baseline_best.pth       (BaselineCNN, fallback)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from pathlib import Path

import pandas as pd
import torch

from circular import predict_with_tta

# ── Checkpoint auto-detection ─────────────────────────────────────────────────
_CHECKPOINT_PRIORITY = [
    ("efficientnet", Path("checkpoints/efficientnet_best.pth")),
    ("efficientnet", Path("checkpoints/efficientnet_phase1_best.pth")),
    ("baseline",     Path("checkpoints/baseline_best.pth")),
]
_IMAGE_SIZE = 480


def _load_model(device: torch.device):
    """Load the best available model and return (model, model_type)."""
    import torch.nn as nn

    for model_type, ckpt_path in _CHECKPOINT_PRIORITY:
        if not ckpt_path.exists():
            continue

        if model_type == "efficientnet":
            from efficientnet_model import EfficientNetModel
            model = EfficientNetModel(pretrained=False)
        else:
            from baseline_cnn import BaselineCNN
            model = BaselineCNN(image_size=_IMAGE_SIZE)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        print(f"Loaded {model_type} checkpoint: {ckpt_path} "
              f"(val MAE: {ckpt.get('val_mae', 'n/a')}°)")
        return model

    raise FileNotFoundError(
        "No checkpoint found. Run train.py first.\n"
        "Expected one of: " + ", ".join(str(p) for _, p in _CHECKPOINT_PRIORITY)
    )


def predict(image_paths: list[Path]) -> list[float]:
    """Predict the correction angle for a list of images.

    Args:
        image_paths: List of paths to .jpg image files.

    Returns:
        List of predicted correction angles in degrees.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(device)

    angles = []
    for i, path in enumerate(image_paths):
        angle = predict_with_tta(model, path, device, image_size=_IMAGE_SIZE)
        angles.append(angle)
        if (i + 1) % 100 == 0:
            print(f"  Predicted {i + 1}/{len(image_paths)} images...")

    return angles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} images in {images_dir}")

    angles = predict(image_paths)

    results = [{
        "image_name": p.name,
        "rotation_angle": a
    } for p, a in zip(image_paths, angles)]
    pd.DataFrame(results).to_csv("eval_test_result.csv", index=False)
    print(f"Saved predictions → eval_test_result.csv")


if __name__ == "__main__":
    main()

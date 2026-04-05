"""Dataset for image straightening task.

Loads images and their correction angles from a CSV file.
Returns image tensors and (sin(angle), cos(angle)) targets for circular regression.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(train: bool, image_size: int = 224) -> transforms.Compose:
    """Return augmentation pipeline.

    NOTE: No random rotation — it would corrupt the angle labels.
    Safe augmentations only: flips (with label correction), color jitter, blur.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class StraightenDataset(Dataset):
    """Image straightening dataset.

    Args:
        images_dir: Path to folder containing .jpg images.
        csv_path: Path to CSV with columns [image_name, correction_angle].
        train: If True, apply training augmentations.
        image_size: Resize target (default 224).
        use_hflip: If True, randomly flip horizontally and negate the angle.
    """

    def __init__(
        self,
        images_dir: str | Path,
        csv_path: str | Path,
        train: bool = True,
        image_size: int = 224,
        use_hflip: bool = True,
        use_synthetic_rotation: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.train = train
        self.use_hflip = use_hflip and train
        self.use_synthetic_rotation = use_synthetic_rotation and train
        self.transform = get_transforms(train, image_size)

        df = pd.read_csv(csv_path)
        self.image_names = df.iloc[:, 0].tolist()
        self.angles = df.iloc[:, 1].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images_dir / self.image_names[idx]
        image = Image.open(img_path).convert("RGB")
        angle = self.angles[idx]

        # Random horizontal flip — negate the angle to match
        if self.use_hflip and np.random.rand() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            angle = -angle

        if self.train and self.use_synthetic_rotation:
            extra_deg = np.random.uniform(-30, 30)
            image = image.rotate(-extra_deg, expand=False, fillcolor=(128, 128, 128))
            angle = angle + extra_deg
            angle = ((angle + 180) % 360) - 180  # keep in [-180, 180]

        image = self.transform(image)

        # Convert angle to (sin, cos) for circular regression
        rad = np.deg2rad(angle)
        target = torch.tensor([np.sin(rad), np.cos(rad)], dtype=torch.float32)

        return image, target


class InferenceDataset(Dataset):
    """Minimal dataset for inference — no labels, no augmentation."""

    def __init__(self, image_paths: list[Path], image_size: int = 480) -> None:
        self.image_paths = image_paths
        self.transform = get_transforms(train=False, image_size=image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.image_paths[idx].name

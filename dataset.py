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
from torchvision.transforms import functional as TF


def pad_to_square(img: Image.Image, fill: int = 128) -> Image.Image:
    """Pad the shorter side with gray so the image becomes square."""
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    pad_left   = (size - w) // 2
    pad_top    = (size - h) // 2
    pad_right  = size - w - pad_left
    pad_bottom = size - h - pad_top
    return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)


def get_transforms(train: bool, image_size: int = 480) -> transforms.Compose:
    """Return augmentation pipeline.

    NOTE: No random rotation — it would corrupt the angle labels.
    Safe augmentations only: flips (with label correction), color jitter, blur.
    """
    if train:
        return transforms.Compose([
            transforms.Lambda(pad_to_square),
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Lambda(pad_to_square),
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
        image_size: Resize target (default 480).
        use_hflip: If True, randomly flip horizontally and negate the angle.
    """

    def __init__(
        self,
        images_dir: str | Path,
        csv_path: str | Path,
        train: bool = True,
        image_size: int = 480,
        use_hflip: bool = True,
        use_synthetic_rotation: bool = False,
        device: torch.device | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.train = train
        self.use_hflip = use_hflip and train
        self.use_synthetic_rotation = use_synthetic_rotation and train
        self.device = device if device is not None else torch.device("cpu")

        df = pd.read_csv(csv_path)
        
        # Filter out images with absolute rotation < 0.5
        mask = df.iloc[:, 1].astype(float).abs() >= 0.5
        df = df[mask]
        
        self.image_names = df.iloc[:, 0].tolist()
        self.angles_list = df.iloc[:, 1].astype(float).tolist()

        print(f"Preloading {len(self.image_names)} images into {self.device} memory...")
        import sys
        try:
            from tqdm import tqdm
            iterator = tqdm(self.image_names, desc="Loading images", file=sys.stdout)
        except ImportError:
            iterator = self.image_names

        images = []
        aspects = []
        for name in iterator:
            img_path = self.images_dir / name
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            aspect = float(w) / float(h)
            aspects.append([np.log(aspect)])

            img = pad_to_square(img)
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR) if hasattr(Image, "Resampling") else img.resize((image_size, image_size), Image.BILINEAR)
            # Store as uint8 to save memory (3.1 GB vs 12.4 GB)
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(device=self.device, dtype=torch.uint8)
            images.append(tensor)

        self.images = torch.stack(images)
        self.aspects = torch.tensor(aspects, dtype=torch.float32, device=self.device)
        self.angles = torch.tensor(self.angles_list, dtype=torch.float32, device=self.device)

        from torchvision.transforms import v2
        if self.train:
            self.color_jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            self.blur = v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert uint8 -> float32 and scale to [0, 1]
        image = self.images[idx].float() / 255.0
        angle = self.angles[idx].item()
        aspect_feat = self.aspects[idx]

        from torchvision.transforms import functional as F

        # Random horizontal flip — negate the angle to match
        if self.use_hflip and torch.rand(1).item() > 0.5:
            image = F.hflip(image)
            angle = -angle

        if self.train and self.use_synthetic_rotation:
            extra_deg = torch.empty(1).uniform_(-3.0, 3.0).item()
            image = F.rotate(image, -extra_deg, fill=[0.5, 0.5, 0.5])
            angle = angle + extra_deg
            angle = ((angle + 180) % 360) - 180  # keep in [-180, 180]

        if self.train:
            image = self.color_jitter(image)
            image = self.blur(image)

        image = self.normalize(image)

        # Convert angle to (sin, cos) for circular regression
        rad = np.deg2rad(angle)
        target = torch.tensor([np.sin(rad), np.cos(rad)], dtype=torch.float32, device=self.device)

        return image, aspect_feat, target


class InferenceDataset(Dataset):
    """Minimal dataset for inference — no labels, no augmentation."""

    def __init__(self, image_paths: list[Path], image_size: int = 480) -> None:
        self.image_paths = image_paths
        self.transform = get_transforms(train=False, image_size=image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        w, h = img.size
        aspect_feat = torch.tensor([np.log(float(w) / float(h))], dtype=torch.float32)
        return self.transform(img), aspect_feat, self.image_paths[idx].name

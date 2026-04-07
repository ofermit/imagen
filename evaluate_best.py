import torch
from train import get_device, val_epoch, DataLoader, StraightenDataset
from efficientnet_model import EfficientNetModel

device = get_device()
full_dataset = StraightenDataset("data/images", "data/ground_truth.csv", train=False, image_size=480, device=device)
val_size  = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(full_dataset), generator=generator).tolist()

from torch.utils.data import Subset
import copy
val_dataset = copy.copy(full_dataset)
val_dataset.train = False
val_ds = Subset(val_dataset, indices[train_size:])
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

model = EfficientNetModel(pretrained=False).to(device)
ckpt = torch.load("checkpoints/efficientnet_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
val_epoch(model, val_loader, device)

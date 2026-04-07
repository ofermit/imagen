import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
from datasets import load_dataset

from efficientnet_model import EfficientNetModel
from circular import circular_mae_loss, circular_mae, sincos_to_deg
from dataset import pad_to_square
from train import get_device

class PretrainDataset(Dataset):
    def __init__(self, hf_dataset, image_size=480):
        self.dataset = hf_dataset
        self.image_size = image_size
        from torchvision.transforms import v2
        self.color_jitter = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        self.blur = v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image'].convert('RGB')
        
        # Original aspect
        w, h = img.size
        aspect_feat = torch.tensor([np.log(float(w) / float(h))], dtype=torch.float32)

        # Pad and resize
        img = pad_to_square(img)
        img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
        image = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        # Random synthetic rotation for pretraining (-45 to 45)
        angle = torch.empty(1).uniform_(-45.0, 45.0).item()
        
        # Horizontal flip chance
        if torch.rand(1).item() > 0.5:
            image = F.hflip(image)
            angle = -angle

        image = F.rotate(image, -angle, fill=[0.5, 0.5, 0.5])
        
        image = self.color_jitter(image)
        image = self.blur(image)
        image = self.normalize(image)

        rad = np.deg2rad(angle)
        target = torch.tensor([np.sin(rad), np.cos(rad)], dtype=torch.float32)

        return image, aspect_feat, target

def pretrain():
    device = get_device()
    print("Loading cats_vs_dogs dataset from HuggingFace...")
    # Load dataset (trust_remote_code might be needed depending on the dataset)
    ds = load_dataset("cats_vs_dogs", trust_remote_code=True)
    
    # Use 90% train, 10% val
    split = ds['train'].train_test_split(test_size=0.1, seed=42)
    
    train_ds = PretrainDataset(split['train'])
    val_ds = PretrainDataset(split['test'])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    
    model = EfficientNetModel(pretrained=True).to(device)
    
    print("Starting pretraining on cats_vs_dogs (-45° to 45° rotation)...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_mae = math.inf
    epochs = 50
    
    Path("checkpoints").mkdir(exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for i, (images, aspects, targets) in enumerate(train_loader):
            images, aspects, targets = images.to(device), aspects.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    preds = model(images, aspects)
                    loss = circular_mae_loss(preds, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images, aspects)
                loss = circular_mae_loss(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            total_loss += loss.item() * len(images)
            
            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_loader)} | Running Loss: {total_loss / ((i+1)*images.size(0)):.4f}")
            
        train_loss = total_loss / len(train_ds)
        
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for images, aspects, targets in val_loader:
                images, aspects, targets = images.to(device), aspects.to(device), targets.to(device)
                preds = model(images, aspects)
                all_pred.append(sincos_to_deg(preds))
                all_true.append(sincos_to_deg(targets))
                
        pred_cat = torch.cat(all_pred)
        true_cat = torch.cat(all_true)
        val_mae = circular_mae(pred_cat, true_cat).item()
        
        print(f"[Pretrain] Epoch {epoch:03d}/{epochs} | loss {train_loss:.4f} | val MAE {val_mae:.3f}°")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_mae": val_mae,
            }, "checkpoints/efficientnet_pretrained.pth")
            print("  Saved best pretrain checkpoint.")

if __name__ == "__main__":
    pretrain()

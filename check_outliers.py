import torch
import pandas as pd
from pathlib import Path
import numpy as np

from train import get_device, DataLoader, StraightenDataset, val_epoch
from efficientnet_model import EfficientNetModel
from circular import sincos_to_deg

def analyze_outliers():
    device = get_device()
    
    # Load dataset (using the same filtered logic if applicable, or full to see everything)
    full_dataset = StraightenDataset(
        "data/images", "data/ground_truth.csv", train=False, image_size=480, device=device
    )
    
    # We want to look at the validation split we used
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
    # Use our best baseline or efficientnet model
    ckpt_path = Path("checkpoints/efficientnet_best.pth")
    if not ckpt_path.exists():
        print("Model checkpoint not found!")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    print("Running predictions on validation set to find outliers...")
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (images, aspects, targets) in enumerate(val_loader):
            images, aspects, targets = images.to(device), aspects.to(device), targets.to(device)
            preds = model(images, aspects)
            
            pred_deg = sincos_to_deg(preds)
            true_deg = sincos_to_deg(targets)
            
            # Calculate absolute error handling wrap around
            diff = torch.abs(pred_deg - true_deg) % 360.0
            error = torch.min(diff, 360.0 - diff)
            
            # Map batch indices to original dataset indices
            for i in range(len(error)):
                global_idx = indices[train_size + batch_idx * 32 + i]
                img_name = full_dataset.image_names[global_idx]
                results.append({
                    "image_name": img_name,
                    "true_angle": true_deg[i].item(),
                    "pred_angle": pred_deg[i].item(),
                    "error": error[i].item()
                })
                
    df = pd.DataFrame(results)
    df = df.sort_values(by="error", ascending=False)
    
    print("\n--- TOP 10 OUTLIERS (Highest Error) ---")
    print(df.head(10).to_string(index=False))
    
    df.to_csv("outliers_analysis.csv", index=False)
    print("\nSaved full analysis to outliers_analysis.csv")

if __name__ == "__main__":
    analyze_outliers()

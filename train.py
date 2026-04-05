"""Training script for image straightening.

Supports both BaselineCNN and EfficientNet-B0.
EfficientNet uses two-phase training: head-only → full fine-tune.

Usage:
    # Baseline
    python train.py --images_dir data/images --csv_path data/ground_truth.csv --model baseline

    # EfficientNet (recommended)
    python train.py --images_dir data/images --csv_path data/ground_truth.csv --model efficientnet
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import StraightenDataset
from baseline_cnn import BaselineCNN
from efficientnet_model import EfficientNetModel
from circular import circular_mae, circular_mse_loss, sincos_to_deg


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(model: nn.Module, path: Path, epoch: int, val_mae: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_mae": val_mae,
    }, path)
    print(f"  Saved checkpoint → {path}  (val MAE: {val_mae:.3f}°)")


# ── Epoch loops ───────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = circular_mse_loss(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    all_pred, all_true = [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        pred_deg = sincos_to_deg(preds)
        true_deg = sincos_to_deg(targets)
        all_pred.append(pred_deg)
        all_true.append(true_deg)
    pred_cat = torch.cat(all_pred)
    true_cat = torch.cat(all_true)
    # In val_epoch, also compute what "always predict 0" would score
    zero_mae = circular_mae(torch.zeros_like(pred_cat), true_cat)
    print(f"  Zero-prediction MAE: {zero_mae:.3f}°")
    return circular_mae(pred_cat, true_cat).item()


# ── Training phases ───────────────────────────────────────────────────────────

def run_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    phase_name: str,
    checkpoint_path: Path,
) -> float:
    best_mae = math.inf
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_mae = val_epoch(model, val_loader, device)
        scheduler.step()
        print(f"[{phase_name}] Epoch {epoch:03d}/{epochs} | "
              f"loss {train_loss:.4f} | val MAE {val_mae:.3f}°")
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(model, checkpoint_path, epoch, val_mae)
    return best_mae


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--csv_path",   type=str, required=True)
    parser.add_argument("--model",      type=str, default="efficientnet",
                        choices=["baseline", "efficientnet"])
    parser.add_argument("--image_size", type=int, default=480)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split",  type=float, default=0.1)
    parser.add_argument("--seed",       type=int, default=42)
    # Phase 1 (head only, EfficientNet) / sole phase (baseline)
    parser.add_argument("--epochs_phase1", type=int, default=5)
    parser.add_argument("--lr_phase1",     type=float, default=1e-3)
    # Phase 2 (full fine-tune, EfficientNet only)
    parser.add_argument("--epochs_phase2", type=int, default=25)
    parser.add_argument("--lr_phase2",     type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume",        type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--use_synthetic_rotation", action="store_true", help="Add synthetic rotation augmentation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    # ── Dataset & splits ──
    full_dataset = StraightenDataset(
        args.images_dir, args.csv_path, train=True, image_size=args.image_size,
        use_synthetic_rotation=args.use_synthetic_rotation
    )
    val_size  = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    # Val split should not use training augmentations
    val_ds.dataset.train = False

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    ckpt_dir = Path(args.checkpoint_dir)

    # ── Model ──
    if args.model == "baseline":
        model = BaselineCNN(image_size=args.image_size).to(device)
        print(f"Baseline CNN | params: {sum(p.numel() for p in model.parameters()):,}")

        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint from {args.resume}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_phase1, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_phase1
        )
        run_phase(model, train_loader, val_loader, optimizer, scheduler, device,
                  args.epochs_phase1, "Baseline", ckpt_dir / "baseline_best.pth")

    else:  # efficientnet — two phases
        model = EfficientNetModel(pretrained=True).to(device)
        print(f"EfficientNet-B0 | params: {sum(p.numel() for p in model.parameters()):,}")

        if args.resume:
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint from {args.resume}")

        # ── Phase 1: head only ──
        print("\n── Phase 1: head-only training ──")
        model.freeze_backbone()
        optimizer1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_phase1, weight_decay=1e-4
        )
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=args.epochs_phase1
        )
        run_phase(model, train_loader, val_loader, optimizer1, scheduler1, device,
                  args.epochs_phase1, "Phase1", ckpt_dir / "efficientnet_phase1_best.pth")

        # Load best phase-1 weights before phase 2
        ckpt = torch.load(ckpt_dir / "efficientnet_phase1_best.pth", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded phase-1 best (val MAE: {ckpt['val_mae']:.3f}°)")

        # ── Phase 2: full fine-tune ──
        print("\n── Phase 2: full fine-tune ──")
        model.unfreeze_backbone()
        optimizer2 = torch.optim.AdamW(
            model.get_param_groups(head_lr=args.lr_phase2, backbone_lr_scale=0.1),
            weight_decay=1e-4
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=args.epochs_phase2
        )
        run_phase(model, train_loader, val_loader, optimizer2, scheduler2, device,
                  args.epochs_phase2, "Phase2", ckpt_dir / "efficientnet_best.pth")


if __name__ == "__main__":
    main()

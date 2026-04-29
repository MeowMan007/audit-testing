"""
Training Script — Train the AccessibilityNet model on the synthetic dataset.

TRAINING PROCESS (Requirement 4):
    1. Load synthetic dataset (500 samples, 80/10/10 split)
    2. Apply data augmentation (RandomFlip, ColorJitter, RandomRotation)
    3. Fine-tune EfficientNet-B0 with BCEWithLogitsLoss
    4. Use AdamW optimizer with cosine annealing LR schedule
    5. Track per-class F1/Precision/Recall metrics
    6. Save best model checkpoint (by validation F1)

USAGE:
    Standalone:
        python -m backend.ml.train --dataset ./dataset --epochs 25 --batch-size 32
    
    From Colab:
        from backend.ml.train import train_model
        history = train_model(dataset_dir="./dataset", epochs=25)

EXPECTED RESULTS (on synthetic data):
    - Training converges in ~15-20 epochs
    - Validation F1: >0.80 per class
    - Best model saved as: dataset/accessibility_model.pth
"""
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from backend.ml.model import AccessibilityNet, get_model, VIOLATION_CLASSES, NUM_CLASSES

logger = logging.getLogger(__name__)


# ============================================================
# Custom PyTorch Dataset
# ============================================================

class AccessibilityDataset(Dataset):
    """PyTorch Dataset for accessibility violation screenshots.
    
    Loads images and multi-hot label vectors from the generated dataset.
    Applies configurable data augmentation transforms.
    
    Args:
        metadata: List of sample dictionaries from metadata.json
        images_dir: Path to the images directory
        transform: torchvision transforms to apply
    """

    def __init__(
        self,
        metadata: List[Dict],
        images_dir: str,
        transform=None,
    ):
        self.metadata = metadata
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.metadata[idx]
        
        # Load image
        img_path = self.images_dir / sample["image"].replace("images/", "")
        if not img_path.exists():
            # Fallback: try full path
            img_path = Path(sample["image"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Create a blank image as fallback
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        # Multi-hot label vector
        labels = torch.tensor(sample["labels"], dtype=torch.float32)

        return image, labels


# ============================================================
# Data Augmentation
# ============================================================

def get_transforms(is_training: bool = True) -> transforms.Compose:
    """Get data augmentation transforms.
    
    Training:  Resize + RandomFlip + ColorJitter + RandomRotation + Normalize
    Validation: Resize + Normalize (no augmentation)
    
    Uses ImageNet normalization (mean/std) since we're fine-tuning
    a model pretrained on ImageNet.
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],    # ImageNet std
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


# ============================================================
# Metrics Calculation
# ============================================================

def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict:
    """Compute per-class and overall metrics for multi-label classification.
    
    Metrics computed:
    - Per-class: Precision, Recall, F1-score
    - Overall: Macro F1, Hamming Loss, Exact Match Accuracy
    
    Args:
        predictions: Raw logits (batch, num_classes)
        targets: Ground truth labels (batch, num_classes)
        threshold: Confidence threshold for positive prediction
        
    Returns:
        Dictionary with all computed metrics
    """
    with torch.no_grad():
        probs = torch.sigmoid(predictions)
        preds = (probs >= threshold).float()

        metrics = {}
        f1_scores = []

        for i, cls_name in enumerate(VIOLATION_CLASSES):
            tp = ((preds[:, i] == 1) & (targets[:, i] == 1)).sum().item()
            fp = ((preds[:, i] == 1) & (targets[:, i] == 0)).sum().item()
            fn = ((preds[:, i] == 0) & (targets[:, i] == 1)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[f"{cls_name}_precision"] = precision
            metrics[f"{cls_name}_recall"] = recall
            metrics[f"{cls_name}_f1"] = f1
            f1_scores.append(f1)

        # Overall metrics
        metrics["macro_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        # Hamming loss: fraction of incorrect labels
        metrics["hamming_loss"] = (preds != targets).float().mean().item()
        
        # Exact match: all labels must be correct
        metrics["exact_match"] = (preds == targets).all(dim=1).float().mean().item()

        return metrics


# ============================================================
# Training Loop
# ============================================================

def train_model(
    dataset_dir: str = "dataset",
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    freeze_backbone: bool = False,
    device: Optional[str] = None,
) -> Dict:
    """Train the AccessibilityNet model.
    
    Full training pipeline:
    1. Load dataset from generated metadata.json + splits.json
    2. Create train/val DataLoaders with augmentation
    3. Initialize model, loss function, optimizer, scheduler
    4. Train for specified epochs, tracking metrics
    5. Save best model checkpoint (by validation macro F1)
    
    Args:
        dataset_dir: Path to the dataset directory
        epochs: Number of training epochs
        batch_size: Batch size for DataLoader
        learning_rate: Initial learning rate for AdamW
        weight_decay: L2 regularization strength
        freeze_backbone: Whether to freeze EfficientNet backbone
        device: 'cuda', 'cpu', or None (auto-detect)
        
    Returns:
        Training history dict with per-epoch metrics
    """
    dataset_dir = Path(dataset_dir)
    
    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # ---- Load dataset ----
    with open(dataset_dir / "metadata.json") as f:
        full_metadata = json.load(f)
    
    with open(dataset_dir / "splits.json") as f:
        splits = json.load(f)

    all_samples = full_metadata["samples"]
    train_samples = [all_samples[i] for i in splits["train"]]
    val_samples = [all_samples[i] for i in splits["val"]]

    logger.info(f"Train: {len(train_samples)} samples, Val: {len(val_samples)} samples")

    # ---- DataLoaders ----
    train_dataset = AccessibilityDataset(
        train_samples, dataset_dir / "images", get_transforms(is_training=True)
    )
    val_dataset = AccessibilityDataset(
        val_samples, dataset_dir / "images", get_transforms(is_training=False)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ---- Model ----
    model = get_model(pretrained=True, freeze_backbone=freeze_backbone).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Loss, Optimizer, Scheduler ----
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ---- Training History ----
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1": [], "val_f1": [],
        "best_epoch": 0, "best_val_f1": 0.0,
        "per_class_f1": [],
    }

    best_val_f1 = 0.0
    model_save_path = dataset_dir / "accessibility_model.pth"

    # ---- Training Loop ----
    for epoch in range(epochs):
        # --- Train Phase ---
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_targets = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            all_train_preds.append(outputs.detach().cpu())
            all_train_targets.append(labels.detach().cpu())

        train_loss /= len(train_dataset)
        train_preds = torch.cat(all_train_preds)
        train_targets = torch.cat(all_train_targets)
        train_metrics = compute_metrics(train_preds, train_targets)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                all_val_preds.append(outputs.cpu())
                all_val_targets.append(labels.cpu())

        val_loss /= len(val_dataset)
        val_preds = torch.cat(all_val_preds)
        val_targets = torch.cat(all_val_targets)
        val_metrics = compute_metrics(val_preds, val_targets)

        # Update scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["macro_f1"])
        history["val_f1"].append(val_metrics["macro_f1"])
        history["per_class_f1"].append({
            cls: val_metrics[f"{cls}_f1"] for cls in VIOLATION_CLASSES
        })

        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            history["best_epoch"] = epoch + 1
            history["best_val_f1"] = best_val_f1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_loss": val_loss,
                "class_names": VIOLATION_CLASSES,
                "num_classes": NUM_CLASSES,
            }, model_save_path)
            logger.info(f"  ★ Best model saved (F1: {best_val_f1:.4f})")

        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train F1: {train_metrics['macro_f1']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # ---- Final Summary ----
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete!")
    logger.info(f"Best Epoch: {history['best_epoch']} | Best Val F1: {history['best_val_f1']:.4f}")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"\nPer-class F1 (best epoch):")
    best_per_class = history["per_class_f1"][history["best_epoch"] - 1]
    for cls, f1 in best_per_class.items():
        logger.info(f"  {cls:20s}: {f1:.4f}")
    logger.info(f"{'='*60}")

    # Save training history
    history_path = dataset_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = argparse.ArgumentParser(description="Train AccessibilityNet model")
    parser.add_argument("--dataset", default="dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--freeze", action="store_true", help="Freeze backbone")
    args = parser.parse_args()

    train_model(
        dataset_dir=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_backbone=args.freeze,
    )

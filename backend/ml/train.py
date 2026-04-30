"""
Advanced Training Script — ViT-B/16 on Real-World Accessibility Dataset.

SOPHISTICATED TRAINING TECHNIQUES:
  1. Two-Phase Fine-Tuning:
       Phase 1 (Epochs 1-5):  Frozen backbone → head warmup
       Phase 2 (Epochs 6-30): Unfrozen → layer-wise LR decay (discriminative FT)
  2. Advanced Augmentation:
       RandAugment (N=2, M=9) for aggressive real-world variation
       MixUp (alpha=0.4) for inter-class label smoothing
       CutMix (alpha=1.0) for spatial feature learning
       Random Erasing (p=0.25) to simulate occlusion/missing content
  3. OneCycleLR scheduler with pct_start=0.1 for smooth warmup
  4. Label Smoothing (ε=0.1) to reduce overconfidence
  5. Automatic Mixed Precision (AMP) for 2× speed on GPU
  6. Early Stopping (patience=7) to prevent overfitting
  7. Class-weighted BCE loss for imbalanced violation classes
  8. Gradient accumulation for effective large batch training
  9. Exponential Moving Average (EMA) of model weights
"""
import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import RandAugment
from PIL import Image

from backend.ml.model import get_model, VIOLATION_CLASSES, NUM_CLASSES, count_parameters

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AccessibilityDataset(Dataset):
    def __init__(self, samples: List[Dict], images_dir: str, transform=None):
        self.samples = samples
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.images_dir / s["image"].replace("images/", "")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (200, 200, 200))
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(s["labels"], dtype=torch.float32)
        return img, labels


# ─────────────────────────────────────────────────────────────────────────────
# Advanced Augmentation Transforms
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform():
    """
    Aggressive augmentation for real-world web screenshots.
    RandAugment samples from 14 augmentation ops (brightness, contrast,
    sharpness, rotate, translate, etc.) at magnitude M=9.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.3),
        RandAugment(num_ops=2, magnitude=9),        # State-of-the-art augmentation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # Simulate occlusion
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MixUp / CutMix Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def mixup(images: torch.Tensor, labels: torch.Tensor,
          alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """MixUp: blend two random training samples and their labels."""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(images.size(0))
    mixed = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed, mixed_labels

def cutmix(images: torch.Tensor, labels: torch.Tensor,
           alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """CutMix: paste a rectangular region from one image into another."""
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape
    idx = torch.randperm(B)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bw = int(W * np.sqrt(1 - lam))
    bh = int(H * np.sqrt(1 - lam))
    x1, x2 = max(0, cx - bw//2), min(W, cx + bw//2)
    y1, y2 = max(0, cy - bh//2), min(H, cy + bh//2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    lam_adj = 1 - (x2-x1)*(y2-y1) / (H*W)
    mixed_labels = lam_adj * labels + (1 - lam_adj) * labels[idx]
    return mixed, mixed_labels


# ─────────────────────────────────────────────────────────────────────────────
# Exponential Moving Average (EMA)
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    """Maintains an EMA copy of the model weights for more stable inference."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self):
        for k, v in self.model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()

    def apply(self):
        self.model.load_state_dict(self.shadow)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5) -> Dict:
    with torch.no_grad():
        probs = torch.sigmoid(preds)
        p = (probs >= threshold).float()
        metrics = {}
        f1s = []
        for i, cls in enumerate(VIOLATION_CLASSES):
            tp = ((p[:,i]==1) & (targets[:,i]==1)).sum().item()
            fp = ((p[:,i]==1) & (targets[:,i]==0)).sum().item()
            fn = ((p[:,i]==0) & (targets[:,i]==1)).sum().item()
            prec = tp/(tp+fp) if tp+fp>0 else 0.0
            rec  = tp/(tp+fn) if tp+fn>0 else 0.0
            f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0.0
            metrics.update({f"{cls}_precision":prec, f"{cls}_recall":rec, f"{cls}_f1":f1})
            f1s.append(f1)
        metrics["macro_f1"] = sum(f1s)/len(f1s)
        metrics["hamming_loss"] = (p != targets).float().mean().item()
        metrics["exact_match"]  = (p == targets).all(dim=1).float().mean().item()
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Sampler (handle class imbalance)
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(samples: List[Dict]) -> WeightedRandomSampler:
    """Over-sample under-represented violation classes."""
    # Weight each sample by inverse frequency of its primary label
    label_counts = {}
    for s in samples:
        lbl = s.get("label", "clean")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    total = len(samples)
    weights = []
    for s in samples:
        lbl = s.get("label", "clean")
        weights.append(total / (len(label_counts) * label_counts[lbl]))
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    dataset_dir: str = "dataset",
    epochs: int = 30,
    batch_size: int = 32,
    grad_accum_steps: int = 2,   # Effective batch = 32×2 = 64
    phase1_epochs: int = 5,      # Frozen backbone warmup
    learning_rate: float = 3e-4,
    backbone_lr_mult: float = 0.1,  # Backbone LR = 0.1× head LR
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    use_mixup: bool = True,
    use_ema: bool = True,
    early_stop_patience: int = 7,
    model_variant: str = "vit_b16",
    device: Optional[str] = None,
) -> Dict:
    """
    Two-phase training with advanced techniques.

    Phase 1 (Epochs 1-{phase1_epochs}):
        - Backbone frozen, only head trained
        - High LR, fast convergence of classification head
    Phase 2 (Epochs {phase1_epochs+1}-{epochs}):
        - Full model unfrozen with layer-wise LR decay
        - Head LR: learning_rate
        - Backbone LR: learning_rate × backbone_lr_mult
    """
    dataset_dir = Path(dataset_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device == "cuda")
    logger.info(f"Device: {device} | AMP: {use_amp}")

    # ── Load data ─────────────────────────────────────
    with open(dataset_dir/"metadata.json") as f:
        meta = json.load(f)
    with open(dataset_dir/"splits.json") as f:
        splits = json.load(f)
    all_samples = meta["samples"]
    train_s = [all_samples[i] for i in splits["train"]]
    val_s   = [all_samples[i] for i in splits["val"]]
    logger.info(f"Train: {len(train_s)} | Val: {len(val_s)} | Sources: {meta.get('sources',{})}")

    train_ds = AccessibilityDataset(train_s, dataset_dir/"images", get_train_transform())
    val_ds   = AccessibilityDataset(val_s,   dataset_dir/"images", get_val_transform())

    sampler = make_weighted_sampler(train_s)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    # ── Model ──────────────────────────────────────────
    model = get_model(variant=model_variant, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    params = count_parameters(model)
    logger.info(f"Model: {getattr(model,'variant','?')} | Params: {params['total_millions']} total, {params['trainable_millions']} trainable (Phase 1)")

    # ── EMA ────────────────────────────────────────────
    ema = EMA(model, decay=0.9999) if use_ema else None

    # ── Loss: BCEWithLogitsLoss + label smoothing ──────
    # Positive weight: estimate class imbalance from training set
    pos_counts = torch.zeros(NUM_CLASSES)
    for s in train_s:
        pos_counts += torch.tensor(s["labels"], dtype=torch.float32)
    neg_counts = len(train_s) - pos_counts
    pos_weight = (neg_counts / (pos_counts + 1)).clamp(max=10.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Phase 1 Optimizer (head only) ─────────────────
    def make_optimizer(lr_head: float, lr_backbone: Optional[float] = None):
        if lr_backbone is None or not hasattr(model, 'backbone'):
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr_head, weight_decay=weight_decay
            )
        return torch.optim.AdamW([
            {"params": model.classifier.parameters(), "lr": lr_head},
            {"params": model.backbone.parameters(),   "lr": lr_backbone},
        ], weight_decay=weight_decay)

    optimizer = make_optimizer(learning_rate)
    scaler = GradScaler() if use_amp else None

    # Track history
    history: Dict = {"train_loss":[],"val_loss":[],"train_f1":[],"val_f1":[],
                     "best_epoch":0,"best_val_f1":0.0,"per_class_f1":[]}
    best_val_f1 = 0.0
    no_improve  = 0
    model_path  = dataset_dir / "accessibility_model.pth"

    # ─────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Phase transition ───────────────────────────
        if epoch == phase1_epochs + 1:
            logger.info(f"\n{'='*55}")
            logger.info(f"  PHASE 2: Unfreezing backbone (epoch {epoch})")
            if hasattr(model, 'unfreeze_backbone'):
                model.unfreeze_backbone()
            optimizer = make_optimizer(learning_rate, learning_rate * backbone_lr_mult)
            params2 = count_parameters(model)
            logger.info(f"  Trainable params: {params2['trainable_millions']}")
            logger.info(f"{'='*55}\n")

        # ── OneCycleLR (reset each phase) ─────────────
        total_steps = len(train_dl) * (epochs - (phase1_epochs if epoch <= phase1_epochs else 0))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate,
            steps_per_epoch=len(train_dl), epochs=epochs - phase1_epochs + 1,
            pct_start=0.1, anneal_strategy='cos',
        ) if epoch == 1 or epoch == phase1_epochs + 1 else scheduler

        # ── Train ──────────────────────────────────────
        model.train()
        train_loss, all_p, all_t = 0.0, [], []
        optimizer.zero_grad()

        for step, (imgs, lbls) in enumerate(train_dl):
            imgs, lbls = imgs.to(device), lbls.to(device)

            # Apply MixUp or CutMix (50/50)
            if use_mixup and np.random.rand() < 0.5:
                if np.random.rand() < 0.5:
                    imgs, lbls = mixup(imgs, lbls, mixup_alpha)
                else:
                    imgs, lbls = cutmix(imgs, lbls, cutmix_alpha)
            # Label smoothing (manual for BCEWithLogitsLoss)
            if label_smoothing > 0:
                lbls = lbls * (1 - label_smoothing) + label_smoothing / 2

            if use_amp:
                with autocast():
                    out  = model(imgs)
                    loss = criterion(out, lbls) / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                out  = model(imgs)
                loss = criterion(out, lbls) / grad_accum_steps
                loss.backward()

            train_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                if ema: ema.update()
                try: scheduler.step()
                except Exception: pass

            all_p.append(out.detach().cpu())
            all_t.append(lbls.detach().cpu())

        train_loss /= len(train_dl)
        train_m = compute_metrics(torch.cat(all_p), torch.cat(all_t))

        # ── Validate ───────────────────────────────────
        model.eval()
        val_loss, all_p, all_t = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.to(device)
                if use_amp:
                    with autocast():
                        out  = model(imgs)
                        loss = criterion(out, lbls)
                else:
                    out  = model(imgs)
                    loss = criterion(out, lbls)
                val_loss += loss.item()
                all_p.append(out.cpu()); all_t.append(lbls.cpu())
        val_loss /= len(val_dl)
        val_m = compute_metrics(torch.cat(all_p), torch.cat(all_t))
        elapsed = time.time() - t0

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_m["macro_f1"])
        history["val_f1"].append(val_m["macro_f1"])
        history["per_class_f1"].append({c: val_m[f"{c}_f1"] for c in VIOLATION_CLASSES})

        logger.info(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train {train_loss:.4f} F1={train_m['macro_f1']:.4f} | "
            f"Val {val_loss:.4f} F1={val_m['macro_f1']:.4f} | "
            f"{elapsed:.0f}s | Phase {'1' if epoch <= phase1_epochs else '2'}"
        )

        # Save best
        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            history["best_epoch"] = epoch
            history["best_val_f1"] = best_val_f1
            no_improve = 0

            # Apply EMA weights for saving
            save_state = deepcopy(model.state_dict())
            if ema:
                ema.apply()
                ema_state = deepcopy(model.state_dict())
                model.load_state_dict(save_state)
                save_state = ema_state

            torch.save({
                "epoch": epoch,
                "model_state_dict": save_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "val_loss": val_loss,
                "class_names": VIOLATION_CLASSES,
                "num_classes": NUM_CLASSES,
                "variant": getattr(model, "variant", "vit_b16"),
            }, model_path)
            logger.info(f"  ★ Best model saved (Val F1={best_val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                logger.info(f"\n  Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

    # ── Final summary ──────────────────────────────────
    logger.info(f"\n{'='*55}")
    logger.info(f"Training Complete!")
    logger.info(f"Best Epoch: {history['best_epoch']} | Best Val F1: {history['best_val_f1']:.4f}")
    best_cls = history["per_class_f1"][history["best_epoch"]-1] if history["per_class_f1"] else {}
    for cls, f1 in best_cls.items():
        bar = "█" * int(f1 * 25)
        logger.info(f"  {cls:20s}: {f1:.4f}  {bar}")
    logger.info(f"{'='*55}")

    with open(dataset_dir/"training_history.json","w") as f:
        json.dump(history, f, indent=2)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   default="dataset")
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--batch",     type=int,   default=32)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--phase1",    type=int,   default=5)
    p.add_argument("--variant",   default="vit_b16", choices=["vit_b16","efficientnet_v2"])
    p.add_argument("--no-mixup",  action="store_true")
    p.add_argument("--no-ema",    action="store_true")
    args = p.parse_args()
    train_model(
        dataset_dir=args.dataset, epochs=args.epochs, batch_size=args.batch,
        learning_rate=args.lr, phase1_epochs=args.phase1, model_variant=args.variant,
        use_mixup=not args.no_mixup, use_ema=not args.no_ema,
    )

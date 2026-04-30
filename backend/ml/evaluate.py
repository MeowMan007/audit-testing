"""
Model Evaluation Module — Academic rigor for the final year project.

This module provides tools to rigorously evaluate the trained ViT model,
generating metrics required for a high-quality academic report:
- Per-class Precision, Recall, F1-Score
- Confusion Matrix visualization
- ROC Curves and AUC metrics
- Precision-Recall Curves and AP metrics
- Threshold optimization
"""
import os
import json
import logging
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    multilabel_confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from backend.ml.model import VIOLATION_CLASSES, NUM_CLASSES, get_model
from backend.ml.train import AccessibilityDataset, get_val_transform

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str, data_dir: str, output_dir: str = "dataset/evaluation"):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
    def load_model(self):
        """Load the model for evaluation."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        variant = checkpoint.get("variant", "vit_b16")
        
        self.model = get_model(variant=variant, pretrained=False, freeze_backbone=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded {variant} model for evaluation")
        return variant
        
    def get_dataloader(self, batch_size=32):
        """Prepare the validation dataloader."""
        val_dataset = AccessibilityDataset(
            img_dir=self.data_dir / "images",
            labels_file=self.data_dir / "labels_val.json",
            transform=get_val_transform()
        )
        return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
    def run_evaluation(self):
        """Run full evaluation suite and generate plots."""
        self.load_model()
        dataloader = self.get_dataloader()
        
        all_targets = []
        all_probs = []
        
        logger.info("Running inference on validation set...")
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = targets.cpu().numpy()
                
                logits = self.model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                all_targets.append(targets)
                all_probs.append(probs)
                
        y_true = np.vstack(all_targets)
        y_prob = np.vstack(all_probs)
        
        # 1. Optimal Threshold Analysis
        logger.info("Computing optimal thresholds...")
        thresholds = self._optimize_thresholds(y_true, y_prob)
        
        y_pred = (y_prob >= thresholds).astype(int)
        
        # 2. Classification Report
        logger.info("Generating classification report...")
        report_dict = classification_report(
            y_true, y_pred, 
            target_names=VIOLATION_CLASSES, 
            output_dict=True,
            zero_division=0
        )
        
        with open(self.output_dir / "classification_report.json", "w") as f:
            json.dump({
                "thresholds": thresholds.tolist(),
                "metrics": report_dict
            }, f, indent=4)
            
        # 3. Visualizations
        logger.info("Generating plots...")
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curves(y_true, y_prob)
        self._plot_pr_curves(y_true, y_prob)
        
        logger.info(f"Evaluation complete. Results saved to {self.output_dir}")
        
    def _optimize_thresholds(self, y_true, y_prob):
        """Find the optimal threshold for each class that maximizes F1 score."""
        best_thresholds = np.zeros(NUM_CLASSES)
        
        for i in range(NUM_CLASSES):
            precision, recall, thresholds = precision_recall_curve(y_true[:, i], y_prob[:, i])
            # Avoid division by zero
            f1_scores = np.divide(
                2 * precision * recall, 
                precision + recall, 
                out=np.zeros_like(precision), 
                where=(precision + recall) != 0
            )
            # thresholds is len(precision)-1, so we take the best one
            if len(thresholds) > 0:
                best_idx = np.argmax(f1_scores[:-1])
                best_thresholds[i] = thresholds[best_idx]
            else:
                best_thresholds[i] = 0.5
                
        return best_thresholds
        
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Plot multi-label confusion matrices."""
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (matrix, ax, cls_name) in enumerate(zip(mcm, axes, VIOLATION_CLASSES)):
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f'Class: {cls_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300)
        plt.close()
        
    def _plot_roc_curves(self, y_true, y_prob):
        """Plot ROC curves and compute AUC."""
        plt.figure(figsize=(10, 8))
        
        for i, cls_name in enumerate(VIOLATION_CLASSES):
            if np.sum(y_true[:, i]) > 0:  # Only if class exists in true labels
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{cls_name} (AUC = {roc_auc:.2f})')
                
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) per Class')
        plt.legend(loc="lower right")
        
        plt.savefig(self.output_dir / "roc_curves.png", dpi=300)
        plt.close()
        
    def _plot_pr_curves(self, y_true, y_prob):
        """Plot Precision-Recall curves."""
        plt.figure(figsize=(10, 8))
        
        for i, cls_name in enumerate(VIOLATION_CLASSES):
            if np.sum(y_true[:, i]) > 0:
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                avg_prec = average_precision_score(y_true[:, i], y_prob[:, i])
                plt.plot(recall, precision, lw=2, label=f'{cls_name} (AP = {avg_prec:.2f})')
                
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve per Class')
        plt.legend(loc="lower left")
        
        plt.savefig(self.output_dir / "pr_curves.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = ModelEvaluator(
        model_path="dataset/accessibility_model.pth",
        data_dir="dataset"
    )
    evaluator.run_evaluation()

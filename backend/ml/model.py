"""
AccessibilityNet — Custom CNN Model for Accessibility Violation Detection.

ARCHITECTURE (Requirement 4 — Machine Learning Model):
    This module defines the neural network architecture used to detect
    visual accessibility violations in web page screenshots.

    Model: AccessibilityNet
    ├── Backbone: EfficientNet-B0 (pretrained on ImageNet, 5.3M params)
    │   └── Feature extraction layers (frozen during initial training)
    └── Classifier Head (custom, trained from scratch):
        ├── AdaptiveAvgPool2d → (batch, 1280)
        ├── Dropout(0.3)
        ├── BatchNorm1d(1280)
        ├── Linear(1280 → 256)
        ├── ReLU
        ├── Dropout(0.2)
        └── Linear(256 → 6)  ← 6 violation classes, sigmoid output

    Output: 6-class multi-label classification
        [low_contrast, small_text, missing_alt, small_targets, bad_heading, poor_layout]

WHY EFFICIENTNET-B0:
    - Best accuracy-to-parameter ratio among standard CNNs
    - Small enough to train on Google Colab T4 GPU (~5.3M params)
    - Pretrained on ImageNet provides strong visual feature extraction
    - Compound scaling (width, depth, resolution) is more efficient than
      scaling only one dimension (as in ResNet/VGG)

WHY MULTI-LABEL (not multi-class):
    - A single screenshot can have MULTIPLE violations simultaneously
      (e.g., low contrast AND missing alt text AND small targets)
    - Sigmoid activation treats each class independently
    - BCEWithLogitsLoss (not CrossEntropyLoss) is used for training

ACCURACY METRICS:
    - Per-class Precision, Recall, F1-score
    - Overall Hamming Loss (fraction of incorrect labels)
    - Expected performance on synthetic data: >80% F1 per class
"""
import torch
import torch.nn as nn
from torchvision import models

# Violation class definitions (must match dataset_generator.py)
VIOLATION_CLASSES = [
    "low_contrast",
    "small_text",
    "missing_alt",
    "small_targets",
    "bad_heading",
    "poor_layout",
]

NUM_CLASSES = len(VIOLATION_CLASSES)


class AccessibilityNet(nn.Module):
    """EfficientNet-B0 based multi-label classifier for accessibility violations.
    
    Architecture:
        1. EfficientNet-B0 feature extractor (pretrained on ImageNet)
        2. Global average pooling → 1280-dim feature vector
        3. Dropout → BatchNorm → Linear(1280→256) → ReLU → Dropout
        4. Linear(256→6) → Sigmoid (during inference)
    
    During training, use BCEWithLogitsLoss (applies sigmoid internally
    for numerical stability). During inference, apply sigmoid manually.
    
    Args:
        num_classes: Number of violation categories (default: 6)
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: If True, freeze the EfficientNet backbone layers
            (useful for small datasets to prevent overfitting)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        # Load EfficientNet-B0 backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        # Get the feature dimension from the backbone's classifier
        backbone_out_features = self.backbone.classifier[1].in_features  # 1280

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Optionally freeze backbone for fine-tuning with small data
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(backbone_out_features),
            nn.Linear(backbone_out_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self):
        """Initialize custom classifier head with Kaiming initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
               Recommended input size: 224x224
            
        Returns:
            Logits tensor of shape (batch, num_classes)
            Apply sigmoid for probabilities during inference.
        """
        features = self.backbone(x)  # (batch, 1280)
        logits = self.classifier(features)  # (batch, num_classes)
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict violations with a confidence threshold.
        
        Args:
            x: Input tensor (batch, 3, H, W)
            threshold: Confidence threshold for positive prediction
            
        Returns:
            Binary prediction tensor (batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs >= threshold).int()

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get violation probabilities (0-1) for each class.
        
        Args:
            x: Input tensor (batch, 3, H, W)
            
        Returns:
            Probability tensor (batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def get_model(pretrained: bool = True, freeze_backbone: bool = False) -> AccessibilityNet:
    """Factory function to create an AccessibilityNet model.
    
    Args:
        pretrained: Use ImageNet pretrained weights for the backbone
        freeze_backbone: Freeze backbone layers (for small dataset training)
        
    Returns:
        AccessibilityNet model instance
    """
    return AccessibilityNet(
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters for documentation.
    
    Returns:
        Dict with total, trainable, and frozen parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": f"{total / 1e6:.2f}M",
    }

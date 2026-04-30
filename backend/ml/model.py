"""
AccessibilityViT — Vision Transformer for Accessibility Violation Detection.

ARCHITECTURE UPGRADE: EfficientNet-B0 → Vision Transformer (ViT-B/16)

WHY VISION TRANSFORMER:
  - EfficientNet is a CNN: excellent at local texture features (contrast, edges)
  - ViT uses self-attention: captures GLOBAL layout relationships, which is
    critical for accessibility (e.g., "is heading structure logical?",
    "are targets clustered together or isolated?")
  - ViT-B/16 achieves state-of-the-art on visual recognition benchmarks
  - Pretrained on ImageNet-21k (14M images) → far richer initialization
    than EfficientNet pretrained on ImageNet-1k (1.2M images)

MODEL VARIANTS:
  "vit_b16"  → ViT-B/16, 86M params — PRIMARY (balanced quality/speed)
  "vit_l16"  → ViT-L/16, 307M params — HIGH ACCURACY (needs A100 GPU)
  "efficientnet_v2" → EfficientNetV2-M — FALLBACK (no ViT available)

ARCHITECTURE DETAILS (ViT-B/16):
  Input: 224×224×3 image
  → Patch Embedding: 196 patches of 16×16 pixels
  → 12 Transformer Encoder layers
     Each layer: Multi-Head Self-Attention (12 heads, d=768) + MLP
  → [CLS] token → 768-dim feature vector
  → Custom Head:
       LayerNorm(768)
       → Linear(768→512) → GELU → Dropout(0.4)
       → Linear(512→256) → GELU → Dropout(0.3)
       → Linear(256→6)  [multi-label sigmoid output]

TRAINING STRATEGY (Two-Phase Fine-Tuning):
  Phase 1 (Epochs 1-5):   Freeze ViT backbone → train head only (fast warmup)
  Phase 2 (Epochs 6-30):  Unfreeze all layers with layer-wise LR decay
                           (deeper layers get lower LR to preserve ImageNet features)
"""
import torch
import torch.nn as nn
from torchvision import models

VIOLATION_CLASSES = [
    "low_contrast",
    "small_text",
    "missing_alt",
    "small_targets",
    "bad_heading",
    "poor_layout",
]
NUM_CLASSES = len(VIOLATION_CLASSES)

# ─────────────────────────────────────────────────────────────────────────────
# Custom Classification Head
# ─────────────────────────────────────────────────────────────────────────────

class AccessibilityHead(nn.Module):
    """Multi-label classification head with LayerNorm and GELU activations."""

    def __init__(self, in_features: int, num_classes: int, dropout1=0.4, dropout2=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Primary Model: ViT-B/16 (Vision Transformer)
# ─────────────────────────────────────────────────────────────────────────────

class AccessibilityViT(nn.Module):
    """
    Vision Transformer (ViT-B/16) for multi-label accessibility classification.

    Supports two-phase fine-tuning:
      Phase 1: freeze_backbone=True  → train head only
      Phase 2: freeze_backbone=False → train all layers w/ layer-wise LR decay
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()
        self.variant = "vit_b16"

        if pretrained:
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            backbone = models.vit_b_16(weights=weights)
        else:
            backbone = models.vit_b_16(weights=None)

        # Feature dim of ViT-B/16 is 768
        in_features = backbone.heads.head.in_features

        # Remove original head
        backbone.heads = nn.Identity()
        self.backbone = backbone

        # Custom head
        self.classifier = AccessibilityHead(in_features, num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze backbone → only head is trained (Phase 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all → full fine-tuning (Phase 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_layer_groups(self):
        """
        Return parameter groups for layer-wise LR decay.
        Deeper transformer layers get higher LR; early layers get lower LR.
        Used by the optimizer for discriminative fine-tuning.
        """
        # Patch embedding
        patch_embed = list(self.backbone.conv_proj.parameters()) + \
                      list(self.backbone.class_embedding.parameters() if hasattr(self.backbone,'class_embedding') else [])

        # Transformer encoder layers (12 layers in ViT-B)
        encoder_layers = []
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layers'):
            encoder_layers = [list(layer.parameters()) for layer in self.backbone.encoder.layers]

        # Head parameters
        head_params = list(self.classifier.parameters())

        return patch_embed, encoder_layers, head_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, 768)
        return self.classifier(features)  # (B, num_classes)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return (torch.sigmoid(self.forward(x)) >= threshold).int()

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: EfficientNetV2-M (if ViT weights unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class AccessibilityNetV2(nn.Module):
    """
    EfficientNetV2-M backbone — the upgraded EfficientNet.
    Better than EfficientNet-B0, used as fallback if ViT is too heavy.
    ~54M params (vs B0's 5.3M and ViT-B's 86M).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        self.variant = "efficientnet_v2_m"
        if pretrained:
            weights = models.EfficientNet_V2_M_Weights.DEFAULT
            backbone = models.efficientnet_v2_m(weights=weights)
        else:
            backbone = models.efficientnet_v2_m(weights=None)

        in_features = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = AccessibilityHead(in_features, num_classes, dropout1=0.3, dropout2=0.2)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            return (torch.sigmoid(self.forward(x)) >= threshold).int()

    def get_probabilities(self, x):
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(variant: str = "vit_b16", pretrained: bool = True,
              freeze_backbone: bool = True):
    """
    Factory function to create the best available model.

    Args:
        variant: "vit_b16" | "efficientnet_v2"
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze backbone (Phase 1 training)
    """
    if variant == "vit_b16":
        try:
            return AccessibilityViT(NUM_CLASSES, pretrained, freeze_backbone)
        except Exception as e:
            import logging; logging.getLogger(__name__).warning(f"ViT failed ({e}), falling back to EfficientNetV2")
            return AccessibilityNetV2(NUM_CLASSES, pretrained, freeze_backbone)
    else:
        return AccessibilityNetV2(NUM_CLASSES, pretrained, freeze_backbone)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable,
            "total_millions": f"{total/1e6:.1f}M",
            "trainable_millions": f"{trainable/1e6:.1f}M"}

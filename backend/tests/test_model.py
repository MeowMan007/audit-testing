import pytest
import torch
from backend.ml.model import AccessibilityViT, NUM_CLASSES

def test_vit_model_initialization():
    # Test initialization without downloading pretrained weights
    model = AccessibilityViT(pretrained=False, freeze_backbone=True)
    assert model is not None
    assert model.variant == "vit_b16"

def test_vit_forward_pass():
    model = AccessibilityViT(pretrained=False, freeze_backbone=True)
    model.eval()
    
    # Dummy input tensor (B, C, H, W) -> (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    assert output.shape == (1, NUM_CLASSES)

"""
Explainable AI (XAI) Module — Grad-CAM for Vision Transformer.

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM)
specifically adapted for the ViT-B/16 architecture. It visualizes which parts
of the screenshot the model is focusing on to make its predictions.
"""
import io
import base64
import logging
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class ViTGradCAM:
    """
    Grad-CAM for Vision Transformers.
    Extracts activations and gradients from the last transformer encoder block.
    """
    def __init__(self, model: nn.Module, target_layer=None):
        self.model = model
        # Set to eval mode, but we still need gradients
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

        self.activations = None
        self.gradients = None
        self.hooks = []
        
        # By default, target the last encoder block in torchvision ViT
        if target_layer is None:
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
                target_layer = model.backbone.encoder.layers[-1].ln_1
            else:
                logger.warning("Could not automatically find target layer for Grad-CAM.")
                return
                
        self._register_hooks(target_layer)
        
    def _register_hooks(self, target_layer):
        def forward_hook(module, input, output):
            # Output of ln_1 is (B, N, C)
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
            
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate Grad-CAM heatmap for a specific class index."""
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Target score for the given class
        score = output[0, class_idx]
        
        # Backward pass
        score.backward(retain_graph=True)
        
        if self.activations is None or self.gradients is None:
            logger.error("Failed to capture activations/gradients")
            return np.zeros((14, 14))
            
        # Activations: (1, 197, 768), Gradients: (1, 197, 768)
        act = self.activations[0].detach()
        grad = self.gradients[0].detach()
        
        # We only care about the 196 spatial tokens (skip CLS token at idx 0)
        spatial_act = act[1:] # (196, 768)
        spatial_grad = grad[1:] # (196, 768)
        
        # Global Average Pooling of gradients over the spatial dimensions
        weights = torch.mean(spatial_grad, dim=0) # (768,)
        
        # Weighted combination of activations
        cam = torch.zeros(spatial_act.shape[0], device=act.device) # (196,)
        for i, w in enumerate(weights):
            cam += w * spatial_act[:, i]
            
        cam = torch.relu(cam) # ReLU
        
        # Reshape to 14x14 (since ViT-B/16 on 224x224 has 196 patches)
        cam = cam.view(14, 14).cpu().numpy()
        
        # Normalize between 0 and 1
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
            
        return cam

def generate_attention_heatmap(
    image_tensor: torch.Tensor, 
    model: nn.Module, 
    class_idx: int,
    original_image_b64: str
) -> str:
    """
    Generate a heatmap overlaid on the original image and return as base64.
    """
    try:
        grad_cam = ViTGradCAM(model)
        cam = grad_cam.generate_cam(image_tensor, class_idx)
        grad_cam.remove_hooks()
        
        # Decode original image
        img_data = base64.b64decode(original_image_b64)
        original_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        orig_w, orig_h = original_img.size
        
        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (orig_w, orig_h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image (0.4 alpha for heatmap, 0.6 for original)
        orig_np = np.array(original_img)
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
        
        # Convert back to base64
        overlay_img = Image.fromarray(overlay)
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    except Exception as e:
        logger.error(f"Failed to generate attention heatmap: {e}")
        return original_image_b64

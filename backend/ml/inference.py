"""
Inference Module — Run predictions using AccessibilityViT (ViT-B/16).
Also supports EfficientNetV2-M as fallback.

Loads model variant from checkpoint's 'variant' field:
  - 'vit_b16'          → AccessibilityViT
  - 'efficientnet_v2'  → AccessibilityNetV2

Takes a web page screenshot (as base64 string, file path, PIL Image, or bytes)
and returns detected accessibility violations with:
- Violation category name
- Confidence score (0.0 - 1.0)
- Severity level (critical/warning/info)
- Related WCAG criterion
- Suggested fix

This module is used by dl_engine.py to integrate CNN predictions
into the overall audit pipeline.
"""
import base64
import io
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple

import torch
from torchvision import transforms
from PIL import Image

from backend.ml.model import AccessibilityViT, get_model, VIOLATION_CLASSES, NUM_CLASSES

logger = logging.getLogger(__name__)


# ============================================================
# Violation Metadata (for rich output)
# ============================================================
VIOLATION_METADATA = {
    "low_contrast": {
        "title": "Low Color Contrast Detected",
        "wcag": "1.4.3",
        "severity": "critical",
        "description": "The AI model detected text areas with insufficient color contrast, making content difficult to read for users with low vision.",
        "suggestion": "Increase text-to-background contrast ratio to at least 4.5:1 for normal text and 3:1 for large text (18px+).",
    },
    "small_text": {
        "title": "Small Text Detected",
        "wcag": "1.4.4",
        "severity": "warning",
        "description": "The AI model found text that appears too small to be easily readable, especially for users with visual impairments.",
        "suggestion": "Use a minimum font size of 16px for body text. Ensure text can be resized up to 200% without loss of content.",
    },
    "missing_alt": {
        "title": "Likely Missing Alt Text",
        "wcag": "1.1.1",
        "severity": "critical",
        "description": "The AI model detected images that likely lack descriptive alternative text, making them inaccessible to screen reader users.",
        "suggestion": "Add descriptive alt attributes to all informational images. Use alt='' for purely decorative images.",
    },
    "small_targets": {
        "title": "Small Touch/Click Targets",
        "wcag": "2.5.5",
        "severity": "warning",
        "description": "The AI model found interactive elements (buttons, links) that appear smaller than the recommended 44x44px minimum touch target size.",
        "suggestion": "Increase button and link sizes to at least 44x44 CSS pixels. Add padding if needed to meet the minimum target size.",
    },
    "bad_heading": {
        "title": "Poor Heading Structure",
        "wcag": "2.4.6",
        "severity": "warning",
        "description": "The AI model detected heading hierarchy issues — headings may be skipped, missing, or improperly nested.",
        "suggestion": "Use a logical heading hierarchy: one <h1> per page, followed by <h2>, <h3>, etc. without skipping levels.",
    },
    "poor_layout": {
        "title": "Cluttered/Inaccessible Layout",
        "wcag": "1.4.10",
        "severity": "warning",
        "description": "The AI model detected layout issues such as overlapping elements, cluttered content, or poor visual organization.",
        "suggestion": "Ensure content reflows properly, avoid overlapping elements, and maintain sufficient whitespace between interactive components.",
    },
}


class AccessibilityInference:
    """Run accessibility violation predictions on web page screenshots.
    
    Loads a trained AccessibilityNet model and provides inference
    methods for single images in various input formats.
    
    Usage:
        engine = AccessibilityInference(model_path="dataset/accessibility_model.pth")
        results = engine.predict(screenshot_base64)
    """

    def __init__(
        self,
        model_path: str = "dataset/accessibility_model.pth",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.model: Optional[nn.Module] = None
        self.loaded = False

        # Image preprocessing (must match training transforms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Try to load the model
        self._load_model()

    def _load_model(self):
        """Load the trained model from disk."""
        if not self.model_path.exists():
            logger.warning(
                f"Model not found at {self.model_path}. "
                "CNN predictions will be skipped. Train the model first."
            )
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            num_classes = checkpoint.get("num_classes", NUM_CLASSES)
            variant = checkpoint.get("variant", "vit_b16")

            from backend.ml.model import get_model
            self.model = get_model(variant=variant, pretrained=False, freeze_backbone=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True

            logger.info(
                f"Model loaded: {variant} from {self.model_path.name} "
                f"(epoch {checkpoint.get('epoch','?')}, "
                f"val_f1={checkpoint.get('val_f1',0):.4f})"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def predict(
        self,
        image_input: Union[str, bytes, Image.Image],
    ) -> List[Dict]:
        """Run inference on a single image.
        
        Args:
            image_input: Image as base64 string, file path, bytes, or PIL Image
            
        Returns:
            List of detected violations with confidence scores and metadata.
            Each detection is a dict with:
                - category: violation class name
                - confidence: model confidence (0.0-1.0)
                - severity: critical/warning/info
                - title: human-readable title
                - description: detailed explanation
                - wcag_criterion: related WCAG criterion
                - suggestion: suggested fix
        """
        if not self.loaded or self.model is None:
            logger.warning("Model not loaded — skipping CNN inference")
            return []

        # Convert input to PIL Image
        image = self._to_pil(image_input)
        if image is None:
            return []

        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu()

        # Build results
        detections = []
        for i, (cls_name, prob) in enumerate(zip(VIOLATION_CLASSES, probs)):
            confidence = prob.item()
            if confidence >= self.confidence_threshold:
                meta = VIOLATION_METADATA.get(cls_name, {})
                detections.append({
                    "category": cls_name,
                    "confidence": round(confidence, 4),
                    "severity": meta.get("severity", "warning"),
                    "title": meta.get("title", cls_name),
                    "description": meta.get("description", ""),
                    "wcag_criterion": meta.get("wcag", ""),
                    "suggestion": meta.get("suggestion", ""),
                })

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        logger.info(f"CNN detected {len(detections)} violations")
        return detections

    def predict_with_explanation(self, screenshot_b64: str) -> Tuple[List[Dict], Optional[str]]:
        """Predict and generate a Grad-CAM heatmap for the most confident violation."""
        if not self.loaded or self.model is None:
            return [], None
            
        try:
            image_tensor = self._to_pil(screenshot_b64)
            if image_tensor is None:
                return [], None
                
            tensor = self.transform(image_tensor).unsqueeze(0).to(self.device)
            tensor.requires_grad_()
            
            probs = self.model.get_probabilities(tensor)[0]
            
            detections = []
            max_prob = -1.0
            max_class_idx = -1
            
            for i, prob in enumerate(probs):
                p = float(prob)
                if p >= self.confidence_threshold:
                    class_name = VIOLATION_CLASSES[i]
                    meta = VIOLATION_METADATA.get(class_name, {})
                    detections.append({
                        "category": class_name,
                        "confidence": p,
                        "severity": meta.get("severity", "warning"),
                        "title": meta.get("title", class_name),
                        "description": meta.get("description", ""),
                        "wcag_criterion": meta.get("wcag", ""),
                        "suggestion": meta.get("suggestion", "")
                    })
                    if p > max_prob:
                        max_prob = p
                        max_class_idx = i
                        
            heatmap_b64 = None
            if max_class_idx >= 0:
                from backend.ml.explainability import generate_attention_heatmap
                heatmap_b64 = generate_attention_heatmap(
                    image_tensor=tensor,
                    model=self.model,
                    class_idx=max_class_idx,
                    original_image_b64=screenshot_b64
                )
                
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            return detections, heatmap_b64
            
        except Exception as e:
            logger.error(f"Inference+XAI failed: {e}")
            return [], None

    def get_all_probabilities(
        self,
        image_input: Union[str, bytes, Image.Image],
    ) -> Dict[str, float]:
        """Get probabilities for ALL classes (even below threshold).
        
        Useful for debugging and reporting.
        
        Returns:
            Dict mapping class name to probability
        """
        if not self.loaded or self.model is None:
            return {}

        image = self._to_pil(image_input)
        if image is None:
            return {}

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu()

        return {
            cls_name: round(prob.item(), 4)
            for cls_name, prob in zip(VIOLATION_CLASSES, probs)
        }

    def _to_pil(self, image_input: Union[str, bytes, Image.Image]) -> Optional[Image.Image]:
        """Convert various image formats to PIL Image."""
        try:
            if isinstance(image_input, Image.Image):
                return image_input.convert("RGB")

            if isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input)).convert("RGB")

            if isinstance(image_input, str):
                # Try as file path first
                if Path(image_input).exists():
                    return Image.open(image_input).convert("RGB")
                
                # Try as base64
                try:
                    img_bytes = base64.b64decode(image_input)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    pass

            logger.error(f"Cannot convert input to image: {type(image_input)}")
            return None
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self.loaded and self.model is not None

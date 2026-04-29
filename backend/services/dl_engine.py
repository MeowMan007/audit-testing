"""
Deep Learning Engine — Orchestrates CNN-based accessibility analysis.

Bridges the trained AccessibilityNet model with the audit pipeline.
Takes page screenshots and returns AI-generated accessibility insights
that supplement the rule-based checks from rule_engine.py.

The DL engine is OPTIONAL — the rule-based engine works without it.
If the trained model is not found, the engine gracefully degrades
and returns an empty insights list.
"""
import logging
from typing import Optional, List, Dict

from backend.models.schemas import DLInsight, SeverityLevel
from backend.ml.inference import AccessibilityInference

logger = logging.getLogger(__name__)


class DLEngine:
    """Orchestrates deep learning analysis for accessibility auditing.
    
    Uses the trained AccessibilityNet CNN model to analyze page
    screenshots and detect visual accessibility violations that
    rule-based checks cannot catch.
    
    Usage:
        engine = DLEngine(model_path="dataset/accessibility_model.pth")
        insights = await engine.analyze(screenshot_b64="...")
    """

    def __init__(self, model_path: str = "dataset/accessibility_model.pth"):
        self.inference = AccessibilityInference(
            model_path=model_path,
            confidence_threshold=0.4,  # Slightly lower to catch more issues
        )

    async def analyze(
        self,
        screenshot_b64: Optional[str] = None,
    ) -> List[DLInsight]:
        """Run CNN analysis on a page screenshot.
        
        Args:
            screenshot_b64: Base64 encoded screenshot PNG
            
        Returns:
            List of DLInsight objects with CNN detections
        """
        if not screenshot_b64:
            logger.info("No screenshot provided — skipping CNN analysis")
            return []

        if not self.inference.is_available:
            logger.info("CNN model not loaded — skipping AI analysis")
            return []

        try:
            # Run CNN inference
            detections = self.inference.predict(screenshot_b64)

            # Convert to DLInsight schema objects
            insights = []
            for det in detections:
                severity_map = {
                    "critical": SeverityLevel.CRITICAL,
                    "warning": SeverityLevel.WARNING,
                    "info": SeverityLevel.INFO,
                }
                insights.append(DLInsight(
                    category=det["category"],
                    confidence=det["confidence"],
                    severity=severity_map.get(det["severity"], SeverityLevel.WARNING),
                    title=det["title"],
                    description=det["description"],
                    wcag_criterion=det.get("wcag_criterion"),
                    suggestion=det.get("suggestion"),
                ))

            logger.info(f"DL engine produced {len(insights)} insights")
            return insights

        except Exception as e:
            logger.error(f"DL engine analysis failed: {e}")
            return []

    @property
    def is_available(self) -> bool:
        """Whether the CNN model is loaded and ready."""
        return self.inference.is_available

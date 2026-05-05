"""
Report Generator — Aggregates rule engine and DL findings into a scored report.

SCORING METHODOLOGY (Requirement 7 — Output):
    - Starts at 100 points
    - Each issue deducts points based on severity:
        Critical: -5 points per issue
        Warning:  -2 points per issue
        Info:     -1 point per issue
    - AI insights add additional deductions (weighted by confidence)
    - Score is clamped to [0, 100]
    
    Grade scale:
        90-100: A (Excellent)
        80-89:  B (Good)
        70-79:  C (Fair)
        60-69:  D (Poor)
        0-59:   F (Failing)

CATEGORIES (WCAG Principles):
    - Perceivable:   Can users perceive the content? (1.x criteria)
    - Operable:      Can users operate the interface? (2.x criteria)
    - Understandable: Can users understand the content? (3.x criteria)
    - Robust:        Is the content robust for assistive tech? (4.x criteria)
    - AI Analysis:   Visual issues detected by the CNN model
"""
import logging
from datetime import datetime, timezone
from typing import List

from backend.models.schemas import (
    AuditReport, CategoryScore, AccessibilityIssue,
    DLInsight, SeverityLevel,
)

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates comprehensive accessibility audit reports.
    
    Combines rule-based WCAG checks with AI/CNN model detections
    to produce a scored, categorized audit report.
    """

    def generate(
        self,
        url: str,
        issues: List[AccessibilityIssue],
        dl_insights: List[DLInsight],
        screenshot_b64: str = None,
        scan_duration: float = 0.0,
        ai_model_used: bool = False,
        reading_order: dict = None,
        focus_trap: dict = None,
    ) -> AuditReport:
        """Generate the final audit report.
        
        Args:
            url: The audited URL
            issues: Rule engine findings
            dl_insights: CNN model findings
            screenshot_b64: Base64 screenshot (included in report)
            scan_duration: Total audit duration in seconds
            ai_model_used: Whether the CNN model was used
            reading_order: Reading order analysis results dict
            focus_trap: Focus trap & keyboard reachability results dict
            
        Returns:
            Complete AuditReport with score, grade, categories, and all findings
        """
        # Calculate overall score
        score = self._calculate_score(issues, dl_insights)
        grade = self._score_to_grade(score)

        # Count by severity
        critical_count = sum(
            1 for i in issues if i.severity == SeverityLevel.CRITICAL
        )
        warning_count = sum(
            1 for i in issues if i.severity == SeverityLevel.WARNING
        )

        # Build category breakdowns
        categories = self._build_categories(issues, dl_insights)

        report = AuditReport(
            url=url,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=round(score, 1),
            grade=grade,
            total_issues=len(issues),
            critical_count=critical_count,
            warning_count=warning_count,
            issues=issues,
            dl_insights=dl_insights,
            categories=categories,
            screenshot=screenshot_b64,
            reading_order=reading_order,
            focus_trap=focus_trap,
            scan_duration=round(scan_duration, 2),
            ai_model_used=ai_model_used,
        )

        logger.info(
            f"Report generated: score={score:.1f}, grade={grade}, "
            f"issues={len(issues)}, critical={critical_count}, warnings={warning_count}"
        )
        return report

    def _calculate_score(
        self,
        issues: List[AccessibilityIssue],
        dl_insights: List[DLInsight],
    ) -> float:
        """Calculate accessibility score (0-100).
        
        Deduction-based scoring:
        - Start at 100
        - Deduct per issue (based on score_impact field)
        - Deduct per AI insight (weighted by confidence)
        - Clamp to [0, 100]
        """
        score = 100.0

        # Rule-based deductions
        for issue in issues:
            score -= issue.score_impact

        # AI insight deductions (weighted by confidence)
        for insight in dl_insights:
            if insight.severity == SeverityLevel.CRITICAL:
                score -= 3.0 * insight.confidence
            elif insight.severity == SeverityLevel.WARNING:
                score -= 1.5 * insight.confidence

        return max(0.0, min(100.0, score))

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _build_categories(
        self,
        issues: List[AccessibilityIssue],
        dl_insights: List[DLInsight],
    ) -> List[CategoryScore]:
        """Break down score by WCAG principle categories.
        
        Maps WCAG criteria to their parent principles:
        - 1.x.x → Perceivable
        - 2.x.x → Operable
        - 3.x.x → Understandable
        - 4.x.x → Robust
        """
        category_map = {
            "Perceivable": {"issues": [], "description": "Content must be presentable to users in ways they can perceive."},
            "Operable": {"issues": [], "description": "User interface components must be operable."},
            "Understandable": {"issues": [], "description": "Information and UI operation must be understandable."},
            "Robust": {"issues": [], "description": "Content must be robust enough for assistive technologies."},
            "AI Analysis": {"issues": [], "description": "Visual accessibility issues detected by the CNN model."},
        }

        # Categorize rule-based issues
        for issue in issues:
            criterion = issue.wcag_criterion
            if criterion.startswith("1."):
                category_map["Perceivable"]["issues"].append(issue)
            elif criterion.startswith("2."):
                category_map["Operable"]["issues"].append(issue)
            elif criterion.startswith("3."):
                category_map["Understandable"]["issues"].append(issue)
            elif criterion.startswith("4."):
                category_map["Robust"]["issues"].append(issue)

        # Build category scores
        categories = []
        for name, data in category_map.items():
            if name == "AI Analysis":
                issue_count = len(dl_insights)
                deduction = sum(
                    3.0 * i.confidence if i.severity == SeverityLevel.CRITICAL else 1.5 * i.confidence
                    for i in dl_insights
                )
            else:
                issue_count = len(data["issues"])
                deduction = sum(i.score_impact for i in data["issues"])
            
            cat_score = max(0, 100 - deduction * (100 / 25))  # Scale to 0-100

            categories.append(CategoryScore(
                name=name,
                score=round(min(100, cat_score), 1),
                max_score=100,
                issue_count=issue_count,
                description=data["description"],
            ))

        return categories

import pytest
from backend.services.report_generator import ReportGenerator
from backend.models.schemas import AccessibilityIssue, DLInsight, SeverityLevel, WCAGLevel

def test_report_generation():
    generator = ReportGenerator()
    
    issues = [
        AccessibilityIssue(
            id="test-1",
            wcag_criterion="1.1.1",
            wcag_level=WCAGLevel.A,
            severity=SeverityLevel.CRITICAL,
            title="Test Critical Issue",
            description="Test",
            suggestion="Test",
            score_impact=5.0
        ),
        AccessibilityIssue(
            id="test-2",
            wcag_criterion="2.4.4",
            wcag_level=WCAGLevel.A,
            severity=SeverityLevel.WARNING,
            title="Test Warning Issue",
            description="Test",
            suggestion="Test",
            score_impact=2.0
        )
    ]
    
    dl_insights = [
        DLInsight(
            category="low_contrast",
            confidence=0.9,
            severity=SeverityLevel.CRITICAL,
            title="AI Low Contrast",
            description="Test",
            wcag_criterion="1.4.3",
            suggestion="Test"
        )
    ]
    
    report = generator.generate(
        url="https://test.com",
        issues=issues,
        dl_insights=dl_insights,
        screenshot_b64=None,
        scan_duration=1.5,
        ai_model_used=True
    )
    
    # 100 - 5.0 (critical rule) - 2.0 (warning rule) - (3.0 * 0.9) (AI critical) = 90.3
    assert abs(report.overall_score - 90.3) < 0.1
    assert report.grade == "A"
    assert report.total_issues == 2
    assert report.critical_count == 1
    assert report.warning_count == 1
    
    # Check categories
    cat_names = [c.name for c in report.categories]
    assert "Perceivable" in cat_names
    assert "Operable" in cat_names
    assert "AI Analysis" in cat_names
    
    perceivable = next(c for c in report.categories if c.name == "Perceivable")
    assert perceivable.issue_count == 1

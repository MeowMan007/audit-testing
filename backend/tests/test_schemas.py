import pytest
from pydantic import ValidationError
from backend.models.schemas import AuditRequest, SeverityLevel, DLInsight

def test_audit_request_validation():
    # Valid request
    req = AuditRequest(url="https://example.com", include_ai=True)
    assert req.url == "https://example.com"
    
    # Missing required field
    with pytest.raises(ValidationError):
        AuditRequest()

def test_dl_insight_validation():
    insight = DLInsight(
        category="low_contrast",
        confidence=0.85,
        severity=SeverityLevel.WARNING,
        title="Test",
        description="Desc"
    )
    assert insight.category == "low_contrast"
    
    with pytest.raises(ValidationError):
        # Missing required title
        DLInsight(category="test", confidence=0.5, severity=SeverityLevel.INFO, description="test")

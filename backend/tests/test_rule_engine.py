import pytest
from backend.services.rule_engine import RuleEngine
from backend.models.schemas import SeverityLevel

def test_rule_engine_good_site(sample_dom_good):
    engine = RuleEngine()
    issues = engine.run_all_checks(sample_dom_good, [])
    
    # The good site has almost no issues
    assert len(issues) == 0

def test_rule_engine_bad_site(sample_dom_bad):
    engine = RuleEngine()
    # Add a mock computed style for the contrast check
    computed_styles = [
        {
            "tag": "p",
            "text": "This is low contrast text.",
            "color": "rgb(153, 153, 153)", # #999999
            "backgroundColor": "rgb(255, 255, 255)", # #ffffff
            "fontSize": "16px",
            "fontWeight": "400"
        }
    ]
    issues = engine.run_all_checks(sample_dom_bad, computed_styles)
    
    # Verify various checks caught the bad markup
    issue_titles = [i.title for i in issues]
    
    assert "Missing Page Title" in issue_titles
    assert "Missing Language Declaration" in issue_titles
    assert "Zoom/Scaling Disabled" in issue_titles
    assert "Skipped Heading Level (h1 → h2)" in issue_titles
    assert "Image Missing Alt Text" in issue_titles
    assert "Generic Link Text" in issue_titles
    assert "Duplicate ID" in issue_titles
    assert "Form Input Missing Label" in issue_titles
    assert "Button Missing Accessible Name" in issue_titles
    assert "Insufficient Color Contrast" in issue_titles
    assert "Iframe Missing Title" in issue_titles
    assert "Missing Skip Navigation Link" in issue_titles

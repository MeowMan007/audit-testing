import pytest
from backend.services.dom_analyzer import DOMAnalyzer

def test_dom_analyzer(sample_html_good):
    analyzer = DOMAnalyzer()
    data = analyzer.analyze(sample_html_good)
    
    assert data.title == "Good Site"
    assert data.lang == "en"
    assert data.has_main is True
    assert len(data.images) == 1
    assert data.images[0].get('alt') == "Company Logo"
    assert len(data.headings) == 1
    assert data.headings[0].get('level') == 'h1'

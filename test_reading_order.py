"""Quick verification test for reading order analysis."""
from backend.services.reading_order import ReadingOrderAnalyzer, _kendall_tau

# Test Kendall's Tau
assert abs(_kendall_tau([0,1,2,3,4], [0,1,2,3,4]) - 1.0) < 0.01, "Perfect match failed"
assert abs(_kendall_tau([0,1,2,3,4], [4,3,2,1,0]) - (-1.0)) < 0.01, "Perfect reversal failed"
tau = _kendall_tau([0,1,2,3,4], [0,2,1,3,4])
assert 0.0 < tau < 1.0, f"Partial disorder tau={tau}"
print(f"Kendall's Tau tests passed (partial disorder tau={tau:.3f})")

# Test with mock element rects - well-ordered page
analyzer = ReadingOrderAnalyzer()
rects = {}
html_parts = []
for i in range(10):
    al_id = f'al-{i}'
    rects[al_id] = {'x': 50, 'y': 50 + i * 60, 'w': 400, 'h': 40}
    html_parts.append(f'<p data-al-id="{al_id}">Paragraph {i}</p>')
html = '<html><body>' + ''.join(html_parts) + '</body></html>'

result = analyzer.analyze(rects, html)
print(f"Well-ordered: tau={result.correlation_score:.3f}, severity={result.severity}, mismatches={result.mismatch_count}")
assert result.severity == 'pass', f"Expected pass, got {result.severity}"

# Test disordered page (visual order reversed from DOM)
rects2 = {}
for i in range(10):
    al_id = f'al-{i}'
    rects2[al_id] = {'x': 50, 'y': 50 + (9 - i) * 60, 'w': 400, 'h': 40}

result2 = analyzer.analyze(rects2, html)
print(f"Reversed:     tau={result2.correlation_score:.3f}, severity={result2.severity}, mismatches={result2.mismatch_count}")
assert result2.severity in ('warning', 'critical'), f"Expected warning/critical, got {result2.severity}"
assert result2.mismatch_count > 0, "Expected mismatches"
assert len(result2.issues) > 0, "Expected WCAG issues"
print(f"  -> {len(result2.issues)} WCAG 1.3.2 issues generated")

# Test to_dict serialization
d = result2.to_dict()
assert 'correlation_score' in d
assert 'visual_order_map' in d
print(f"  -> to_dict() serialization OK")

print("\n[OK] All reading order tests passed!")

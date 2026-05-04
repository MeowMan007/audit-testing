"""
Reading Order Analyzer — Visual vs DOM Order Mismatch Detection.

NOVEL FEATURE: No existing accessibility tool implements this check.

Detects when the visual layout order (what sighted users perceive) diverges
from the DOM source order (what screen readers follow). This violates
WCAG 1.3.2 — Meaningful Sequence.

HOW IT WORKS:
  1. Extracts meaningful content elements from the bounding box data
     that PageFetcher already collects via getBoundingClientRect().
  2. Sorts elements by visual position (top-to-bottom, left-to-right),
     grouping elements into visual "rows" by Y-coordinate proximity.
  3. Compares the visual order to the DOM source order (the sequential
     al-0, al-1, al-2... IDs assigned during DOM traversal).
  4. Computes Kendall's Tau rank correlation between the two orderings.
  5. Flags elements with large positional drift as WCAG 1.3.2 issues.

INTERPRETATION:
  tau ~1.0  → Visual and DOM order match perfectly (good)
  tau ~0.7  → Moderate mismatch — some elements are out of order
  tau <0.4  → Severe mismatch — screen reader users hear a different page
"""
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from backend.models.schemas import AccessibilityIssue, SeverityLevel, WCAGLevel

logger = logging.getLogger(__name__)

# Tags that carry meaningful content (skip wrappers like div, span)
CONTENT_TAGS = {
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'p', 'a', 'button', 'img', 'input', 'select', 'textarea',
    'li', 'td', 'th', 'label', 'figcaption', 'blockquote',
    'nav', 'main', 'article', 'section', 'header', 'footer',
}

# Minimum dimensions to consider an element visible and meaningful
MIN_WIDTH = 20
MIN_HEIGHT = 10
# Y-coordinate tolerance for grouping elements into the same visual row
ROW_TOLERANCE = 25
# Minimum rank drift to consider an element "mismatched"
DRIFT_THRESHOLD = 5
# Minimum elements to perform a meaningful analysis
MIN_ELEMENTS = 6


@dataclass
class ReadingOrderResult:
    """Results from the reading order analysis."""
    correlation_score: float = 0.0
    mismatch_count: int = 0
    total_elements_analyzed: int = 0
    severity: str = "pass"
    mismatched_elements: List[dict] = field(default_factory=list)
    visual_order_map: List[dict] = field(default_factory=list)
    issues: List[AccessibilityIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "correlation_score": round(self.correlation_score, 3),
            "mismatch_count": self.mismatch_count,
            "total_elements_analyzed": self.total_elements_analyzed,
            "severity": self.severity,
            "mismatched_elements": self.mismatched_elements,
            "visual_order_map": self.visual_order_map[:50],  # Cap for JSON size
        }


def _kendall_tau(order_a: List[int], order_b: List[int]) -> float:
    """Compute Kendall's Tau rank correlation without scipy.
    
    Returns a value between -1 (perfectly inverted) and 1 (identical order).
    Implemented manually to avoid adding scipy as a dependency.
    """
    n = len(order_a)
    if n < 2:
        return 1.0

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Compare pair ordering in both sequences
            diff_a = order_a[i] - order_a[j]
            diff_b = order_b[i] - order_b[j]
            product = diff_a * diff_b

            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            # Ties (product == 0) are ignored

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 1.0

    return (concordant - discordant) / total_pairs


def _extract_tag_from_html(html: str, al_id: str) -> Tuple[str, str]:
    """Extract the tag name and a text preview for an element by its data-al-id.
    
    Does a simple regex search — not perfect, but good enough for display purposes.
    """
    pattern = rf'<(\w+)\s[^>]*data-al-id="{re.escape(al_id)}"[^>]*>(.*?)</\1'
    match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
    if match:
        tag = match.group(1).lower()
        text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        text = ' '.join(text.split())[:60]  # Collapse whitespace, truncate
        return tag, text if text else f"<{tag}>"

    # Fallback: just find the tag name
    pattern2 = rf'<(\w+)\s[^>]*data-al-id="{re.escape(al_id)}"'
    match2 = re.search(pattern2, html, re.IGNORECASE)
    if match2:
        return match2.group(1).lower(), ""
    return "unknown", ""


class ReadingOrderAnalyzer:
    """Analyzes whether visual reading order matches DOM source order."""

    def analyze(
        self,
        element_rects: Dict[str, dict],
        html: str,
    ) -> ReadingOrderResult:
        """Run the reading order analysis.
        
        Args:
            element_rects: Bounding boxes from PageFetcher
                           {"al-0": {"x": 10, "y": 20, "w": 100, "h": 30}, ...}
            html: Rendered HTML source (contains data-al-id attributes)
            
        Returns:
            ReadingOrderResult with correlation score and mismatched elements
        """
        result = ReadingOrderResult()

        if not element_rects or len(element_rects) < MIN_ELEMENTS:
            logger.info(f"Too few elements for reading order analysis ({len(element_rects or {})})")
            result.severity = "pass"
            return result

        # Step 1: Filter to meaningful content elements
        content_elements = []
        for al_id, bbox in element_rects.items():
            # Skip tiny/invisible elements
            if bbox.get('w', 0) < MIN_WIDTH or bbox.get('h', 0) < MIN_HEIGHT:
                continue

            tag, text_preview = _extract_tag_from_html(html, al_id)

            # Only keep content-bearing tags
            if tag not in CONTENT_TAGS:
                continue

            # Extract the sequential number from the al-id for DOM order
            id_num = int(al_id.replace('al-', ''))

            content_elements.append({
                'al_id': al_id,
                'dom_index': id_num,
                'tag': tag,
                'text': text_preview,
                'x': bbox.get('x', 0),
                'y': bbox.get('y', 0),
                'w': bbox.get('w', 0),
                'h': bbox.get('h', 0),
            })

        if len(content_elements) < MIN_ELEMENTS:
            logger.info(f"Too few content elements after filtering ({len(content_elements)})")
            result.severity = "pass"
            return result

        # Step 2: Sort by DOM order (already have dom_index)
        content_elements.sort(key=lambda e: e['dom_index'])
        dom_ordered = list(content_elements)

        # Step 3: Sort by visual reading order (top-to-bottom, left-to-right)
        # Group into rows first: elements within ROW_TOLERANCE px of each other
        # are considered the same visual row, then sorted left-to-right within each row
        visual_sorted = sorted(content_elements, key=lambda e: (e['y'], e['x']))

        # Group into rows by Y proximity
        rows = []
        current_row = [visual_sorted[0]]
        for elem in visual_sorted[1:]:
            if abs(elem['y'] - current_row[0]['y']) <= ROW_TOLERANCE:
                current_row.append(elem)
            else:
                rows.append(current_row)
                current_row = [elem]
        rows.append(current_row)

        # Sort within each row by X, then flatten
        visual_ordered = []
        for row in rows:
            row.sort(key=lambda e: e['x'])
            visual_ordered.extend(row)

        # Step 4: Assign ranks
        dom_id_to_rank = {e['al_id']: rank for rank, e in enumerate(dom_ordered)}
        visual_id_to_rank = {e['al_id']: rank for rank, e in enumerate(visual_ordered)}

        # Build rank arrays (same element ordering for both)
        dom_ranks = []
        visual_ranks = []
        for elem in dom_ordered:
            dom_ranks.append(dom_id_to_rank[elem['al_id']])
            visual_ranks.append(visual_id_to_rank[elem['al_id']])

        # Step 5: Compute Kendall's Tau
        tau = _kendall_tau(dom_ranks, visual_ranks)

        # Step 6: Find mismatched elements (large rank drift)
        mismatched = []
        visual_order_map = []

        for elem in dom_ordered:
            d_rank = dom_id_to_rank[elem['al_id']]
            v_rank = visual_id_to_rank[elem['al_id']]
            drift = abs(d_rank - v_rank)

            entry = {
                'al_id': elem['al_id'],
                'tag': elem['tag'],
                'text': elem['text'],
                'dom_rank': d_rank + 1,   # 1-indexed for display
                'visual_rank': v_rank + 1,
                'drift': drift,
                'bbox': {'x': elem['x'], 'y': elem['y'], 'w': elem['w'], 'h': elem['h']},
            }
            visual_order_map.append(entry)

            if drift >= DRIFT_THRESHOLD:
                mismatched.append(entry)

        # Sort mismatched by drift (worst first)
        mismatched.sort(key=lambda e: e['drift'], reverse=True)

        # Step 7: Determine severity
        if tau >= 0.8:
            severity = "pass"
        elif tau >= 0.5:
            severity = "warning"
        else:
            severity = "critical"

        # Step 8: Generate WCAG issues for mismatches
        issues = []
        if severity != "pass":
            # Overall reading order issue
            issues.append(AccessibilityIssue(
                id="reading-order-mismatch",
                wcag_criterion="1.3.2",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.CRITICAL if severity == "critical" else SeverityLevel.WARNING,
                title=f"Reading Order Mismatch (τ={tau:.2f})",
                description=(
                    f"The visual layout order of content differs significantly from the DOM source order. "
                    f"Kendall's Tau correlation is {tau:.2f} (1.0 = perfect match). "
                    f"{len(mismatched)} element(s) are in significantly different positions visually "
                    f"compared to their DOM order. Screen reader users will hear content in a different "
                    f"sequence than what sighted users see."
                ),
                element=f"{len(mismatched)} elements with rank drift ≥ {DRIFT_THRESHOLD}",
                suggestion=(
                    "Review CSS properties like flexbox 'order', CSS Grid placement, and "
                    "'position: absolute/fixed' that may reorder content visually without changing "
                    "the DOM. Ensure the DOM source order matches the intended reading sequence, "
                    "or use aria-flowto to explicitly define the reading order."
                ),
                score_impact=5.0 if severity == "critical" else 3.0,
            ))

            # Individual element issues (top 3 worst offenders)
            for i, elem in enumerate(mismatched[:3]):
                issues.append(AccessibilityIssue(
                    id=f"reading-order-drift-{i+1}",
                    wcag_criterion="1.3.2",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.WARNING,
                    title=f"Element Out of Reading Order: <{elem['tag']}>",
                    description=(
                        f"The <{elem['tag']}> element "
                        f"{'\"' + elem['text'][:40] + '\"' if elem['text'] else ''} "
                        f"appears at DOM position #{elem['dom_rank']} but is visually at "
                        f"position #{elem['visual_rank']} (drift of {elem['drift']} positions)."
                    ),
                    element=f"<{elem['tag']}> at ({elem['bbox']['x']:.0f}, {elem['bbox']['y']:.0f})",
                    data_al_id=elem['al_id'],
                    suggestion=(
                        f"Move this element in the DOM to match its visual position, or adjust "
                        f"the CSS so its visual position matches its DOM order."
                    ),
                    score_impact=2.0,
                ))

        # Build result
        result.correlation_score = tau
        result.mismatch_count = len(mismatched)
        result.total_elements_analyzed = len(content_elements)
        result.severity = severity
        result.mismatched_elements = mismatched[:10]  # Cap at 10
        result.visual_order_map = visual_order_map
        result.issues = issues

        logger.info(
            f"Reading order analysis: τ={tau:.3f}, {len(mismatched)} mismatches "
            f"out of {len(content_elements)} elements → {severity}"
        )

        return result

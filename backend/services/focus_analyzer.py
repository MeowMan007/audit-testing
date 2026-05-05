"""
Focus Trap & Keyboard Reachability Analyzer — WCAG 2.1.1 / 2.1.2.

NOVEL FEATURE: No existing accessibility tool (Lighthouse, WAVE, axe-core)
can detect keyboard traps because they are static, in-browser analyzers.
AccessLens uses Selenium as an EXTERNAL orchestrator to send real Tab
keystrokes and build the complete focus graph of the page.

HOW IT WORKS:
  1. PageFetcher sends Tab key presses and records document.activeElement
     after each press, producing a focus_path (ordered list of element IDs).
  2. This analyzer walks the focus_path to detect:
     a) KEYBOARD TRAPS — cyclic sub-loops where focus revisits a small
        subset of elements without ever reaching the rest of the page.
        (Violates WCAG 2.1.2 — No Keyboard Trap)
     b) UNREACHABLE ELEMENTS — interactive elements (links, buttons, inputs)
        that never appear in the focus_path, meaning keyboard-only users
        can never reach them.
        (Violates WCAG 2.1.1 — Keyboard)

ALGORITHM (Cycle Detection):
  - Walk the path and track the "visited" set.
  - When an element is revisited, extract the sub-path between the two
    occurrences → that's a potential cycle.
  - If the cycle length is small relative to the total expected focusable
    elements AND the cycle repeats, it's a confirmed trap.

INTERPRETATION:
  reachability 100%  → All interactive elements are keyboard-accessible (good)
  reachability <80%  → Significant content is unreachable via keyboard
  has_trap = True    → CRITICAL: users cannot escape a focus loop
"""
import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field

from backend.models.schemas import AccessibilityIssue, SeverityLevel, WCAGLevel

logger = logging.getLogger(__name__)

# Minimum focusable elements to run analysis
MIN_FOCUSABLE = 3
# A cycle must be this much smaller than total focusable to be a trap
TRAP_RATIO_THRESHOLD = 0.5
# Minimum cycle repetitions to confirm a trap
MIN_CYCLE_REPEATS = 2
# Maximum unreachable elements to report individually
MAX_UNREACHABLE_ISSUES = 5


@dataclass
class FocusAnalysisResult:
    """Results from the keyboard focus analysis."""
    has_trap: bool = False
    trap_cycle: List[str] = field(default_factory=list)
    trap_cycle_tags: List[dict] = field(default_factory=list)
    unreachable: List[str] = field(default_factory=list)
    unreachable_details: List[dict] = field(default_factory=list)
    total_focusable: int = 0
    total_reached: int = 0
    reachability_pct: float = 100.0
    focus_path_length: int = 0
    focus_path_summary: List[str] = field(default_factory=list)
    issues: List[AccessibilityIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_trap": self.has_trap,
            "trap_cycle": self.trap_cycle,
            "trap_cycle_tags": self.trap_cycle_tags,
            "unreachable": self.unreachable[:20],
            "unreachable_details": self.unreachable_details[:20],
            "total_focusable": self.total_focusable,
            "total_reached": self.total_reached,
            "reachability_pct": round(self.reachability_pct, 1),
            "focus_path_length": self.focus_path_length,
            "focus_path_summary": self.focus_path_summary[:30],
        }


def _extract_element_info(html: str, al_id: str) -> dict:
    """Extract tag name, text preview, and type for an element by data-al-id."""
    info = {"al_id": al_id, "tag": "unknown", "text": "", "type": ""}

    # Find the opening tag
    pattern = rf'<(\w+)\s[^>]*data-al-id="{re.escape(al_id)}"[^>]*>(.*?)</\1'
    match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
    if match:
        info["tag"] = match.group(1).lower()
        text = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        info["text"] = ' '.join(text.split())[:60]
    else:
        # Fallback: just find the tag name
        pattern2 = rf'<(\w+)\s[^>]*data-al-id="{re.escape(al_id)}"'
        match2 = re.search(pattern2, html, re.IGNORECASE)
        if match2:
            info["tag"] = match2.group(1).lower()

    # Try to extract type attribute (for inputs)
    type_pattern = rf'data-al-id="{re.escape(al_id)}"[^>]*type="([^"]*)"'
    type_match = re.search(type_pattern, html, re.IGNORECASE)
    if type_match:
        info["type"] = type_match.group(1)

    return info


def _detect_cycles(focus_path: List[str]) -> Tuple[bool, List[str]]:
    """Detect repeating sub-cycles in the focus path.
    
    Walks the path looking for the first element that is revisited.
    Extracts the cycle between the two occurrences, then checks if
    the cycle repeats consistently (indicating a trap).
    
    Returns:
        (has_trap, cycle_elements) — True + the trapped element IDs,
        or False + empty list.
    """
    if len(focus_path) < 4:
        return False, []

    # Track first occurrence index of each element
    first_seen: Dict[str, int] = {}

    for i, el_id in enumerate(focus_path):
        if not el_id or el_id == "body" or el_id == "null":
            continue

        if el_id in first_seen:
            # Found a revisit — extract the cycle
            cycle_start = first_seen[el_id]
            cycle = focus_path[cycle_start:i]

            if len(cycle) < 2:
                # Single-element "cycle" isn't meaningful
                first_seen[el_id] = i
                continue

            # Check if this cycle repeats after this point
            cycle_len = len(cycle)
            repeats = 1
            pos = i

            while pos + cycle_len <= len(focus_path):
                window = focus_path[pos:pos + cycle_len]
                if window == cycle:
                    repeats += 1
                    pos += cycle_len
                else:
                    break

            if repeats >= MIN_CYCLE_REPEATS:
                # Confirmed trap: cycle repeats multiple times
                unique_in_cycle = list(dict.fromkeys(cycle))  # Preserve order, dedupe
                return True, unique_in_cycle

        first_seen[el_id] = i

    return False, []


class FocusAnalyzer:
    """Analyzes keyboard focus paths for traps and unreachable elements."""

    def analyze(
        self,
        focus_path: List[str],
        expected_focusable: List[str],
        element_rects: Dict[str, dict] = None,
        html: str = "",
    ) -> FocusAnalysisResult:
        """Run the focus trap and reachability analysis.
        
        Args:
            focus_path: Ordered list of data-al-id values from Tab simulation
            expected_focusable: List of data-al-id values for all theoretically
                               focusable elements on the page
            element_rects: Bounding boxes (for position info in reports)
            html: Page HTML (for extracting element info for display)
            
        Returns:
            FocusAnalysisResult with trap detection and reachability data
        """
        result = FocusAnalysisResult()
        result.focus_path_length = len(focus_path)

        if not focus_path or len(expected_focusable) < MIN_FOCUSABLE:
            logger.info(
                f"Skipping focus analysis: path={len(focus_path)}, "
                f"expected={len(expected_focusable)}"
            )
            return result

        result.total_focusable = len(expected_focusable)

        # Clean the focus path — remove null/body entries
        clean_path = [
            el for el in focus_path
            if el and el != "null" and el != "body"
        ]

        # Build focus path summary (first 30 unique in order)
        seen_summary: Set[str] = set()
        for el in clean_path:
            if el not in seen_summary:
                seen_summary.add(el)
                result.focus_path_summary.append(el)
            if len(result.focus_path_summary) >= 30:
                break

        # ── Step 1: Cycle Detection (WCAG 2.1.2) ──
        has_trap, trap_cycle = _detect_cycles(clean_path)

        if has_trap:
            # Verify it's a real trap: the cycle must be significantly smaller
            # than the total focusable elements
            cycle_ratio = len(trap_cycle) / max(1, len(expected_focusable))

            if cycle_ratio < TRAP_RATIO_THRESHOLD:
                result.has_trap = True
                result.trap_cycle = trap_cycle

                # Get element details for the trapped elements
                for al_id in trap_cycle:
                    info = _extract_element_info(html, al_id)
                    if element_rects and al_id in element_rects:
                        info["bbox"] = element_rects[al_id]
                    result.trap_cycle_tags.append(info)

                logger.warning(
                    f"KEYBOARD TRAP DETECTED: {len(trap_cycle)} elements in cycle "
                    f"({cycle_ratio:.0%} of {len(expected_focusable)} focusable)"
                )
            else:
                logger.info(
                    f"Cycle found but covers {cycle_ratio:.0%} of focusable "
                    f"elements — likely normal tab cycling, not a trap"
                )

        # ── Step 2: Reachability Analysis (WCAG 2.1.1) ──
        reached_set = set(clean_path)
        expected_set = set(expected_focusable)
        unreachable = expected_set - reached_set

        result.total_reached = len(reached_set & expected_set)
        result.unreachable = sorted(list(unreachable))

        if result.total_focusable > 0:
            result.reachability_pct = (
                result.total_reached / result.total_focusable
            ) * 100
        else:
            result.reachability_pct = 100.0

        # Get details for unreachable elements
        for al_id in list(unreachable)[:20]:
            info = _extract_element_info(html, al_id)
            if element_rects and al_id in element_rects:
                info["bbox"] = element_rects[al_id]
            result.unreachable_details.append(info)

        # ── Step 3: Generate WCAG Issues ──
        issues = []

        # WCAG 2.1.2 — No Keyboard Trap
        if result.has_trap:
            trap_elements = ", ".join(
                f"<{t['tag']}>{' \"' + t['text'][:30] + '\"' if t['text'] else ''}"
                for t in result.trap_cycle_tags[:5]
            )
            issues.append(AccessibilityIssue(
                id="keyboard-trap-detected",
                wcag_criterion="2.1.2",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.CRITICAL,
                title=f"Keyboard Trap Detected ({len(result.trap_cycle)} elements)",
                description=(
                    f"A keyboard focus trap was detected. When navigating with the Tab key, "
                    f"focus becomes trapped in a cycle of {len(result.trap_cycle)} elements "
                    f"and cannot reach the remaining {len(unreachable)} interactive elements "
                    f"on the page. Trapped elements: {trap_elements}. "
                    f"This prevents keyboard-only users from accessing the full page content."
                ),
                element=f"Cycle: {' → '.join(result.trap_cycle[:6])}{'...' if len(result.trap_cycle) > 6 else ''}",
                suggestion=(
                    "Ensure all modal dialogs, dropdown menus, and custom widgets allow users "
                    "to exit using the Escape key or Tab. Check for JavaScript event handlers "
                    "that forcibly redirect focus (e.g., onfocus, onblur handlers that call "
                    ".focus() on a sibling element). Add proper focus management with "
                    "aria-modal='true' and return focus to the trigger element on close."
                ),
                score_impact=8.0,
            ))

        # WCAG 2.1.1 — Keyboard (unreachable elements)
        if unreachable and result.reachability_pct < 90:
            issues.append(AccessibilityIssue(
                id="keyboard-unreachable-content",
                wcag_criterion="2.1.1",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.CRITICAL if result.reachability_pct < 70 else SeverityLevel.WARNING,
                title=f"Keyboard Unreachable Elements ({len(unreachable)} of {result.total_focusable})",
                description=(
                    f"Only {result.reachability_pct:.0f}% of interactive elements "
                    f"({result.total_reached} of {result.total_focusable}) can be reached "
                    f"using keyboard Tab navigation. {len(unreachable)} elements are completely "
                    f"inaccessible to keyboard-only users, screen reader users, and users "
                    f"of switch devices."
                ),
                element=f"{len(unreachable)} unreachable elements",
                suggestion=(
                    "Ensure all interactive elements (links, buttons, inputs, custom controls) "
                    "are reachable via the Tab key. Check for tabindex='-1' on elements that "
                    "should be focusable, hidden elements that are interactive, and custom "
                    "JavaScript widgets that don't implement keyboard support. Use native "
                    "HTML elements where possible."
                ),
                score_impact=5.0 if result.reachability_pct < 70 else 3.0,
            ))

            # Individual unreachable element issues (top offenders)
            for i, detail in enumerate(result.unreachable_details[:MAX_UNREACHABLE_ISSUES]):
                tag = detail.get('tag', 'unknown')
                text = detail.get('text', '')
                issues.append(AccessibilityIssue(
                    id=f"keyboard-unreachable-{i+1}",
                    wcag_criterion="2.1.1",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.WARNING,
                    title=f"Unreachable: <{tag}>{' \"' + text[:30] + '\"' if text else ''}",
                    description=(
                        f"The <{tag}> element "
                        f"{'\"' + text[:40] + '\" ' if text else ''}"
                        f"cannot be reached using keyboard navigation. "
                        f"Keyboard-only users will not be able to interact with this element."
                    ),
                    element=f"<{tag}> ({detail.get('al_id', 'unknown')})",
                    data_al_id=detail.get('al_id'),
                    suggestion=(
                        f"Make this <{tag}> element focusable by ensuring it uses a native "
                        f"interactive HTML element, or add tabindex='0' and appropriate "
                        f"ARIA roles and keyboard event handlers."
                    ),
                    score_impact=2.0,
                ))

        result.issues = issues

        logger.info(
            f"Focus analysis: trap={'YES' if result.has_trap else 'no'}, "
            f"reached={result.total_reached}/{result.total_focusable} "
            f"({result.reachability_pct:.0f}%), "
            f"{len(unreachable)} unreachable, {len(issues)} issues"
        )

        return result

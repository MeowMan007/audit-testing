"""
Rule Engine — WCAG 2.1 Rule-Based Accessibility Checks.

Implements 15+ deterministic checks based on WCAG 2.1 success criteria.
Each check method returns a list of AccessibilityIssue objects.

This is the "traditional" part of the audit pipeline. It catches
violations that can be detected programmatically without AI:
- Missing alt text, empty links, form labels
- Heading hierarchy violations
- Color contrast (from computed styles)
- ARIA validity, landmark structure
- Page metadata (title, language)
- Keyboard accessibility indicators

The AI/CNN model (in dl_engine.py) supplements these checks with
visual analysis that rule-based approaches cannot perform.

Rule coverage summary:
  WCAG 1.1.1  - Non-text content (images)
  WCAG 1.3.1  - Info and relationships (headings, forms)
  WCAG 1.4.3  - Contrast minimum
  WCAG 2.4.1  - Bypass blocks (skip links)
  WCAG 2.4.2  - Page titled
  WCAG 2.4.4  - Link purpose
  WCAG 2.4.6  - Headings and labels
  WCAG 3.1.1  - Language of page
  WCAG 3.3.2  - Labels or instructions
  WCAG 4.1.1  - Parsing (duplicate IDs)
  WCAG 4.1.2  - Name, role, value (ARIA, buttons)
"""
import re
import math
import logging
from typing import List, Optional, Tuple

from backend.models.schemas import AccessibilityIssue, SeverityLevel, WCAGLevel
from backend.models.wcag_rules import (
    VALID_ARIA_ROLES,
    GENERIC_LINK_TEXTS,
    WCAG_CRITERIA,
    CONTRAST_RATIOS,
)
from backend.services.dom_analyzer import DOMData

logger = logging.getLogger(__name__)


class RuleEngine:
    """Runs all WCAG rule-based checks against parsed DOM data.
    
    Usage:
        engine = RuleEngine()
        issues = engine.run_all_checks(dom_data, computed_styles)
    """

    def run_all_checks(
        self,
        dom_data: DOMData,
        computed_styles: Optional[list] = None,
    ) -> List[AccessibilityIssue]:
        """Execute all WCAG checks and return discovered issues.
        
        Args:
            dom_data: Structured DOM data from DOMAnalyzer
            computed_styles: CSS computed styles from Selenium (for contrast)
            
        Returns:
            List of AccessibilityIssue objects, sorted by severity
        """
        issues: List[AccessibilityIssue] = []
        issue_counter = {"count": 0}

        def _id(prefix: str) -> str:
            issue_counter["count"] += 1
            return f"{prefix}-{issue_counter['count']}"

        # --- WCAG 2.4.2: Page Title ---
        issues.extend(self._check_page_title(dom_data, _id))

        # --- WCAG 3.1.1: Language ---
        issues.extend(self._check_language(dom_data, _id))

        # --- WCAG 1.1.1: Image Alt Text ---
        issues.extend(self._check_images(dom_data, _id))

        # --- WCAG 2.4.6: Heading Hierarchy ---
        issues.extend(self._check_headings(dom_data, _id))

        # --- WCAG 2.4.4: Link Purpose ---
        issues.extend(self._check_links(dom_data, _id))

        # --- WCAG 3.3.2: Form Labels ---
        issues.extend(self._check_form_labels(dom_data, _id))

        # --- WCAG 4.1.2: ARIA Validity ---
        issues.extend(self._check_aria(dom_data, _id))

        # --- WCAG 2.4.1: Skip Link ---
        issues.extend(self._check_skip_link(dom_data, _id))

        # --- WCAG 4.1.1: Duplicate IDs ---
        issues.extend(self._check_duplicate_ids(dom_data, _id))

        # --- WCAG 1.4.3: Color Contrast ---
        if computed_styles:
            issues.extend(self._check_contrast(computed_styles, _id))

        # --- WCAG 1.4.4: Viewport Scaling ---
        issues.extend(self._check_viewport(dom_data, _id))

        # --- Tables ---
        issues.extend(self._check_tables(dom_data, _id))

        # --- Iframes ---
        issues.extend(self._check_iframes(dom_data, _id))

        # --- Buttons ---
        issues.extend(self._check_buttons(dom_data, _id))

        # --- Focus Outlines ---
        issues.extend(self._check_focus_outlines(dom_data, _id))

        # Sort: critical first, then warning, then info
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        issues.sort(key=lambda x: severity_order.get(x.severity.value, 3))

        logger.info(f"Rule engine found {len(issues)} issues")
        return issues

    # ================================================================
    # Individual Check Methods
    # ================================================================

    def _check_page_title(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 2.4.2: Web pages must have descriptive titles."""
        issues = []
        if not dom.title:
            issues.append(AccessibilityIssue(
                id=_id("title"),
                wcag_criterion="2.4.2",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.CRITICAL,
                title="Missing Page Title",
                description="The page does not have a <title> element. Screen readers announce the page title when users navigate between tabs.",
                element="<title> (missing)",
                suggestion="Add a descriptive <title> tag inside <head> that describes the page content.",
                score_impact=5.0,
            ))
        elif len(dom.title) < 3:
            issues.append(AccessibilityIssue(
                id=_id("title"),
                wcag_criterion="2.4.2",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.WARNING,
                title="Non-Descriptive Page Title",
                description=f"The page title '{dom.title}' is too short to be meaningful.",
                element=f"<title>{dom.title}</title>",
                suggestion="Use a descriptive title that summarizes the page content (e.g., 'Home - Company Name').",
                score_impact=3.0,
            ))
        return issues

    def _check_language(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 3.1.1: Page must declare a language."""
        issues = []
        if not dom.lang:
            issues.append(AccessibilityIssue(
                id=_id("lang"),
                wcag_criterion="3.1.1",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.CRITICAL,
                title="Missing Language Declaration",
                description="The <html> element does not have a 'lang' attribute. Screen readers use this to select the correct pronunciation rules.",
                element='<html> (missing lang attribute)',
                suggestion='Add lang="en" (or appropriate language code) to the <html> tag.',
                score_impact=5.0,
            ))
        return issues

    def _check_images(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 1.1.1: All images must have text alternatives."""
        issues = []
        for img in dom.images:
            src = img.get("src", "unknown")
            src_short = src.split("/")[-1][:50] if src else "unknown"

            if not img.get("has_alt"):
                # No alt attribute at all — critical
                if not img.get("aria_label") and not img.get("aria_labelledby"):
                    issues.append(AccessibilityIssue(
                        id=_id("img-alt"),
                        wcag_criterion="1.1.1",
                        wcag_level=WCAGLevel.A,
                        severity=SeverityLevel.CRITICAL,
                        title="Image Missing Alt Text",
                        description=f"Image '{src_short}' has no alt attribute. Screen readers cannot describe this image to users.",
                        element=f'<img src="{src_short}">',
                        data_al_id=img.get("data_al_id"),
                        suggestion=f'Add alt="descriptive text" to the <img> tag. If decorative, use alt="" and role="presentation".',
                        score_impact=5.0,
                    ))
            elif img.get("alt") and len(img["alt"]) > 150:
                # Alt text is excessively long
                issues.append(AccessibilityIssue(
                    id=_id("img-alt-long"),
                    wcag_criterion="1.1.1",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.WARNING,
                    title="Excessively Long Alt Text",
                    description=f"Image '{src_short}' has alt text with {len(img['alt'])} characters. Alt text should be concise.",
                    element=f'<img alt="{img["alt"][:50]}...">',
                    data_al_id=img.get("data_al_id"),
                    suggestion="Keep alt text under 125 characters. Use longdesc or aria-describedby for detailed descriptions.",
                    score_impact=2.0,
                ))
        return issues

    def _check_headings(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 2.4.6: Heading hierarchy must not skip levels."""
        issues = []

        if not dom.headings:
            issues.append(AccessibilityIssue(
                id=_id("heading"),
                wcag_criterion="2.4.6",
                wcag_level=WCAGLevel.AA,
                severity=SeverityLevel.WARNING,
                title="No Headings Found",
                description="The page has no heading elements (h1-h6). Headings provide document structure for screen reader navigation.",
                suggestion="Add heading elements to structure your content. Start with a single <h1> for the main topic.",
                score_impact=3.0,
            ))
            return issues

        # Check for missing h1
        h1_count = sum(1 for h in dom.headings if h["level"] == 1)
        if h1_count == 0:
            issues.append(AccessibilityIssue(
                id=_id("heading-h1"),
                wcag_criterion="2.4.6",
                wcag_level=WCAGLevel.AA,
                severity=SeverityLevel.WARNING,
                title="Missing H1 Heading",
                description="No <h1> element found. Every page should have exactly one <h1> as the main heading.",
                suggestion="Add an <h1> element that describes the main content of the page.",
                score_impact=3.0,
            ))
        elif h1_count > 1:
            issues.append(AccessibilityIssue(
                id=_id("heading-h1-multi"),
                wcag_criterion="2.4.6",
                wcag_level=WCAGLevel.AA,
                severity=SeverityLevel.WARNING,
                title="Multiple H1 Headings",
                description=f"Found {h1_count} <h1> elements. Best practice is to have exactly one <h1> per page.",
                suggestion="Keep only one <h1> as the primary page heading. Demote others to <h2> or lower.",
                score_impact=2.0,
            ))

        # Check for skipped heading levels
        prev_level = 0
        for heading in dom.headings:
            level = heading["level"]
            if prev_level > 0 and level > prev_level + 1:
                issues.append(AccessibilityIssue(
                    id=_id("heading-skip"),
                    wcag_criterion="2.4.6",
                    wcag_level=WCAGLevel.AA,
                    severity=SeverityLevel.WARNING,
                    title=f"Skipped Heading Level (h{prev_level} → h{level})",
                    description=f"Heading jumps from <h{prev_level}> to <h{level}>, skipping level(s). This confuses screen reader navigation.",
                    element=f'<h{level}>{heading["text"][:60]}</h{level}>',
                    data_al_id=heading.get("data_al_id"),
                    suggestion=f"Use <h{prev_level + 1}> instead, or add the missing intermediate heading level(s).",
                    score_impact=2.0,
                ))
            prev_level = level

        # Check for empty headings
        for heading in dom.headings:
            if not heading["text"].strip():
                issues.append(AccessibilityIssue(
                    id=_id("heading-empty"),
                    wcag_criterion="2.4.6",
                    wcag_level=WCAGLevel.AA,
                    severity=SeverityLevel.WARNING,
                    title=f"Empty Heading (h{heading['level']})",
                    description=f"An <h{heading['level']}> element has no text content.",
                    element=f'<h{heading["level"]}></h{heading["level"]}>',
                    data_al_id=heading.get("data_al_id"),
                    suggestion="Add descriptive text to the heading, or remove it if not needed.",
                    score_impact=2.0,
                ))

        return issues

    def _check_links(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 2.4.4: Link purpose must be determinable from link text."""
        issues = []
        for link in dom.links:
            text = link.get("text", "").strip()
            aria = link.get("aria_label", "").strip()
            accessible_name = text or aria

            if not accessible_name:
                issues.append(AccessibilityIssue(
                    id=_id("link-empty"),
                    wcag_criterion="2.4.4",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.CRITICAL,
                    title="Empty Link",
                    description="A link has no text content and no aria-label. Users cannot determine where this link goes.",
                    element=f'<a href="{link.get("href", "")[:60]}"></a>',
                    data_al_id=link.get("data_al_id"),
                    suggestion="Add descriptive text inside the <a> tag, or add an aria-label attribute.",
                    score_impact=4.0,
                ))
            elif accessible_name.lower() in GENERIC_LINK_TEXTS:
                issues.append(AccessibilityIssue(
                    id=_id("link-generic"),
                    wcag_criterion="2.4.4",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.WARNING,
                    title=f'Generic Link Text: "{accessible_name}"',
                    description=f'Link text "{accessible_name}" is too generic. Users (especially screen reader users) cannot determine the link purpose.',
                    element=f'<a href="...">{accessible_name}</a>',
                    data_al_id=link.get("data_al_id"),
                    suggestion=f'Replace "{accessible_name}" with specific text describing the destination (e.g., "Read our pricing guide").',
                    score_impact=2.0,
                ))
        return issues

    def _check_form_labels(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 3.3.2: Form inputs must have labels or instructions."""
        issues = []
        for inp in dom.form_inputs:
            if not inp.get("has_label") and not inp.get("has_aria_label"):
                name = inp.get("name") or inp.get("id") or inp.get("type", "input")
                issues.append(AccessibilityIssue(
                    id=_id("form-label"),
                    wcag_criterion="3.3.2",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.CRITICAL,
                    title=f'Form Input Missing Label ({name})',
                    description=f'A <{inp["tag"]}> element (type="{inp["type"]}") has no associated <label> or aria-label. Screen readers cannot identify this field.',
                    element=f'<{inp["tag"]} type="{inp["type"]}" name="{name}">',
                    data_al_id=inp.get("data_al_id"),
                    suggestion=f'Add <label for="{inp.get("id", name)}">{name}</label> before the input, or add aria-label="{name}".',
                    score_impact=4.0,
                ))
        return issues

    def _check_aria(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 4.1.2: ARIA roles must be valid."""
        issues = []
        for el in dom.aria_elements:
            role = el.get("role", "")
            if role and role not in VALID_ARIA_ROLES:
                issues.append(AccessibilityIssue(
                    id=_id("aria-role"),
                    wcag_criterion="4.1.2",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.WARNING,
                    title=f'Invalid ARIA Role: "{role}"',
                    description=f'Element <{el["tag"]}> uses an invalid ARIA role "{role}". This confuses assistive technologies.',
                    element=f'<{el["tag"]} role="{role}">',
                    suggestion=f'Use a valid WAI-ARIA role. See https://www.w3.org/TR/wai-aria-1.2/#role_definitions',
                    score_impact=3.0,
                ))
        return issues

    def _check_skip_link(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 2.4.1: Provide a mechanism to skip repeated content."""
        issues = []
        if not dom.has_skip_link and len(dom.links) > 5:
            issues.append(AccessibilityIssue(
                id=_id("skip-link"),
                wcag_criterion="2.4.1",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.WARNING,
                title="Missing Skip Navigation Link",
                description="No skip-to-content link found. Keyboard users must tab through all navigation links to reach main content.",
                suggestion='Add <a href="#main-content" class="skip-link">Skip to main content</a> as the first focusable element.',
                score_impact=3.0,
            ))
        return issues

    def _check_duplicate_ids(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 4.1.1: IDs must be unique within the page."""
        issues = []
        for dup_id in dom.duplicate_ids:
            issues.append(AccessibilityIssue(
                id=_id("dup-id"),
                wcag_criterion="4.1.1",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.WARNING,
                title=f'Duplicate ID: "{dup_id}"',
                description=f'The ID "{dup_id}" is used on multiple elements. IDs must be unique for ARIA references and label associations to work correctly.',
                element=f'id="{dup_id}" (duplicated)',
                suggestion=f'Ensure each element has a unique ID. Rename duplicates (e.g., "{dup_id}-1", "{dup_id}-2").',
                score_impact=2.0,
            ))
        return issues

    def _check_contrast(self, computed_styles: list, _id) -> List[AccessibilityIssue]:
        """WCAG 1.4.3: Check color contrast ratios from computed styles.
        
        Uses the relative luminance formula from WCAG 2.1:
        L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        Contrast ratio = (L1 + 0.05) / (L2 + 0.05)
        """
        issues = []
        checked = 0
        for style in computed_styles:
            if checked >= 20:  # Limit to prevent flooding
                break
            fg = self._parse_color(style.get("color", ""))
            bg = self._parse_color(style.get("backgroundColor", ""))
            if fg and bg:
                ratio = self._contrast_ratio(fg, bg)
                font_size = self._parse_font_size(style.get("fontSize", "16px"))
                is_large = font_size >= 18 or (font_size >= 14 and style.get("fontWeight", "400") in ("bold", "700", "800", "900"))

                min_ratio = CONTRAST_RATIOS["AA_large"] if is_large else CONTRAST_RATIOS["AA_normal"]
                if ratio < min_ratio:
                    checked += 1
                    text_preview = style.get("text", "")[:40]
                    issues.append(AccessibilityIssue(
                        id=_id("contrast"),
                        wcag_criterion="1.4.3",
                        wcag_level=WCAGLevel.AA,
                        severity=SeverityLevel.CRITICAL,
                        title=f"Insufficient Color Contrast ({ratio:.1f}:1)",
                        description=(
                            f'Text "{text_preview}" has a contrast ratio of {ratio:.2f}:1 '
                            f'(minimum required: {min_ratio}:1 for {"large" if is_large else "normal"} text).'
                        ),
                        element=f'<{style.get("tag", "span")}>{text_preview}</...>',
                        suggestion=f"Increase contrast to at least {min_ratio}:1. Darken the text or lighten the background.",
                        score_impact=4.0,
                    ))
        return issues

    def _check_viewport(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 1.4.4: Do not disable pinch-to-zoom."""
        issues = []
        if dom.has_viewport_meta and not dom.viewport_scalable:
            issues.append(AccessibilityIssue(
                id=_id("viewport"),
                wcag_criterion="1.4.4",
                wcag_level=WCAGLevel.AA,
                severity=SeverityLevel.CRITICAL,
                title="Zoom/Scaling Disabled",
                description="The viewport meta tag prevents users from zooming. This blocks users with low vision from enlarging text.",
                element='<meta name="viewport" content="...user-scalable=no...">',
                suggestion='Remove user-scalable=no and maximum-scale=1 from the viewport meta tag.',
                score_impact=5.0,
            ))
        return issues

    def _check_tables(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """Check tables for proper header structure."""
        issues = []
        for table in dom.tables:
            if table["rows"] > 1 and not table["has_th"]:
                issues.append(AccessibilityIssue(
                    id=_id("table"),
                    wcag_criterion="1.3.1",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.WARNING,
                    title="Table Missing Headers",
                    description="A data table has no <th> header cells. Screen readers cannot associate data cells with their headers.",
                    element="<table> (without <th>)",
                    suggestion="Add <th> elements in the first row or column to identify table headers.",
                    score_impact=3.0,
                ))
        return issues

    def _check_iframes(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """Check iframes for title attributes."""
        issues = []
        for iframe in dom.iframes:
            if not iframe.get("has_title") and not iframe.get("aria_label"):
                issues.append(AccessibilityIssue(
                    id=_id("iframe"),
                    wcag_criterion="4.1.2",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.WARNING,
                    title="Iframe Missing Title",
                    description="An <iframe> has no title attribute. Screen readers use the title to describe embedded content.",
                    element=f'<iframe src="{iframe.get("src", "")[:50]}">',
                    data_al_id=iframe.get("data_al_id"),
                    suggestion='Add a descriptive title attribute: <iframe title="Description of embedded content">.',
                    score_impact=2.0,
                ))
        return issues

    def _check_buttons(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """WCAG 4.1.2: Buttons must have accessible names."""
        issues = []
        for btn in dom.buttons:
            if not btn.get("has_accessible_name"):
                issues.append(AccessibilityIssue(
                    id=_id("button"),
                    wcag_criterion="4.1.2",
                    wcag_level=WCAGLevel.A,
                    severity=SeverityLevel.CRITICAL,
                    title="Button Missing Accessible Name",
                    description="A button has no text, value, or aria-label. Screen readers cannot determine the button's purpose.",
                    element=f'<{btn["tag"]} type="{btn.get("type", "")}">',
                    data_al_id=btn.get("data_al_id"),
                    suggestion="Add text content, a value attribute, or aria-label to the button.",
                    score_impact=4.0,
                ))
        return issues

    def _check_focus_outlines(self, dom: DOMData, _id) -> List[AccessibilityIssue]:
        """Check if focus outlines are removed via CSS."""
        issues = []
        if dom.focus_outline_removed:
            issues.append(AccessibilityIssue(
                id=_id("focus"),
                wcag_criterion="2.1.1",
                wcag_level=WCAGLevel.A,
                severity=SeverityLevel.CRITICAL,
                title="Focus Outline Removed",
                description="CSS removes the focus outline (:focus { outline: none }). Keyboard users cannot see which element is focused.",
                element=":focus { outline: none }",
                suggestion="Replace 'outline: none' with a visible custom focus style (e.g., 'outline: 2px solid #4A90D9').",
                score_impact=5.0,
            ))
        return issues

    # ================================================================
    # Color Contrast Utility Methods
    # ================================================================

    def _parse_color(self, color_str: str) -> Optional[Tuple[int, int, int]]:
        """Parse CSS color value to RGB tuple.
        
        Supports formats: rgb(r,g,b), rgba(r,g,b,a), #hex
        """
        if not color_str:
            return None
        
        # rgb/rgba format
        match = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color_str)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # hex format
        match = re.match(r"#([0-9a-fA-F]{6})", color_str)
        if match:
            hex_val = match.group(1)
            return (int(hex_val[0:2], 16), int(hex_val[2:4], 16), int(hex_val[4:6], 16))
        
        # Short hex
        match = re.match(r"#([0-9a-fA-F]{3})", color_str)
        if match:
            hex_val = match.group(1)
            return (int(hex_val[0]*2, 16), int(hex_val[1]*2, 16), int(hex_val[2]*2, 16))
        
        return None

    def _relative_luminance(self, rgb: Tuple[int, int, int]) -> float:
        """Calculate relative luminance per WCAG 2.1 formula.
        
        Formula: L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        Where R, G, B are linearized sRGB values.
        """
        def linearize(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        
        r, g, b = rgb
        return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)

    def _contrast_ratio(self, fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> float:
        """Calculate WCAG contrast ratio between two colors.
        
        Ratio = (L_lighter + 0.05) / (L_darker + 0.05)
        Result is between 1:1 (no contrast) and 21:1 (black on white).
        """
        l1 = self._relative_luminance(fg)
        l2 = self._relative_luminance(bg)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def _parse_font_size(self, size_str: str) -> float:
        """Parse CSS font-size to points."""
        match = re.match(r"([\d.]+)px", size_str)
        if match:
            return float(match.group(1)) * 0.75  # px to pt
        return 12.0  # default

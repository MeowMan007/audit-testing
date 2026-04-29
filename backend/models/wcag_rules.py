"""
WCAG 2.1 Criteria Definitions and Reference Data.

This module contains the WCAG success criteria used by the rule engine,
along with reference data for ARIA roles, generic link texts, and
landmark element mappings. These definitions drive the rule-based
accessibility checks in rule_engine.py.

Reference: https://www.w3.org/TR/WCAG21/
"""

# ============================================================
# WCAG 2.1 Success Criteria
# Each criterion has: title, conformance level, description
# ============================================================
WCAG_CRITERIA = {
    "1.1.1": {
        "title": "Non-text Content",
        "level": "A",
        "description": "All non-text content has a text alternative that serves the equivalent purpose.",
    },
    "1.3.1": {
        "title": "Info and Relationships",
        "level": "A",
        "description": "Information, structure, and relationships can be programmatically determined.",
    },
    "1.4.1": {
        "title": "Use of Color",
        "level": "A",
        "description": "Color is not used as the only visual means of conveying information.",
    },
    "1.4.3": {
        "title": "Contrast (Minimum)",
        "level": "AA",
        "description": "Text has a contrast ratio of at least 4.5:1 (3:1 for large text).",
    },
    "1.4.4": {
        "title": "Resize Text",
        "level": "AA",
        "description": "Text can be resized up to 200% without loss of content or functionality.",
    },
    "1.4.6": {
        "title": "Contrast (Enhanced)",
        "level": "AAA",
        "description": "Text has a contrast ratio of at least 7:1 (4.5:1 for large text).",
    },
    "1.4.10": {
        "title": "Reflow",
        "level": "AA",
        "description": "Content can be presented without loss of information or functionality at 320px width.",
    },
    "2.1.1": {
        "title": "Keyboard",
        "level": "A",
        "description": "All functionality is operable through a keyboard interface.",
    },
    "2.4.1": {
        "title": "Bypass Blocks",
        "level": "A",
        "description": "A mechanism is available to bypass blocks of content that are repeated.",
    },
    "2.4.2": {
        "title": "Page Titled",
        "level": "A",
        "description": "Web pages have titles that describe topic or purpose.",
    },
    "2.4.4": {
        "title": "Link Purpose (In Context)",
        "level": "A",
        "description": "The purpose of each link can be determined from the link text alone.",
    },
    "2.4.6": {
        "title": "Headings and Labels",
        "level": "AA",
        "description": "Headings and labels describe topic or purpose.",
    },
    "2.5.5": {
        "title": "Target Size",
        "level": "AAA",
        "description": "The size of the target for pointer inputs is at least 44×44 CSS pixels.",
    },
    "3.1.1": {
        "title": "Language of Page",
        "level": "A",
        "description": "The default human language of each page can be programmatically determined.",
    },
    "3.3.2": {
        "title": "Labels or Instructions",
        "level": "A",
        "description": "Labels or instructions are provided when content requires user input.",
    },
    "4.1.1": {
        "title": "Parsing",
        "level": "A",
        "description": "Elements have complete start and end tags, no duplicate attributes, and unique IDs.",
    },
    "4.1.2": {
        "title": "Name, Role, Value",
        "level": "A",
        "description": "For all UI components, the name and role can be programmatically determined.",
    },
}

# ============================================================
# Valid ARIA Roles (subset of WAI-ARIA 1.2)
# Used by the rule engine to validate role attributes
# ============================================================
VALID_ARIA_ROLES = {
    # Landmark roles
    "banner", "complementary", "contentinfo", "form", "main",
    "navigation", "region", "search",
    # Widget roles
    "alert", "alertdialog", "button", "checkbox", "dialog",
    "gridcell", "link", "log", "marquee", "menuitem",
    "menuitemcheckbox", "menuitemradio", "option", "progressbar",
    "radio", "scrollbar", "searchbox", "slider", "spinbutton",
    "status", "switch", "tab", "tabpanel", "textbox",
    "timer", "tooltip", "treeitem",
    # Document structure roles
    "article", "cell", "columnheader", "definition", "directory",
    "document", "feed", "figure", "group", "heading", "img",
    "list", "listitem", "math", "none", "note", "presentation",
    "row", "rowgroup", "rowheader", "separator", "table",
    "term", "toolbar", "treegrid",
    # Abstract roles (should not be used, but recognized)
    "application", "grid", "listbox", "menu", "menubar",
    "radiogroup", "tablist", "tree",
}

# ============================================================
# Generic Link Texts (inaccessible patterns)
# Links with these texts fail WCAG 2.4.4
# ============================================================
GENERIC_LINK_TEXTS = {
    "click here", "here", "read more", "more", "learn more",
    "link", "click", "this", "go", "details", "continue",
    "download", "info", "click here for more",
    "click here to learn more", "see more", "view more",
}

# ============================================================
# HTML5 Landmark Element Mappings
# Maps HTML5 semantic elements to their implicit ARIA roles
# ============================================================
LANDMARK_ELEMENTS = {
    "header": "banner",
    "nav": "navigation",
    "main": "main",
    "aside": "complementary",
    "footer": "contentinfo",
    "form": "form",
    "section": "region",
}

# ============================================================
# Minimum Contrast Ratios per WCAG
# ============================================================
CONTRAST_RATIOS = {
    "AA_normal": 4.5,    # Normal text (< 18pt or < 14pt bold)
    "AA_large": 3.0,     # Large text (>= 18pt or >= 14pt bold)
    "AAA_normal": 7.0,   # Enhanced contrast for normal text
    "AAA_large": 4.5,    # Enhanced contrast for large text
}

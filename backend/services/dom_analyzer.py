"""
DOM Analyzer Service — Parses HTML with BeautifulSoup to extract accessibility data.

Uses BeautifulSoup (as required by the project spec) to parse the HTML DOM
and extract all elements relevant to WCAG accessibility checking:
- Heading hierarchy (h1-h6)
- Images with/without alt text
- Links with text content
- Form inputs and their labels
- ARIA attributes and roles
- Landmark regions
- Duplicate IDs
- Focus-related CSS patterns

This structured data is consumed by the RuleEngine for automated checks.
"""
import logging
import re
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


class DOMData:
    """Structured representation of accessibility-relevant DOM data.
    
    Attributes:
        title: Page <title> text
        lang: <html lang="..."> attribute
        headings: List of {level, text} for h1-h6 elements
        images: List of {src, alt, role, aria_label, ...}
        links: List of {href, text, aria_label, ...}
        forms: List of form elements with their inputs
        form_inputs: List of individual form controls
        aria_elements: Elements with ARIA attributes
        landmarks: Landmark regions (nav, main, aside, etc.)
        duplicate_ids: IDs that appear more than once
        has_skip_link: Whether a skip-to-content link exists
        has_viewport_meta: Whether viewport meta tag exists
        viewport_scalable: Whether user-scalable is enabled
        tables: Tables with/without proper headers
        iframes: Iframes with/without title attributes
        buttons: Buttons with/without accessible names
        focus_outline_removed: Whether CSS removes focus outlines
    """
    def __init__(self):
        self.title: str = ""
        self.lang: str = ""
        self.headings: List[Dict] = []
        self.images: List[Dict] = []
        self.links: List[Dict] = []
        self.forms: List[Dict] = []
        self.form_inputs: List[Dict] = []
        self.aria_elements: List[Dict] = []
        self.landmarks: List[Dict] = []
        self.duplicate_ids: List[str] = []
        self.has_skip_link: bool = False
        self.has_viewport_meta: bool = False
        self.viewport_scalable: bool = True
        self.tables: List[Dict] = []
        self.iframes: List[Dict] = []
        self.buttons: List[Dict] = []
        self.focus_outline_removed: bool = False
        self.total_elements: int = 0


class DOMAnalyzer:
    """Parses HTML using BeautifulSoup and extracts structured accessibility data.
    
    Usage:
        analyzer = DOMAnalyzer()
        dom_data = analyzer.analyze(html_string)
    """

    def analyze(self, html: str) -> DOMData:
        """Parse HTML and extract all accessibility-relevant data.
        
        Args:
            html: Raw HTML string to analyze
            
        Returns:
            DOMData containing structured accessibility information
        """
        data = DOMData()
        if not html:
            return data

        soup = BeautifulSoup(html, "lxml")
        data.total_elements = len(soup.find_all(True))

        self._extract_title(soup, data)
        self._extract_language(soup, data)
        self._extract_headings(soup, data)
        self._extract_images(soup, data)
        self._extract_links(soup, data)
        self._extract_forms(soup, data)
        self._extract_aria(soup, data)
        self._extract_landmarks(soup, data)
        self._find_duplicate_ids(soup, data)
        self._check_skip_link(soup, data)
        self._check_viewport(soup, data)
        self._extract_tables(soup, data)
        self._extract_iframes(soup, data)
        self._extract_buttons(soup, data)
        self._check_focus_styles(soup, data)

        logger.info(
            f"DOM analysis complete: {len(data.headings)} headings, "
            f"{len(data.images)} images, {len(data.links)} links, "
            f"{len(data.form_inputs)} form inputs"
        )
        return data

    def _extract_title(self, soup: BeautifulSoup, data: DOMData):
        """Extract <title> tag content."""
        title_tag = soup.find("title")
        data.title = title_tag.get_text(strip=True) if title_tag else ""

    def _extract_language(self, soup: BeautifulSoup, data: DOMData):
        """Extract lang attribute from <html> tag."""
        html_tag = soup.find("html")
        data.lang = html_tag.get("lang", "") if html_tag else ""

    def _extract_headings(self, soup: BeautifulSoup, data: DOMData):
        """Extract heading hierarchy (h1 through h6).
        
        Records the level, text content, and order of appearance
        for heading hierarchy validation.
        """
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                text = heading.get_text(strip=True)
                data.headings.append({
                    "level": level,
                    "text": text[:200],
                    "tag": f"h{level}",
                    "id": heading.get("id", ""),
                })

    def _extract_images(self, soup: BeautifulSoup, data: DOMData):
        """Extract all <img> elements with their accessibility attributes.
        
        Checks for: alt text, role, aria-label, aria-labelledby,
        and whether the image is decorative.
        """
        for img in soup.find_all("img"):
            alt = img.get("alt")
            data.images.append({
                "src": img.get("src", ""),
                "alt": alt,
                "has_alt": alt is not None,
                "alt_empty": alt == "" if alt is not None else False,
                "role": img.get("role", ""),
                "aria_label": img.get("aria-label", ""),
                "aria_labelledby": img.get("aria-labelledby", ""),
                "is_decorative": img.get("role") == "presentation" or alt == "",
                "width": img.get("width", ""),
                "height": img.get("height", ""),
            })

    def _extract_links(self, soup: BeautifulSoup, data: DOMData):
        """Extract all <a> elements with accessibility-relevant attributes."""
        for link in soup.find_all("a"):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            data.links.append({
                "href": href,
                "text": text[:200],
                "aria_label": link.get("aria-label", ""),
                "title": link.get("title", ""),
                "target": link.get("target", ""),
                "has_text": bool(text),
                "is_empty": not text and not link.get("aria-label"),
            })

    def _extract_forms(self, soup: BeautifulSoup, data: DOMData):
        """Extract form controls and check for proper labeling.
        
        Each input should have either:
        - An associated <label> (via 'for' attribute matching input 'id')
        - An aria-label attribute
        - An aria-labelledby attribute
        - A wrapping <label> element
        """
        for form in soup.find_all("form"):
            data.forms.append({
                "action": form.get("action", ""),
                "method": form.get("method", ""),
                "inputs": len(form.find_all(["input", "select", "textarea"])),
            })

        for inp in soup.find_all(["input", "select", "textarea"]):
            input_type = inp.get("type", "text") if inp.name == "input" else inp.name
            if input_type in ("hidden", "submit", "button", "reset", "image"):
                continue

            inp_id = inp.get("id", "")
            has_label = False

            # Check for associated <label for="...">
            if inp_id:
                label = soup.find("label", attrs={"for": inp_id})
                if label:
                    has_label = True

            # Check for wrapping <label>
            if not has_label:
                parent = inp.parent
                while parent:
                    if parent.name == "label":
                        has_label = True
                        break
                    parent = parent.parent

            # Check for ARIA labels
            has_aria = bool(inp.get("aria-label") or inp.get("aria-labelledby"))

            data.form_inputs.append({
                "tag": inp.name,
                "type": input_type,
                "id": inp_id,
                "name": inp.get("name", ""),
                "has_label": has_label,
                "has_aria_label": has_aria,
                "placeholder": inp.get("placeholder", ""),
                "required": inp.has_attr("required"),
            })

    def _extract_aria(self, soup: BeautifulSoup, data: DOMData):
        """Extract elements with ARIA attributes for validation."""
        for el in soup.find_all(True):
            aria_attrs = {
                k: v for k, v in el.attrs.items()
                if isinstance(k, str) and k.startswith("aria-")
            }
            role = el.get("role", "")
            if aria_attrs or role:
                data.aria_elements.append({
                    "tag": el.name,
                    "role": role,
                    "aria_attrs": aria_attrs,
                    "text": el.get_text(strip=True)[:100],
                })

    def _extract_landmarks(self, soup: BeautifulSoup, data: DOMData):
        """Extract HTML5 landmark regions and ARIA landmark roles."""
        landmark_tags = {"header", "nav", "main", "aside", "footer", "section", "article"}
        for tag_name in landmark_tags:
            for el in soup.find_all(tag_name):
                data.landmarks.append({
                    "tag": tag_name,
                    "role": el.get("role", ""),
                    "aria_label": el.get("aria-label", ""),
                    "id": el.get("id", ""),
                })

        # Also check for role="..." landmarks
        for el in soup.find_all(attrs={"role": True}):
            role = el.get("role", "")
            if role in ("banner", "navigation", "main", "complementary", "contentinfo", "search"):
                data.landmarks.append({
                    "tag": el.name,
                    "role": role,
                    "aria_label": el.get("aria-label", ""),
                    "id": el.get("id", ""),
                })

    def _find_duplicate_ids(self, soup: BeautifulSoup, data: DOMData):
        """Find duplicate ID attributes (violates WCAG 4.1.1)."""
        id_counts: Dict[str, int] = {}
        for el in soup.find_all(True, id=True):
            el_id = el.get("id", "")
            if el_id:
                id_counts[el_id] = id_counts.get(el_id, 0) + 1

        data.duplicate_ids = [id_ for id_, count in id_counts.items() if count > 1]

    def _check_skip_link(self, soup: BeautifulSoup, data: DOMData):
        """Check for skip-to-content/skip-navigation links (WCAG 2.4.1)."""
        skip_patterns = ["#main", "#content", "#skip", "#maincontent"]
        for link in soup.find_all("a", href=True):
            href = link.get("href", "").lower()
            text = link.get_text(strip=True).lower()
            if any(p in href for p in skip_patterns) or "skip" in text:
                data.has_skip_link = True
                break

    def _check_viewport(self, soup: BeautifulSoup, data: DOMData):
        """Check viewport meta tag for user-scalable restriction (WCAG 1.4.4)."""
        viewport = soup.find("meta", attrs={"name": "viewport"})
        if viewport:
            data.has_viewport_meta = True
            content = viewport.get("content", "").lower()
            if "user-scalable=no" in content or "maximum-scale=1" in content:
                data.viewport_scalable = False

    def _extract_tables(self, soup: BeautifulSoup, data: DOMData):
        """Extract tables and check for proper header cells."""
        for table in soup.find_all("table"):
            data.tables.append({
                "has_thead": table.find("thead") is not None,
                "has_th": table.find("th") is not None,
                "has_caption": table.find("caption") is not None,
                "rows": len(table.find_all("tr")),
                "summary": table.get("summary", ""),
            })

    def _extract_iframes(self, soup: BeautifulSoup, data: DOMData):
        """Extract iframes and check for title attributes."""
        for iframe in soup.find_all("iframe"):
            data.iframes.append({
                "src": iframe.get("src", ""),
                "title": iframe.get("title", ""),
                "has_title": bool(iframe.get("title")),
                "aria_label": iframe.get("aria-label", ""),
            })

    def _extract_buttons(self, soup: BeautifulSoup, data: DOMData):
        """Extract buttons and check for accessible names."""
        for btn in soup.find_all(["button", "input"]):
            if btn.name == "input" and btn.get("type") not in ("button", "submit", "reset"):
                continue
            text = btn.get_text(strip=True) if btn.name == "button" else ""
            data.buttons.append({
                "tag": btn.name,
                "type": btn.get("type", ""),
                "text": text,
                "value": btn.get("value", ""),
                "aria_label": btn.get("aria-label", ""),
                "has_accessible_name": bool(
                    text or btn.get("value") or btn.get("aria-label") or btn.get("aria-labelledby")
                ),
            })

    def _check_focus_styles(self, soup: BeautifulSoup, data: DOMData):
        """Check if CSS removes focus outlines (anti-pattern for keyboard users)."""
        for style in soup.find_all("style"):
            css_text = style.get_text()
            if re.search(r"outline\s*:\s*none|outline\s*:\s*0", css_text):
                if ":focus" in css_text:
                    data.focus_outline_removed = True
                    break

"""
Page Fetcher Service — Uses Selenium to render pages and capture screenshots.

This service is responsible for:
1. Navigating to the target URL using a headless Chrome browser
2. Capturing a full-page screenshot (returned as base64)
3. Extracting the rendered HTML DOM
4. Extracting computed CSS styles for contrast analysis
5. Falling back to httpx if Selenium is unavailable

Selenium is used (per project requirements) because it:
- Renders JavaScript-heavy pages (SPAs, dynamic content)
- Provides accurate computed styles
- Captures visual representation for CNN model input
"""
import asyncio
import base64
import logging
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger(__name__)


class PageData:
    """Container for all data extracted from a web page."""
    def __init__(self):
        self.url: str = ""
        self.html: str = ""
        self.title: str = ""
        self.screenshot_b64: Optional[str] = None
        self.computed_styles: list = []
        self.images: list = []
        self.links: list = []
        self.element_rects: Dict[str, Dict[str, float]] = {}
        self.success: bool = False
        self.error: Optional[str] = None


class PageFetcher:
    """Fetches web pages using Selenium for full rendering + screenshots.
    
    Falls back to httpx for basic HTML fetching if Selenium/Chrome
    is not available (e.g., local development without Chrome installed).
    """

    def __init__(self):
        self._driver = None

    def _init_selenium(self):
        """Initialize Selenium WebDriver with headless Chrome.
        
        Uses webdriver-manager for automatic ChromeDriver management,
        so the user doesn't need to manually download ChromeDriver.
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager

            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-extensions")
            options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            service = Service(ChromeDriverManager().install())
            self._driver = webdriver.Chrome(service=service, options=options)
            self._driver.set_page_load_timeout(30)
            logger.info("Selenium WebDriver initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Selenium init failed: {e}. Will use httpx fallback.")
            return False

    async def fetch(self, url: str) -> PageData:
        """Fetch a page, render it, and extract all accessibility-relevant data.
        
        Pipeline:
        1. Try Selenium for full JS rendering + screenshot
        2. Fall back to httpx for basic HTML if Selenium fails
        3. Extract images, links, and computed styles
        
        Args:
            url: The URL to fetch and analyze
            
        Returns:
            PageData with HTML, screenshot, styles, and element lists
        """
        page_data = PageData()
        page_data.url = url

        # Try Selenium first
        selenium_ok = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_with_selenium, url, page_data
        )

        if not selenium_ok:
            # Fallback to httpx
            await self._fetch_with_httpx(url, page_data)

        return page_data

    def _fetch_with_selenium(self, url: str, page_data: PageData) -> bool:
        """Render page with Selenium and extract data.
        
        This runs in a thread executor because Selenium is synchronous.
        Extracts:
        - Full rendered HTML (after JS execution)
        - Full-page screenshot as base64 PNG
        - Computed background/foreground colors for contrast checking
        - Image sources and alt attributes
        - Link hrefs and text content
        """
        try:
            if not self._driver:
                if not self._init_selenium():
                    return False

            self._driver.get(url)

            # Wait for page to settle
            import time
            time.sleep(2)

            # Inject unique IDs and extract bounding boxes for all visible elements
            try:
                page_data.element_rects = self._driver.execute_script("""
                    const rects = {};
                    let id = 0;
                    document.querySelectorAll('body *').forEach(el => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            const alId = 'al-' + (id++);
                            el.setAttribute('data-al-id', alId);
                            rects[alId] = {
                                x: rect.x + window.scrollX,
                                y: rect.y + window.scrollY,
                                w: rect.width,
                                h: rect.height
                            };
                        }
                    });
                    return rects;
                """)
            except Exception as e:
                logger.warning(f"Could not extract bounding boxes: {e}")
                page_data.element_rects = {}

            # Get rendered HTML (now contains data-al-id)
            page_data.html = self._driver.page_source
            page_data.title = self._driver.title

            # Take screenshot (base64 PNG)
            screenshot_bytes = self._driver.get_screenshot_as_png()
            page_data.screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

            # Extract computed styles for text elements (for contrast analysis)
            try:
                page_data.computed_styles = self._driver.execute_script("""
                    const elements = document.querySelectorAll('p, span, a, h1, h2, h3, h4, h5, h6, li, td, th, label, button, input');
                    const styles = [];
                    for (let i = 0; i < Math.min(elements.length, 100); i++) {
                        const el = elements[i];
                        const cs = window.getComputedStyle(el);
                        const text = el.textContent.trim().substring(0, 100);
                        if (text) {
                            styles.push({
                                tag: el.tagName.toLowerCase(),
                                text: text,
                                color: cs.color,
                                backgroundColor: cs.backgroundColor,
                                fontSize: cs.fontSize,
                                fontWeight: cs.fontWeight,
                            });
                        }
                    }
                    return styles;
                """)
            except Exception as e:
                logger.warning(f"Could not extract computed styles: {e}")
                page_data.computed_styles = []

            # Extract image data
            try:
                page_data.images = self._driver.execute_script("""
                    return Array.from(document.images).map(img => ({
                        src: img.src,
                        alt: img.getAttribute('alt'),
                        width: img.naturalWidth,
                        height: img.naturalHeight,
                        role: img.getAttribute('role'),
                        ariaLabel: img.getAttribute('aria-label'),
                    }));
                """)
            except Exception:
                page_data.images = []

            # Extract link data
            try:
                page_data.links = self._driver.execute_script("""
                    return Array.from(document.links).map(a => ({
                        href: a.href,
                        text: a.textContent.trim().substring(0, 200),
                        ariaLabel: a.getAttribute('aria-label'),
                        target: a.target,
                        title: a.title,
                    }));
                """)
            except Exception:
                page_data.links = []

            page_data.success = True
            logger.info(f"Successfully fetched {url} with Selenium")
            return True

        except Exception as e:
            logger.error(f"Selenium fetch failed for {url}: {e}")
            page_data.error = str(e)
            return False

    async def _fetch_with_httpx(self, url: str, page_data: PageData):
        """Fallback: Fetch raw HTML using httpx (no JS rendering, no screenshot).
        
        This is used when Selenium is unavailable. It provides basic HTML
        for rule-based checks but cannot:
        - Execute JavaScript
        - Capture screenshots
        - Extract computed styles
        """
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=20.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                page_data.html = response.text
                page_data.success = True
                logger.info(f"Fetched {url} with httpx fallback (no screenshot)")
        except Exception as e:
            page_data.error = f"Both Selenium and httpx failed: {e}"
            logger.error(page_data.error)

    def close(self):
        """Clean up Selenium WebDriver."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

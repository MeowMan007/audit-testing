"""
Synthetic Dataset Generator — Creates labeled accessibility violation screenshots.

PURPOSE (Requirement 3 — Dataset):
    This module generates a synthetic dataset of web page screenshots with
    deliberate WCAG violations injected. This is the "Custom Dataset" described
    in the project spec: 500 examples of good vs bad accessibility patterns.

HOW IT WORKS:
    1. Generates random HTML pages with normal (accessible) content
    2. Randomly injects 0-3 accessibility violations per page
    3. Renders each page using Selenium headless Chrome
    4. Saves a screenshot (PNG) and a label vector (JSON)

VIOLATION CATEGORIES (6 classes):
    0 - low_contrast:   Text with insufficient color contrast (< 4.5:1)
    1 - small_text:     Text smaller than 10px
    2 - missing_alt:    Images without alt attributes
    3 - small_targets:  Click targets smaller than 44x44px
    4 - bad_heading:    Skipped or missing heading hierarchy
    5 - poor_layout:    Cluttered, overlapping, or illegible layouts

OUTPUT FORMAT (JSON per sample):
    {
        "image": "dataset/images/img_0001.png",
        "labels": [1, 0, 1, 0, 0, 0],
        "label_names": ["low_contrast", "missing_alt"],
        "label": "bad_contrast"  // primary label for simple classification
    }

DATA SPLIT:
    - Training:   80% (400 samples)
    - Validation: 10% (50 samples)
    - Test:       10% (50 samples)
"""
import json
import os
import random
import logging
import hashlib
from typing import List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# Violation Class Definitions
# ============================================================
VIOLATION_CLASSES = [
    "low_contrast",    # 0 - Text contrast < 4.5:1
    "small_text",      # 1 - Font size < 10px
    "missing_alt",     # 2 - Images without alt text
    "small_targets",   # 3 - Buttons/links < 44x44px
    "bad_heading",     # 4 - Skipped heading levels
    "poor_layout",     # 5 - Cluttered/overlapping elements
]

NUM_CLASSES = len(VIOLATION_CLASSES)

# ============================================================
# Content Templates for Realistic HTML Generation
# ============================================================
SAMPLE_TITLES = [
    "Welcome to Our Store", "About Us - Company", "Contact Information",
    "Product Catalog", "News & Updates", "Services Overview",
    "FAQ - Help Center", "Blog - Latest Posts", "Team Members",
    "Privacy Policy", "Terms of Service", "Pricing Plans",
]

SAMPLE_PARAGRAPHS = [
    "We provide high-quality products and services to customers worldwide.",
    "Our team of experts is dedicated to delivering the best solutions.",
    "Contact us today to learn more about our offerings and pricing.",
    "Browse our extensive catalog of products in various categories.",
    "Stay updated with the latest news and announcements from our team.",
    "Our services are designed to meet the needs of modern businesses.",
    "Read our frequently asked questions for quick answers to common queries.",
    "Explore our blog for insights, tutorials, and industry analysis.",
]

SAMPLE_IMAGES = [
    ("Product photo showing electronics", "electronics.jpg"),
    ("Team photo of employees", "team.jpg"),
    ("Office building exterior", "office.jpg"),
    ("Customer testimonial portrait", "testimonial.jpg"),
    ("Infographic with data visualization", "infographic.png"),
    ("Logo of the company", "logo.png"),
    ("Hero banner for the homepage", "hero.jpg"),
    ("Navigation menu icon", "menu-icon.svg"),
]

# ============================================================
# Color Pairs: (text_color, bg_color, contrast_label)
# ============================================================
GOOD_CONTRAST_PAIRS = [
    ("#000000", "#ffffff", "good"),  # Black on white = 21:1
    ("#1a1a1a", "#f5f5f5", "good"),  # Near-black on near-white
    ("#ffffff", "#333333", "good"),  # White on dark gray
    ("#0000ff", "#ffffff", "good"),  # Blue on white
    ("#ffffff", "#006600", "good"),  # White on dark green
]

BAD_CONTRAST_PAIRS = [
    ("#999999", "#ffffff", "bad"),   # Light gray on white ~2.8:1
    ("#cccccc", "#ffffff", "bad"),   # Very light gray on white ~1.6:1
    ("#aaaaaa", "#dddddd", "bad"),   # Light gray on lighter gray ~1.4:1
    ("#ff9999", "#ffcccc", "bad"),   # Pink on lighter pink ~1.3:1
    ("#88cc88", "#ccffcc", "bad"),   # Light green on very light green
    ("#8888ff", "#ccccff", "bad"),   # Light blue on lighter blue
]


class DatasetGenerator:
    """Generates synthetic accessibility violation screenshots.
    
    Creates HTML pages with controlled violations, renders them
    via Selenium, and produces labeled image datasets for training
    the AccessibilityNet CNN model.
    
    Usage:
        gen = DatasetGenerator(output_dir="dataset", num_samples=500)
        gen.generate()
    """

    def __init__(self, output_dir: str = "dataset", num_samples: int = 500):
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.images_dir = self.output_dir / "images"
        self.metadata: List[Dict] = []

    def generate(self):
        """Generate the complete dataset.
        
        Pipeline:
        1. Create output directories
        2. Generate HTML pages with random violations
        3. Render screenshots with Selenium
        4. Save metadata JSON with labels
        5. Create train/val/test splits
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating {self.num_samples} samples to {self.output_dir}")

        # Try Selenium first
        use_selenium = self._check_selenium()
        
        for i in range(self.num_samples):
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{self.num_samples}")

            # Decide which violations to inject (0-3 per sample)
            num_violations = random.choices([0, 1, 2, 3], weights=[0.2, 0.35, 0.3, 0.15])[0]
            violations = random.sample(range(NUM_CLASSES), min(num_violations, NUM_CLASSES))

            # Generate HTML with violations
            html = self._generate_html(violations)

            # Create multi-hot label vector
            labels = [1 if j in violations else 0 for j in range(NUM_CLASSES)]
            label_names = [VIOLATION_CLASSES[j] for j in violations]
            
            # Primary label for simple classification
            primary_label = "good" if not violations else label_names[0]

            # Save screenshot
            img_filename = f"img_{i:04d}.png"
            img_path = self.images_dir / img_filename

            if use_selenium:
                self._render_with_selenium(html, str(img_path))
            else:
                self._render_with_pillow(html, str(img_path), violations)

            # Record metadata
            self.metadata.append({
                "image": f"images/{img_filename}",
                "labels": labels,
                "label_names": label_names,
                "label": primary_label,
                "num_violations": num_violations,
            })

        # Save metadata
        self._save_metadata()
        
        # Create train/val/test splits
        self._create_splits()

        logger.info(f"Dataset generation complete: {len(self.metadata)} samples")

    def _check_selenium(self) -> bool:
        """Check if Selenium + Chrome is available."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            driver = webdriver.Chrome(options=options)
            driver.quit()
            return True
        except Exception:
            logger.warning("Selenium not available — using Pillow for image generation")
            return False

    def _generate_html(self, violations: List[int]) -> str:
        """Generate an HTML page with specified violations injected.
        
        Args:
            violations: List of violation class indices to inject
            
        Returns:
            Complete HTML string
        """
        title = random.choice(SAMPLE_TITLES)
        
        # Default accessible styles
        text_color = "#1a1a1a"
        bg_color = "#ffffff"
        font_size = "16px"
        heading_structure = "proper"
        include_alts = True
        button_size = "44px"
        layout_style = "clean"

        # Inject violations
        if 0 in violations:  # low_contrast
            pair = random.choice(BAD_CONTRAST_PAIRS)
            text_color, bg_color = pair[0], pair[1]

        if 1 in violations:  # small_text
            font_size = random.choice(["8px", "9px", "7px", "6px"])

        if 2 in violations:  # missing_alt
            include_alts = False

        if 3 in violations:  # small_targets
            button_size = random.choice(["18px", "20px", "22px", "15px"])

        if 4 in violations:  # bad_heading
            heading_structure = "broken"

        if 5 in violations:  # poor_layout
            layout_style = "cluttered"

        # Build heading section
        if heading_structure == "proper":
            headings_html = f"<h1>{title}</h1>\n<h2>Our Services</h2>\n<h3>Web Development</h3>"
        else:
            # Skipped levels: h1 -> h4 (skipping h2, h3)
            headings_html = f"<h1>{title}</h1>\n<h4>Our Services</h4>\n<h6>Web Development</h6>"

        # Build image section
        img_data = random.sample(SAMPLE_IMAGES, min(3, len(SAMPLE_IMAGES)))
        images_html = ""
        for alt_text, filename in img_data:
            if include_alts:
                images_html += f'<img src="{filename}" alt="{alt_text}" width="200" height="150">\n'
            else:
                images_html += f'<img src="{filename}" width="200" height="150">\n'

        # Build paragraphs
        paragraphs = random.sample(SAMPLE_PARAGRAPHS, min(3, len(SAMPLE_PARAGRAPHS)))
        paragraphs_html = "\n".join(f"<p>{p}</p>" for p in paragraphs)

        # Build buttons
        buttons_html = f"""
        <button style="padding: 10px 20px; min-width: {button_size}; min-height: {button_size};">
            Subscribe
        </button>
        <a href="#contact" style="display: inline-block; padding: 8px 16px; min-width: {button_size}; min-height: {button_size};">
            Contact Us
        </a>
        """

        # Layout style
        if layout_style == "cluttered":
            layout_css = """
                .content { position: relative; }
                .content p { position: absolute; top: 10px; left: 10px; }
                .overlap { margin-top: -50px; margin-left: 100px; opacity: 0.7; }
            """
            extra_class = "overlap"
        else:
            layout_css = ".content { max-width: 800px; margin: 0 auto; padding: 20px; }"
            extra_class = ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            color: {text_color};
            background-color: {bg_color};
            font-size: {font_size};
            line-height: 1.5;
            margin: 0;
            padding: 20px;
        }}
        {layout_css}
        nav {{ padding: 10px; background: #f0f0f0; margin-bottom: 20px; }}
        nav a {{ margin-right: 15px; color: {text_color}; text-decoration: none; }}
        button, .btn {{ cursor: pointer; border: 1px solid #999; border-radius: 4px; }}
        .images {{ display: flex; gap: 10px; margin: 20px 0; flex-wrap: wrap; }}
        .images img {{ border: 1px solid #ddd; border-radius: 4px; }}
        footer {{ margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; font-size: 14px; }}
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/about">About</a>
        <a href="/services">Services</a>
        <a href="/contact">Contact</a>
    </nav>

    <div class="content {extra_class}">
        {headings_html}
        {paragraphs_html}
        
        <div class="images">
            {images_html}
        </div>

        <form>
            <label for="email">Email Address</label>
            <input type="email" id="email" name="email" placeholder="you@example.com">
            {buttons_html}
        </form>
    </div>

    <footer>
        <p>&copy; 2026 {title}. All rights reserved.</p>
    </footer>
</body>
</html>"""
        return html

    def _render_with_selenium(self, html: str, output_path: str):
        """Render HTML to a screenshot using Selenium headless Chrome."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            import tempfile

            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1280,720")

            driver = webdriver.Chrome(options=options)
            
            # Write HTML to temp file and open it
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w', encoding='utf-8') as f:
                f.write(html)
                temp_path = f.name
            
            driver.get(f"file://{temp_path}")
            driver.save_screenshot(output_path)
            driver.quit()
            
            os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Selenium render failed: {e}")
            self._render_with_pillow(html, output_path, [])

    def _render_with_pillow(self, html: str, output_path: str, violations: List[int]):
        """Fallback: Generate a synthetic screenshot using Pillow.
        
        Creates a representative image without needing a browser.
        Uses colors, font sizes, and layout patterns that reflect
        the violations being simulated.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            # Create a minimal placeholder
            img = Image.new("RGB", (1280, 720), (255, 255, 255))
            img.save(output_path)
            return

        # Deterministic seed from html content for reproducibility
        seed = int(hashlib.md5(html.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Extract colors from violations
        if 0 in violations:  # low_contrast
            pair = rng.choice(BAD_CONTRAST_PAIRS)
            text_color = pair[0]
            bg_color = pair[1]
        else:
            pair = rng.choice(GOOD_CONTRAST_PAIRS)
            text_color = pair[0]
            bg_color = pair[1]

        # Convert hex to RGB
        def hex_to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        bg_rgb = hex_to_rgb(bg_color)
        text_rgb = hex_to_rgb(text_color)

        img = Image.new("RGB", (1280, 720), bg_rgb)
        draw = ImageDraw.Draw(img)

        # Try to load a font
        try:
            font_size = 8 if 1 in violations else 16
            font = ImageFont.truetype("arial.ttf", font_size)
            title_font = ImageFont.truetype("arial.ttf", font_size * 2)
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = font

        # Draw nav bar
        draw.rectangle([0, 0, 1280, 50], fill=(240, 240, 240))
        draw.text((20, 15), "Home    About    Services    Contact", fill=(50, 50, 50), font=font)

        # Draw title
        title = rng.choice(SAMPLE_TITLES)
        draw.text((40, 70), title, fill=text_rgb, font=title_font)

        # Draw paragraphs
        y = 130
        for para in rng.sample(SAMPLE_PARAGRAPHS, 3):
            draw.text((40, y), para, fill=text_rgb, font=font)
            y += 30

        # Draw image placeholders
        y += 20
        for i in range(3):
            x = 40 + i * 220
            draw.rectangle([x, y, x + 200, y + 150], outline=(200, 200, 200), width=2)
            if 2 not in violations:  # has alt text
                draw.text((x + 10, y + 65), "[Image]", fill=(150, 150, 150), font=font)
            else:
                draw.text((x + 10, y + 65), "[No alt]", fill=(255, 100, 100), font=font)

        # Draw buttons
        y += 180
        btn_size = 18 if 3 in violations else 44
        draw.rectangle([40, y, 40 + max(btn_size, 120), y + btn_size], 
                       outline=(100, 100, 100), fill=(230, 230, 250))
        draw.text((50, y + 5), "Subscribe", fill=text_rgb, font=font)

        # Draw cluttered overlap if poor_layout
        if 5 in violations:
            for _ in range(5):
                ox = rng.randint(0, 1200)
                oy = rng.randint(0, 650)
                draw.rectangle([ox, oy, ox + 200, oy + 60], fill=bg_rgb, outline=text_rgb)
                draw.text((ox + 10, oy + 20), rng.choice(SAMPLE_PARAGRAPHS)[:30], fill=text_rgb, font=font)

        img.save(output_path)

    def _save_metadata(self):
        """Save dataset metadata to JSON file."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "num_samples": len(self.metadata),
                "num_classes": NUM_CLASSES,
                "class_names": VIOLATION_CLASSES,
                "samples": self.metadata,
            }, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    def _create_splits(self):
        """Split dataset into train/val/test sets (80/10/10).
        
        Saves split indices to separate JSON files for reproducibility.
        """
        indices = list(range(len(self.metadata)))
        random.shuffle(indices)

        n = len(indices)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        splits = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        splits_path = self.output_dir / "splits.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=2)

        logger.info(
            f"Splits created — Train: {len(splits['train'])}, "
            f"Val: {len(splits['val'])}, Test: {len(splits['test'])}"
        )


# ============================================================
# CLI Entry Point
# ============================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Generate accessibility violation dataset")
    parser.add_argument("--output", default="dataset", help="Output directory")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples")
    args = parser.parse_args()

    gen = DatasetGenerator(output_dir=args.output, num_samples=args.samples)
    gen.generate()

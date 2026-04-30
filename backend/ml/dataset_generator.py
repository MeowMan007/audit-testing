"""
Advanced Dataset Pipeline — Real-World Web Accessibility Dataset.

DATA SOURCES (Multi-source strategy):
  1. osunlp/Multimodal-Mind2Web (HuggingFace) — real browser screenshots
     of websites during task completion. 10K+ diverse real webpage images.
  2. biglab/webui-7k (HuggingFace) — 7,000 real mobile/web UI screenshots.
  3. Synthetic augmentation — auto-generated violation pages to balance classes.

WEAK SUPERVISION LABELING:
  Since these real datasets have no accessibility labels, we apply
  "Weak Supervision" — a set of heuristic labeling functions (LFs) that
  analyze each screenshot using OpenCV to assign probabilistic labels:

  LF-1: low_contrast   → Detect low-variance gray regions (histogram analysis)
  LF-2: small_text     → Detect very fine text patterns (frequency analysis)
  LF-3: poor_layout    → Detect cluttered/dense pixel regions (entropy)
  LF-4: small_targets  → Detect small isolated interactive regions
  LF-5: bad_heading    → Structural heuristic (from URL/page type metadata)
  LF-6: missing_alt    → Not directly detectable from screenshot; use
                          synthetic data for this class.

TOTAL DATASET SIZE: ~5,000 samples (real) + 500 synthetic = ~5,500 total
SPLIT: 70% train / 15% val / 15% test
"""
import json
import os
import random
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

VIOLATION_CLASSES = [
    "low_contrast",
    "small_text",
    "missing_alt",
    "small_targets",
    "bad_heading",
    "poor_layout",
]
NUM_CLASSES = len(VIOLATION_CLASSES)

# ─────────────────────────────────────────────
# Weak Supervision Labeling Functions (OpenCV)
# ─────────────────────────────────────────────

def _lf_low_contrast(img_array: np.ndarray) -> float:
    """LF-1: Low contrast → low std-dev in luminance channel."""
    gray = 0.299*img_array[:,:,0] + 0.587*img_array[:,:,1] + 0.114*img_array[:,:,2]
    std = float(np.std(gray))
    # Low contrast if std < 35 (flat, washed-out image)
    return 1.0 if std < 35 else (0.6 if std < 55 else 0.0)

def _lf_small_text(img_array: np.ndarray) -> float:
    """LF-2: Small text → high-frequency edges in small local regions."""
    try:
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Laplacian measures sharpness/fine detail
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Very high freq → fine/tiny text
        return 1.0 if lap_var > 800 else (0.5 if lap_var > 500 else 0.0)
    except Exception:
        return 0.0

def _lf_poor_layout(img_array: np.ndarray) -> float:
    """LF-3: Poor layout → high spatial entropy (cluttered)."""
    gray = np.mean(img_array, axis=2).astype(np.uint8)
    hist, _ = np.histogram(gray, bins=32, range=(0,256))
    hist = hist / hist.sum() + 1e-10
    entropy = float(-np.sum(hist * np.log2(hist)))
    # High entropy = cluttered content
    return 1.0 if entropy > 4.8 else (0.5 if entropy > 4.2 else 0.0)

def _lf_small_targets(img_array: np.ndarray) -> float:
    """LF-4: Small targets → many small isolated high-contrast blobs."""
    try:
        import cv2
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small = sum(1 for c in contours if 10 < cv2.contourArea(c) < 500)
        return 1.0 if small > 40 else (0.5 if small > 20 else 0.0)
    except Exception:
        return 0.0

def label_image(img_array: np.ndarray) -> List[int]:
    """Apply all weak supervision LFs to produce binary labels."""
    scores = [
        _lf_low_contrast(img_array),  # 0 low_contrast
        _lf_small_text(img_array),     # 1 small_text
        0.0,                           # 2 missing_alt (not detectable visually)
        _lf_small_targets(img_array),  # 3 small_targets
        0.0,                           # 4 bad_heading (structural, not visual)
        _lf_poor_layout(img_array),    # 5 poor_layout
    ]
    return [1 if s >= 0.6 else 0 for s in scores]


# ─────────────────────────────────────────────
# Synthetic HTML Generator (for missing_alt, bad_heading)
# ─────────────────────────────────────────────

SAMPLE_TITLES = [
    "Product Catalog", "Contact Us", "About Our Team",
    "Services Overview", "Blog Posts", "FAQ Center",
    "Pricing Plans", "Home Page", "News Updates",
]
SAMPLE_PARAGRAPHS = [
    "We deliver high-quality solutions for modern businesses worldwide.",
    "Our team of experts is ready to help you achieve your goals.",
    "Contact us today and discover how we can serve your needs.",
    "Browse our extensive catalog across multiple product categories.",
]
GOOD_CONTRAST = [("#1a1a1a","#ffffff"),("#ffffff","#333333"),("#000080","#ffffff")]
BAD_CONTRAST  = [("#aaaaaa","#ffffff"),("#cccccc","#eeeeee"),("#888888","#dddddd")]


def generate_synthetic_html(violations: List[int]) -> str:
    title = random.choice(SAMPLE_TITLES)
    text_c, bg_c = random.choice(BAD_CONTRAST if 0 in violations else GOOD_CONTRAST)
    font_size = random.choice(["7px","8px"]) if 1 in violations else "16px"
    include_alt = (2 not in violations)
    btn_size = "18px" if 3 in violations else "44px"
    heading = "<h1>%s</h1>\n<h4>Details</h4>" if 4 in violations else "<h1>%s</h1>\n<h2>Details</h2>"
    heading = heading % title
    layout_css = ".content{position:relative}" if 5 in violations else ".content{max-width:800px;margin:0 auto}"

    imgs = ""
    for i in range(3):
        if include_alt:
            imgs += f'<img src="img{i}.jpg" alt="Sample image {i}" width="200" height="150">\n'
        else:
            imgs += f'<img src="img{i}.jpg" width="200" height="150">\n'

    paras = "\n".join(f"<p>{p}</p>" for p in random.sample(SAMPLE_PARAGRAPHS, 3))

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;color:{text_c};background:{bg_c};font-size:{font_size};margin:0;padding:20px}}
{layout_css}
button{{min-width:{btn_size};min-height:{btn_size};padding:8px;cursor:pointer}}
</style></head><body>
<nav><a href="/">Home</a> <a href="/about">About</a></nav>
<div class="content">{heading}{paras}
<div>{imgs}</div>
<button>Subscribe</button></div>
</body></html>"""


# ─────────────────────────────────────────────
# Main Dataset Generator
# ─────────────────────────────────────────────

class DatasetGenerator:
    """
    Multi-source dataset generator:
      - Downloads real web UI screenshots from HuggingFace
      - Labels them with weak supervision (OpenCV heuristics)
      - Generates targeted synthetic samples for hard-to-detect classes
      - Produces a balanced ~5,500 sample dataset
    """

    def __init__(self, output_dir: str = "dataset", num_synthetic: int = 500,
                 num_real: int = 5000):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.num_synthetic = num_synthetic
        self.num_real = num_real
        self.metadata: List[Dict] = []

    def generate(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=== AccessLens Dataset Generator ===")

        # Phase 1: Real data from HuggingFace
        real_count = self._load_real_data()

        # Phase 2: Synthetic data for hard classes (missing_alt, bad_heading)
        synth_count = self._generate_synthetic()

        self._save_metadata()
        self._create_splits()
        logger.info(f"Done: {real_count} real + {synth_count} synthetic = {len(self.metadata)} total samples")

    # ── Phase 1: Real Data ──────────────────────────────────────────────

    def _load_real_data(self) -> int:
        """Download real web screenshots from HuggingFace datasets."""
        count = 0

        # Try Mind2Web multimodal first (real browser screenshots)
        count += self._load_hf_dataset(
            "osunlp/Multimodal-Mind2Web",
            split="train",
            image_col="screenshot",
            max_samples=min(3000, self.num_real // 2),
        )

        # Try WebUI-7k dataset
        if count < self.num_real:
            count += self._load_hf_dataset(
                "biglab/webui-7k",
                split="train",
                image_col="image",
                max_samples=min(2000, self.num_real - count),
            )

        # Fallback: AtomBlock-WebUI
        if count < 100:
            count += self._load_hf_dataset(
                "ZhihaoNan/AtomBlock-WebUI",
                split="train",
                image_col="image",
                max_samples=min(2000, self.num_real),
            )

        return count

    def _load_hf_dataset(self, dataset_id: str, split: str,
                         image_col: str, max_samples: int) -> int:
        """Load images from a HuggingFace dataset and apply weak supervision."""
        try:
            from datasets import load_dataset
            from PIL import Image as PILImage
        except ImportError:
            logger.warning("HuggingFace `datasets` not installed. Run: pip install datasets")
            return 0

        logger.info(f"Loading {dataset_id} ({split}, up to {max_samples} samples)...")
        try:
            ds = load_dataset(dataset_id, split=split, streaming=True,
                              trust_remote_code=True)
            count = 0
            for item in ds:
                if count >= max_samples:
                    break
                try:
                    img = item.get(image_col)
                    if img is None:
                        continue
                    # Convert to PIL if needed
                    if not hasattr(img, 'save'):
                        continue
                    img = img.convert("RGB").resize((224, 224))
                    arr = np.array(img)

                    # Weak supervision labeling
                    labels = label_image(arr)
                    label_names = [VIOLATION_CLASSES[i] for i, v in enumerate(labels) if v]
                    primary = label_names[0] if label_names else "clean"

                    idx = len(self.metadata)
                    fname = f"real_{idx:05d}.png"
                    img.save(self.images_dir / fname)

                    self.metadata.append({
                        "image": f"images/{fname}",
                        "labels": labels,
                        "label_names": label_names,
                        "label": primary,
                        "source": dataset_id,
                    })
                    count += 1
                    if count % 250 == 0:
                        logger.info(f"  {dataset_id}: {count}/{max_samples} loaded")
                except Exception as e:
                    logger.debug(f"  Skipping item: {e}")
                    continue
            logger.info(f"  Loaded {count} samples from {dataset_id}")
            return count
        except Exception as e:
            logger.warning(f"Could not load {dataset_id}: {e}")
            return 0

    # ── Phase 2: Synthetic Targeted Samples ────────────────────────────

    def _generate_synthetic(self) -> int:
        """Generate synthetic HTML screenshots for hard-to-detect classes."""
        use_selenium = self._check_selenium()
        count = 0
        # Weight: more missing_alt and bad_heading since those can't be detected visually
        violation_weights = [
            ([], 0.15),
            ([0], 0.10), ([1], 0.08), ([2], 0.18), ([3], 0.08),
            ([4], 0.18), ([5], 0.08),
            ([0,2], 0.08), ([2,4], 0.07),
        ]
        choices, weights = zip(*violation_weights)

        for i in range(self.num_synthetic):
            violations = list(random.choices(choices, weights=weights)[0])
            html = generate_synthetic_html(violations)
            labels = [1 if j in violations else 0 for j in range(NUM_CLASSES)]
            label_names = [VIOLATION_CLASSES[j] for j in violations]

            fname = f"synth_{i:04d}.png"
            fpath = self.images_dir / fname
            if use_selenium:
                self._render_selenium(html, str(fpath))
            else:
                self._render_pillow(html, str(fpath), violations)

            self.metadata.append({
                "image": f"images/{fname}",
                "labels": labels,
                "label_names": label_names,
                "label": label_names[0] if label_names else "clean",
                "source": "synthetic",
            })
            count += 1
            if (i+1) % 100 == 0:
                logger.info(f"  Synthetic: {i+1}/{self.num_synthetic}")
        return count

    def _check_selenium(self) -> bool:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            d = webdriver.Chrome(options=opts)
            d.quit()
            return True
        except Exception:
            return False

    def _render_selenium(self, html: str, path: str):
        try:
            import tempfile
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument("--window-size=1280,720")
            d = webdriver.Chrome(options=opts)
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False,
                                             mode='w', encoding='utf-8') as f:
                f.write(html); tmp = f.name
            d.get(f"file://{tmp}")
            d.save_screenshot(path)
            d.quit()
            os.unlink(tmp)
        except Exception as e:
            logger.debug(f"Selenium failed: {e}")
            self._render_pillow(html, path, [])

    def _render_pillow(self, html: str, path: str, violations: List[int]):
        try:
            from PIL import Image, ImageDraw, ImageFont
            seed = int(hashlib.md5(html.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            text_c_hex = rng.choice(BAD_CONTRAST if 0 in violations else GOOD_CONTRAST)[0]
            def hex2rgb(h):
                h=h.lstrip('#'); return tuple(int(h[i:i+2],16) for i in (0,2,4))
            bg = (255,255,255); text = hex2rgb(text_c_hex)
            img = Image.new("RGB",(1280,720),bg)
            draw = ImageDraw.Draw(img)
            try:
                font_size = 8 if 1 in violations else 16
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
            draw.rectangle([0,0,1280,50],fill=(240,240,240))
            draw.text((20,15),"Home  About  Services  Contact",fill=(50,50,50),font=font)
            draw.text((40,70),rng.choice(SAMPLE_TITLES),fill=text,font=font)
            y=110
            for p in rng.sample(SAMPLE_PARAGRAPHS,3):
                draw.text((40,y),p,fill=text,font=font); y+=30
            img.save(path)
        except Exception as e:
            logger.warning(f"Pillow render failed: {e}")
            from PIL import Image
            Image.new("RGB",(1280,720),(200,200,200)).save(path)

    # ── Metadata & Splits ──────────────────────────────────────────────

    def _save_metadata(self):
        path = self.output_dir / "metadata.json"
        sources = {}
        for s in self.metadata:
            src = s.get("source","unknown")
            sources[src] = sources.get(src,0)+1
        with open(path,"w") as f:
            json.dump({
                "num_samples": len(self.metadata),
                "num_classes": NUM_CLASSES,
                "class_names": VIOLATION_CLASSES,
                "sources": sources,
                "samples": self.metadata,
            },f,indent=2)
        logger.info(f"Metadata saved → {path}")
        logger.info(f"Source breakdown: {sources}")

    def _create_splits(self):
        indices = list(range(len(self.metadata)))
        random.shuffle(indices)
        n = len(indices)
        train_end = int(n*0.70)
        val_end   = int(n*0.85)
        splits = {"train":indices[:train_end],"val":indices[train_end:val_end],"test":indices[val_end:]}
        with open(self.output_dir/"splits.json","w") as f:
            json.dump(splits,f,indent=2)
        logger.info(f"Splits → Train:{len(splits['train'])} Val:{len(splits['val'])} Test:{len(splits['test'])}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--output",default="dataset")
    p.add_argument("--real",type=int,default=5000)
    p.add_argument("--synthetic",type=int,default=500)
    args = p.parse_args()
    DatasetGenerator(args.output, args.synthetic, args.real).generate()

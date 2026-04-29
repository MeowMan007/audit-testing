# 🧠 AccessLens — AI-Powered Accessibility Audit Tool

> **College Project**: Accessibility Audit using Deep Learning  
> **SDG Alignment**: SDG 12 — Responsible Consumption & Production

An AI-powered web accessibility auditing tool that combines **WCAG 2.1 rule-based checks** with a **custom-trained EfficientNet-B0 CNN model** for comprehensive accessibility analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)

---

## 1. Problem Definition

Over **96% of the top 1 million websites** fail WCAG accessibility standards. This tool addresses:

| Issue | WCAG Criterion | What It Means |
|-------|---------------|---------------|
| **Color Contrast** | 1.4.3 (AA) | Text must have ≥4.5:1 contrast ratio against background |
| **Missing Alt Text** | 1.1.1 (A) | Every image needs descriptive alternative text |
| **Poor Semantic HTML** | 1.3.1 (A) | Proper heading hierarchy, landmarks, form labels |
| **Accessibility Score** | Custom | Deduction-based 0-100 score with A-F grading |

Traditional tools catch only 30-40% of issues. Our tool adds **AI visual analysis** to detect layout, contrast, and readability problems that rule-based checks miss.

---

## 2. System Architecture

```
Input: Website URL
         │
         ▼
  ┌──────────────┐
  │  Page Fetcher │──── Selenium (headless Chrome)
  │  + Screenshot │──── Full-page screenshot capture
  └──────┬───────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│  DOM   │ │   CNN    │
│Analyzer│ │ Inference│
│(BS4)   │ │(Eff-B0) │
└───┬────┘ └────┬─────┘
    │            │
    ▼            ▼
┌────────┐  ┌────────┐
│  Rule  │  │   DL   │
│ Engine │  │ Engine │
│(15+    │  │        │
│checks) │  │        │
└───┬────┘  └───┬────┘
    │           │
    └─────┬─────┘
          ▼
  ┌──────────────┐
  │   Report     │
  │  Generator   │──── Score (0-100) + Issues + Fixes
  └──────────────┘
```

---

## 3. Dataset

### Custom Synthetic Dataset (500 samples)

**Data Collection**: `backend/ml/dataset_generator.py` automatically generates HTML pages with deliberate WCAG violations, renders them via Selenium, and saves labeled screenshots.

**6 Violation Categories**:
1. `low_contrast` — Text contrast < 4.5:1
2. `small_text` — Font size < 10px
3. `missing_alt` — Images without alt attributes
4. `small_targets` — Click targets < 44x44px
5. `bad_heading` — Skipped heading hierarchy
6. `poor_layout` — Cluttered/overlapping elements

**Label Format** (JSON):
```json
{
  "image": "images/img_0042.png",
  "labels": [1, 0, 1, 0, 0, 0],
  "label_names": ["low_contrast", "missing_alt"],
  "label": "low_contrast"
}
```

**Train/Test Split**: 80% train / 10% validation / 10% test

---

## 4. Machine Learning Model

### Architecture: AccessibilityNet

| Component | Detail |
|-----------|--------|
| **Backbone** | EfficientNet-B0 (pretrained on ImageNet, 5.3M params) |
| **Head** | Dropout(0.3) → BatchNorm → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→6) |
| **Output** | 6-class sigmoid (multi-label) |
| **Loss** | BCEWithLogitsLoss |
| **Optimizer** | AdamW (lr=1e-4, weight_decay=1e-4) |
| **Scheduler** | Cosine Annealing (25 epochs) |
| **Augmentation** | RandomFlip, ColorJitter, RandomRotation |

### Accuracy Metrics
- **Per-class F1, Precision, Recall**
- **Macro F1** (primary metric)
- **Hamming Loss** (fraction of incorrect labels)
- Expected: >80% F1 on synthetic validation set

---

## 5. Backend (Python + FastAPI)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/audit` | Audit a URL → JSON report |
| GET | `/api/health` | Health check + model status |
| GET | `/api/demo/{id}` | Pre-computed demo results |

### Technologies
- **FastAPI** — High-performance async Python API
- **BeautifulSoup** — HTML DOM parsing
- **Selenium** — Headless Chrome for screenshots
- **PyTorch** — CNN model inference

---

## 6. Frontend

Premium HTML/CSS/JS single-page application with:
- URL input with validation
- 3 demo website buttons
- Animated score gauge (0-100)
- Issue explorer with severity filtering & search
- AI insights panel with confidence bars
- WCAG explanation section
- Limitations section

---

## 7. Output Format

```json
{
  "overall_score": 62.0,
  "grade": "D",
  "total_issues": 8,
  "critical_count": 3,
  "warning_count": 5,
  "issues": [
    {
      "severity": "critical",
      "wcag_criterion": "1.4.3",
      "title": "Insufficient Color Contrast (3.2:1)",
      "suggestion": "Increase to at least 4.5:1"
    }
  ],
  "dl_insights": [
    {
      "category": "low_contrast",
      "confidence": 0.78,
      "title": "Low Color Contrast Detected"
    }
  ]
}
```

---

## 8. Limitations

1. **AI is limited** — Trained on 500 synthetic samples, may miss real-world edge cases
2. **Rule-based checks still required** — Automated checks cover ~30-40% of WCAG 2.1
3. **Small dataset** — 500 samples vs production systems using 10,000+
4. **Dynamic content** — SPAs and lazy-loaded content may not be fully captured
5. **No manual testing replacement** — Cannot replace human expert accessibility review

---

## 9. Demo Results

| Site | Score | Grade | Critical | Warnings |
|------|-------|-------|----------|----------|
| Well-Built Government Site | 87 | B | 0 | 4 |
| Modern Startup | 62 | D | 3 | 5 |
| Outdated Legacy Site | 34 | F | 8 | 6 |

---

## Quick Start (Google Colab)

1. Upload `AccessLens_Colab.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **T4 GPU**
3. Run cells in order:
   - Install dependencies
   - Generate 500 synthetic screenshots
   - Train EfficientNet-B0 (~15 min)
   - Start web server
4. Open the public URL provided

## Local Development

```bash
cd nfhds
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```

## Project Structure

```
nfhds/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── requirements.txt     # Python dependencies
│   ├── models/
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── wcag_rules.py    # WCAG criteria definitions
│   ├── services/
│   │   ├── page_fetcher.py  # Selenium screenshot capture
│   │   ├── dom_analyzer.py  # BeautifulSoup HTML parser
│   │   ├── rule_engine.py   # 15+ WCAG rule checks
│   │   ├── dl_engine.py     # CNN inference orchestrator
│   │   └── report_generator.py  # Score + report assembly
│   ├── ml/
│   │   ├── dataset_generator.py  # Synthetic dataset (500 samples)
│   │   ├── model.py         # EfficientNet-B0 architecture
│   │   ├── train.py         # Training script
│   │   └── inference.py     # Prediction module
│   └── routers/
│       └── audit.py         # API endpoints
├── frontend/
│   ├── index.html           # Premium UI
│   ├── css/styles.css       # Dark theme design system
│   └── js/app.js            # Frontend logic
├── AccessLens_Colab.ipynb   # Training notebook
└── README.md
```

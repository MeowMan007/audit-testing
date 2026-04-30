# 🧠 AccessLens — AI-Powered Accessibility Audit Tool

> **College Project**: Accessibility Audit using Deep Learning  
> **SDG Alignment**: SDG 12 — Responsible Consumption & Production

An AI-powered web accessibility auditing tool that combines **WCAG 2.1 rule-based checks** with a **custom-trained Vision Transformer (ViT-B/16)** for comprehensive accessibility analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-yellow)

---

## 1. Problem Definition

Over **96% of the top 1 million websites** fail WCAG accessibility standards. Traditional tools catch only 30-40% of issues. Our tool adds **AI visual analysis** to detect layout, contrast, and readability problems that rule-based checks miss.

---

## 2. System Architecture

```text
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
│  DOM   │ │   ViT    │
│Analyzer│ │ Inference│
│(BS4)   │ │(ViT-B/16)│
└───┬────┘ └────┬─────┘
    │            │
    ▼            ▼
┌────────┐  ┌────────┐
│  Rule  │  │   DL   │
│ Engine │  │ Engine │
│(15+    │  │ + XAI  │
│checks) │  │(GradCAM)│
└───┬────┘  └───┬────┘
    │           │
    └─────┬─────┘
          ▼
  ┌──────────────┐
  │   Report     │──── PDF Export
  │  Generator   │──── SQLite Database
  └──────────────┘
```

---

## 3. Academic & Engineering Highlights

This project demonstrates rigorous software engineering and machine learning principles required for a high-level final year project:

1. **Transformer Architecture**: Upgraded from CNN (EfficientNet) to Vision Transformer (ViT-B/16) for better global context understanding.
2. **Explainability (XAI)**: Implemented Grad-CAM to visualize model decision-making via attention heatmaps.
3. **Advanced ML Pipeline**: Synthetic data generation, weak supervision labels, and two-phase fine-tuning strategies.
4. **Comprehensive Evaluation Suite**: Scripts to generate Confusion Matrices, ROC curves, and Precision-Recall metrics.
5. **Robust System Design**: Component-based backend (FastAPI), SQLite database for history, PDF report generation, and a vanilla JS frontend with interactive dashboards.

---

## 4. Dataset

### Multi-Source Real-World Dataset (~5,500 samples)
Combines real-world web UI screenshots with weak supervision labeling and targeted synthetic data to train the ViT model.

---

## 5. Machine Learning Model

### Architecture: AccessibilityViT
- **Backbone**: ViT-B/16 (Vision Transformer, ~86M params, pretrained on ImageNet-21k)
- **Head**: Custom multi-label classification head with GELU and Dropout.
- **Explainability**: Custom Attention Rollout using Grad-CAM on the last transformer encoder block to generate localized heatmaps.

---

## 6. Installation & Quick Start

### Prerequisites
- Python 3.9+
- Chrome/Chromium installed (for Selenium headless browser)

### Local Development
```bash
cd nfhds
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000
```

### Evaluation & Training
```bash
# Train the model
python -m backend.ml.train

# Evaluate the model and generate metrics
python -m backend.ml.evaluate
```

---

## 7. Project Structure

```text
nfhds/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Centralized settings
│   ├── models/              # Pydantic schemas
│   ├── services/            # Core business logic (DB, PDF, Rules, DL)
│   ├── ml/                  # Deep Learning pipeline
│   │   ├── model.py         # ViT-B/16 Architecture
│   │   ├── train.py         # Training logic
│   │   ├── evaluate.py      # Metrics generation
│   │   ├── explainability.py# Grad-CAM attention heatmaps
│   │   └── inference.py     # Prediction module
│   ├── routers/             # API endpoints
│   └── tests/               # Unit testing suite
├── frontend/                # Vanilla HTML/CSS/JS dashboard
│   ├── index.html           
│   ├── css/styles.css       
│   └── js/app.js            
└── README.md
```

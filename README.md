---
title: AccessLens
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

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

### Hugging Face Token (Required for AI Insights & Dataset)
1. Go to [Hugging Face](https://huggingface.co/) and create an account or log in.
2. Go to your **Settings** > **Access Tokens** (or visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
3. Click **New token**, give it a name (e.g., "accesslens"), and set the role to **Read**.
4. Copy the generated token.
5. Create a `.env` file in the root `nfhds/` directory and add the token:
   ```env
   HF_TOKEN=your_hugging_face_token_here
   ```

### Running on Google Colab (Recommended for GPU Training)
Since training the ViT-B/16 model requires significant GPU resources, running on Google Colab is highly recommended.

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Go to **Runtime** > **Change runtime type** and select **T4 GPU** as the hardware accelerator.
3. Clone the repository and install Python dependencies:
   ```python
   !git clone https://github.com/yourusername/nfhds.git # Replace with your repo URL
   %cd nfhds
   !pip install -r backend/requirements.txt
   ```
4. Install system dependencies for Selenium (Headless Chromium):
   ```python
   !apt-get update
   !apt-get install -y chromium-browser chromium-chromedriver
   ```
5. Set your Hugging Face Token as an environment variable:
   ```python
   import os
   os.environ["HF_TOKEN"] = "your_hugging_face_token_here"
   ```
6. Run the training script (uses the T4 GPU automatically):
   ```python
   !python -m backend.ml.train
   ```
7. (Optional) Start the FastAPI backend and expose it using `localtunnel` or `cloudflared` to access the dashboard from the Colab instance:
   ```python
   import subprocess
   import time
   
   # Start Uvicorn in the background
   subprocess.Popen(["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])
   time.sleep(3) # Wait for server to start
   
   # Expose using localtunnel
   !npm install -g localtunnel
   !npx localtunnel --port 8000
   ```
   Click the `localtunnel` URL generated in the output to access the AccessLens dashboard!

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

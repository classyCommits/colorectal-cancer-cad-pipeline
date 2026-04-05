# Stage 5 — Streamlit Dashboard

## Overview

The dashboard is the final deliverable of the pipeline — a clinical-grade
web interface where a pathologist (or researcher) can:

1. Upload an H&E histopathology patch
2. Get instant cancer detection (colon_aca vs colon_n)
3. See a Grad-CAM heatmap showing which tissue regions drove the prediction
4. Download the annotated image and JSON report

---

## Screenshots

> Run the app to see the dashboard in action.

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit torch torchvision timm pillow numpy matplotlib
```

### 2. Ensure model weights exist

```
stage3_classification/
    checkpoints/
        efficientnet_b0_colon.pth   ← required
```

Train using `notebooks/03_model_training.ipynb` or download from Google Drive.

### 3. Run the dashboard

From the **project root**:

```bash
streamlit run stage5_dashboard/app.py
```

Opens at `http://localhost:8501`

---

## Files

| File | Description |
|---|---|
| `app.py` | Main Streamlit application — UI, upload, results display |
| `gradcam.py` | Grad-CAM implementation for EfficientNet-B0 |
| `utils.py` | Model loading, preprocessing, prediction, validation |

---

## Features

### Cancer Detection
- Binary classification: `colon_aca` (cancer) vs `colon_n` (normal)
- Confidence score with visual probability bars
- Risk level assessment (High / Moderate / Low / Very Low)

### Grad-CAM Visualization
- Heatmap overlay showing model attention regions
- Adjustable opacity and colormap via sidebar
- Download annotated image

### Report Export
- JSON report with prediction, confidence, risk level
- Downloadable from dashboard

### Model Info Panel
- Displays loaded checkpoint epoch and validation accuracy
- Device information

---

## Design

Clinical dark theme — inspired by medical imaging software:
- **Font:** IBM Plex Mono + IBM Plex Sans
- **Color:** Dark navy base with red/green accent for cancer/normal
- **Layout:** Three-column results (original | Grad-CAM | prediction)

---

## Pipeline Position

```
Stage 1 — Stain Normalization     ✅
Stage 2 — Patch Extraction        ✅
Stage 3 — Cancer Detection        ✅ (99.93% accuracy)
Stage 4 — Cancer Grading          ✅ (code ready)
Stage 5 — Dashboard               ✅ YOU ARE HERE
```

---

## Disclaimer

> This system is for **research purposes only**.
> It is not validated for clinical use and must not be used
> to make medical decisions. Always consult a qualified pathologist.

# Stage 4 — Cancer Grading Model

## What is Cancer Grading?

Once cancer is **detected** (Stage 3), the next clinical question is:
**how severe is it?** This is called grading.

| Grade | Name | Meaning | Treatment Implication |
|---|---|---|---|
| Grade 1 | Well-differentiated | Cells look nearly normal | Less aggressive treatment |
| Grade 2 | Moderately-differentiated | Cells somewhat abnormal | Moderate treatment |
| Grade 3 | Poorly-differentiated | Cells very abnormal | Aggressive treatment |

Grading drives treatment decisions — surgery type, chemotherapy dosage,
radiation planning. Automating it reduces inter-pathologist variability
and speeds up diagnosis.

---

## Architecture

Same EfficientNet-B0 backbone as Stage 3, extended with a deeper
classification head for the harder 3-class grading task:

```
Input (224×224 H&E patch)
        ↓
EfficientNet-B0 backbone (pretrained ImageNet)
        ↓ features (1280-dim)
Linear(1280 → 256) → ReLU → Dropout(0.4)
        ↓
Linear(256 → 128) → ReLU → Dropout(0.2)
        ↓
Linear(128 → 3)
        ↓
Grade 1 | Grade 2 | Grade 3
```

Key metric: **Quadratic Weighted Kappa** — standard for ordinal grading tasks.
Penalizes Grade 1 vs Grade 3 errors more than Grade 1 vs Grade 2.

---

## Files

| File | Description |
|---|---|
| `model.py` | `CancerGradingModel` — EfficientNet-B0 with 3-class grading head |
| `dataset.py` | `GradingDataset` — loads graded patches, handles class imbalance |
| `train.py` | Full training loop with weighted loss, early stopping, kappa metric |
| `evaluate.py` | Evaluation with confusion matrix and confidence distribution plots |
| `predict.py` | Single image grade prediction with confidence scores |

---

## Dataset Status

> ⚠️ **Training deferred pending graded dataset access.**

This stage requires H&E patches with **pathologist-annotated cancer grades**.
The code is fully implemented and ready to train once a graded dataset
is available.

### Recommended Datasets

**TCGA-COAD (recommended):**
- Source: GDC Data Portal — https://portal.gdc.cancer.gov/
- Contains colorectal WSIs with clinical grade annotations
- Workflow: Download WSIs → Stage 2 patch extraction → organize by grade

**Kather et al. Graded CRC:**
- Source: https://zenodo.org/record/53169
- 5000 patches, 8 tissue classes (proxy for grading)

### Expected Folder Structure

```
data/graded/
    grade1/      ← well-differentiated patches
        patch_001.png
        patch_002.png
    grade2/      ← moderately-differentiated patches
        ...
    grade3/      ← poorly-differentiated patches
        ...
```

---

## Usage (when dataset available)

### Train
```bash
python train.py \
    --data    data/graded \
    --output  checkpoints \
    --epochs  20 \
    --batch   32 \
    --lr      1e-4
```

### Evaluate
```bash
python evaluate.py \
    --checkpoint checkpoints/grading_model_best.pth \
    --data       data/graded \
    --output     outputs/grading_evaluation.png
```

### Predict single image
```bash
python predict.py \
    --checkpoint checkpoints/grading_model_best.pth \
    --image      path/to/patch.png
```

---

## How This Connects to the Pipeline

```
Stage 1 — Stain Normalization     ✅ Complete
Stage 2 — Patch Extraction        ✅ Complete
Stage 3 — Cancer Detection        ✅ Complete (99.93% accuracy)
Stage 4 — Cancer Grading          ⏳ Code ready, awaiting graded dataset
Stage 5 — Dashboard               🔜 Next
```

In clinical deployment:
1. Stage 3 detects **whether** cancer is present
2. Stage 4 determines **how severe** the cancer is
3. Stage 5 presents both results to the pathologist

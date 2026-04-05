# Stage 2 — Patch Extraction

## What is Patch Extraction and Why Does it Matter?

Histopathology slides are scanned as **Whole Slide Images (WSI)** — gigapixel
images that can be 100,000 × 80,000 pixels or larger. No GPU can process an
image that large directly.

Patch extraction solves this by dividing the WSI into small fixed-size tiles
(typically 224×224 pixels) that CNNs can process. Only tiles containing actual
tissue are kept — blank glass regions are discarded.

```
Gigapixel WSI (100,000 × 80,000 px)
           ↓
   Sliding window (224×224, stride=224)
           ↓
   Background filtering (remove blank tiles)
           ↓
   ~50,000 tissue patches ready for training
```

---

## Two Operating Modes

### Mode 1 — WSI Mode (Production / Colab)
For real `.svs` / `.ndpi` whole slide image files.
Requires `openslide-python` and the OpenSlide C library.

### Mode 2 — Standard Image Mode (Local Development)
For regular JPEG/PNG images (like LC25000).
Treats each image as a high-resolution source and extracts sub-patches.
**No additional dependencies needed.**

---

## Files

| File | Description |
|---|---|
| `extract_patches.py` | Core `PatchExtractor` class — both WSI and standard image modes |
| `filter_background.py` | `BackgroundFilter` — detects and removes blank/background tiles |
| `visualize_patches.py` | Grid visualization and single-image extraction demo |
| `tests/test_patch_extraction.py` | 13 unit tests |

---

## Setup

```bash
# Required for both modes
pip install numpy pillow matplotlib tqdm

# Required ONLY for WSI mode
pip install openslide-python
# Also install OpenSlide C library: https://openslide.org/download/
```

---

## Usage

### 1. Extract patches from LC25000 (local development)

```bash
python extract_patches.py \
    --input  data/raw/colon_image_sets \
    --output data/patches \
    --size   224 \
    --stride 224 \
    --ext    .jpeg \
    --mode   image
```

### 2. Visualize extraction on a single image

```bash
python visualize_patches.py \
    --source data/raw/colon_image_sets/colon_aca/colonca1.jpeg \
    --output outputs/extraction_demo.png
```

### 3. Visualize extracted patch grid

```bash
python visualize_patches.py \
    --folder data/patches/colon_aca \
    --n      25 \
    --output outputs/patch_grid.png
```

### 4. Extract from WSI (Colab / production)

```python
from extract_patches import PatchExtractor

extractor = PatchExtractor(patch_size=224, stride=224)
patches   = extractor.extract_from_wsi("slide.svs", level=0)
```

### 5. Run tests

```bash
pytest tests/test_patch_extraction.py -v
```

---

## Background Filtering

The `BackgroundFilter` uses three criteria to identify tissue patches:

| Criterion | Threshold | Rationale |
|---|---|---|
| Mean brightness | < 210 | Blank glass is very bright (RGB ~240) |
| Tissue coverage | > 15% | At least 15% dark pixels required |
| Std deviation | > 10.0 | Very uniform patches = blank |

These thresholds work well for standard H&E slides. Adjust via constructor:

```python
filter = BackgroundFilter(
    brightness_threshold=200,  # stricter brightness
    min_tissue_ratio=0.20,     # require 20% tissue
)
```

---

## How This Connects to the Pipeline

```
Stage 1 (Stain Normalization)
    ↓ normalized patches
Stage 2 (Patch Extraction)     ← YOU ARE HERE
    ↓ 224×224 tissue patches
Stage 3 (Classification)
    ↓ model weights
Stage 4 (Grading)
    ↓ grading model
Stage 5 (Dashboard)
```

In production, Stage 2 runs on TCGA-COAD `.svs` files from the GDC Data Portal
to generate training patches. For this portfolio project, we demonstrate the
full pipeline logic using LC25000 colon images as source data.

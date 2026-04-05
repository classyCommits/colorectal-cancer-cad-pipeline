# Stage 1 — Stain Normalization

## What is H&E Stain Normalization and Why Does it Matter?

H&E (Hematoxylin and Eosin) staining is the standard preparation for histopathology slides.
However, slides from **different hospitals, scanners, or preparation batches** look visually
different — even for identical tissue types. This is called **stain variability**.

If you train a deep learning model on slides from one hospital and test on another, the model
often fails — not because the cancer looks different, but because the **colors** look different.

Stain normalization is the preprocessing step that **standardizes all patches to look like
they came from the same scanner**, making the model focus on tissue morphology rather than
color artifacts.

---

## Algorithm: Macenko Normalization

This stage implements the **Macenko method** (Macenko et al., ISBI 2009), which works as follows:

```
1. Convert RGB image → Optical Density (OD) space
        OD = -log(RGB / 255)

2. Remove background pixels (low OD = white/blank regions)

3. Apply SVD on the OD values
        → Find the 2D subspace spanned by the two stains (H and E)

4. Project pixels onto the stain plane
        → Find the two extreme angles (hematoxylin and eosin directions)

5. Solve for stain concentrations via least squares
        OD = C × StainMatrix

6. Normalize concentrations to match reference image range

7. Reconstruct normalized RGB image using reference stain matrix
        OD_normalized = C_normalized × StainMatrix_reference
        RGB_normalized = exp(-OD_normalized) × 255
```

---

## Files

| File | Description |
|---|---|
| `macenko.py` | Core `MacenkoNormalizer` class — fit on reference, transform any image |
| `normalize_dataset.py` | Batch normalize an entire dataset folder with multiprocessing |
| `visualize_normalization.py` | Before/after comparison grid + channel histogram plots |

---

## Setup

```bash
pip install numpy pillow matplotlib tqdm
```

---

## Usage

### 1. Normalize a single image in Python

```python
from PIL import Image
from macenko import MacenkoNormalizer

normalizer = MacenkoNormalizer()
normalizer.fit(Image.open("reference.png"))

normalized = normalizer.transform(Image.open("patch.png"))
normalized.save("patch_normalized.png")
```

---

### 2. Batch normalize an entire dataset

```bash
python normalize_dataset.py \
    --input  data/raw/ \
    --output data/normalized/ \
    --reference data/sample/reference.png \
    --workers 4
```

If `--reference` is omitted, the first image found in `--input` is used automatically.

---

### 3. Visualize before/after results

```bash
# Compare specific images
python visualize_normalization.py \
    --reference data/sample/reference.png \
    --images data/raw/ADI/patch_001.png data/raw/MUC/patch_005.png \
    --output outputs/comparison.png

# Auto-pick 6 random images from a folder
python visualize_normalization.py \
    --reference data/sample/reference.png \
    --folder data/raw/ \
    --n 6 \
    --output outputs/comparison.png \
    --histograms
```

---

## Dataset

This stage is dataset-agnostic — it works on any folder of H&E patch images.

For this pipeline we use the **NCT-CRC-HE-100K** dataset:

| Detail | Info |
|---|---|
| Source | Zenodo: https://zenodo.org/record/1214456 |
| Size | 100,000 patches, 224×224 px, 9 tissue classes |
| Tissue | Human colorectal cancer (exact domain match for this project) |
| Format | PNG, RGB, already 224×224 |
| License | CC BY 4.0 |

### Download Steps

1. Visit https://zenodo.org/record/1214456
2. Download `NCT-CRC-HE-100K.zip` (~3.8 GB)
3. Extract to `data/raw/`

Expected structure after extraction:
```
data/raw/
    ADI/    ← Adipose tissue
    BACK/   ← Background
    DEB/    ← Debris
    LYM/    ← Lymphocytes
    MUC/    ← Mucus
    MUS/    ← Smooth muscle
    NORM/   ← Normal colon mucosa
    STR/    ← Cancer-associated stroma
    TUM/    ← Colorectal adenocarcinoma epithelium  ← most important class
```

### Choosing a Reference Image

Pick a **high-quality TUM (tumor) patch** as your reference image.
- Should be tissue-rich (minimal white/background)
- Should have clear purple (hematoxylin) and pink (eosin) regions

Copy it to `data/sample/reference.png`.

---

## Output

After running `normalize_dataset.py`, your `data/normalized/` folder will mirror the
structure of `data/raw/` but with all patches stain-normalized and ready for Stage 3.
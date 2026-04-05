"""
find_best_reference.py
----------------------
Selects the best reference image for Macenko stain normalization.

Why the previous version failed on JPEG images with diffuse RBCs
----------------------------------------------------------------
Three simultaneous failure modes:

1. RGB threshold too tight:
   Old filter: R>180 AND G<120 AND B<160
   JPEG compression blurs RBC pixels with neighbours, pushing G UP
   and B UP. An RBC at (220,80,100) becomes (210,125,148) after
   JPEG + median filter — it now passes G<120 as G=125 slips through.

2. Wrong color space for the check:
   Fixed thresholds on individual channels are fragile. The correct
   discriminator is the R-G channel GAP. Erythrocytes absorb green
   strongly regardless of JPEG compression — R is always substantially
   higher than G. Eosin pink has a smaller R-G gap AND elevated B.
   Using (R - G > 60) AND (B < R - 20) catches JPEG-shifted RBCs
   that the old filter missed.

3. Threshold too permissive:
   5% RBC threshold allowed images with visually obvious contamination
   to pass. Lowered to 3%.

4. Stain diversity gate removed:
   The eigenvalue ratio (eig2/eig1) was class-dependent — colon adenocarcinoma
   patches are inherently hematoxylin-heavy (dense nuclei) and score low even
   when perfectly stained. Removing this gate lets the medoid selection
   handle image quality implicitly.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, uniform_filter
from tqdm import tqdm


# ------------------------------------------------------------------
# OD conversion
# ------------------------------------------------------------------

def get_optical_density(img: np.ndarray, beta: float = 0.15):
    """
    Convert (H, W, 3) uint8 RGB to optical density.
    Uses 98th-percentile white point per channel.

    Returns
    -------
    od          : (H, W, 3)
    od_tissue   : (N, 3)    foreground pixels (any channel OD > beta)
    tissue_mask : (H, W)    bool
    """
    img_f       = median_filter(img.astype(np.float32), size=(3, 3, 1))
    Io          = np.maximum(np.percentile(img_f, 98, axis=(0, 1)), 1.0)
    od          = -np.log(np.clip(img_f, 1.0, None) / Io)
    tissue_mask = np.any(od > beta, axis=2)
    od_tissue   = od[tissue_mask]
    return od, od_tissue, tissue_mask


# ------------------------------------------------------------------
# RBC detector — two independent layers
# ------------------------------------------------------------------

def erythrocyte_fraction(img: np.ndarray) -> tuple:
    """
    Detect erythrocyte contamination using both global fraction and
    local density (clustered RBC check).

    Returns (global_frac, max_local_density) as a tuple.

    Why two metrics:
      - Global fraction: works for images with many RBCs spread across tissue
      - Local density:   catches small but CLUSTERED RBC groups that occupy
                         <1% globally but dominate a local region, which is
                         enough to corrupt Macenko's stain vector estimation
                         since it uses percentile-based angle extremes.

    RBC detection layers:
      Layer 1 — R-G gap (JPEG-robust): R - G > 60, R > 160, B < R - 20
      Layer 2 — strict absolute: R > 185, G < 115, B < 155
    """
    R = img[:, :, 0].astype(np.int16)
    G = img[:, :, 1].astype(np.int16)
    B = img[:, :, 2].astype(np.int16)

    # Layer 1: R-G gap with tightened B cap
    # B < 160 excludes magenta staining artifacts (which have B=160-240)
    # while keeping real erythrocytes (B typically 90-155 in H&E)
    layer1   = (R - G > 60) & (R > 160) & (B < 160)

    # Layer 2: strict absolute thresholds for vivid/uncompressed RBCs
    layer2   = (R > 185) & (G < 115) & (B < 155)

    rbc_mask = (layer1 | layer2).astype(np.float32)

    global_frac   = float(rbc_mask.mean())
    # Local density: catches small but spatially concentrated RBC clusters
    # that occupy <1% globally but corrupt Macenko's percentile-based
    # stain vector estimation
    local_density = uniform_filter(rbc_mask, size=32)
    max_local     = float(local_density.max())

    return global_frac, max_local





# ------------------------------------------------------------------
# Per-image quality gates
# ------------------------------------------------------------------

def compute_features(path: Path, beta: float = 0.15, rbc_threshold: float = 0.03):
    """
    Compute Macenko-suitability features for one image.
    Returns None if any quality gate fails.

    Quality gates (in order)
    ------------------------
    1. White fraction > 60%    → mostly scanner background
    2. RBC fraction > 3%       → erythrocyte contamination
    3. Tissue ratio < 20%      → not enough foreground for stain estimation
    """
    img_raw = np.array(Image.open(path).convert("RGB"))

    # Gate 1: scanner background
    white_frac = (np.mean(img_raw, axis=2) > 245).mean()
    if white_frac > 0.60:
        return None

    # Gate 2: RBC contamination — global fraction AND local cluster density
    rbc_frac, rbc_local = erythrocyte_fraction(img_raw)
    if rbc_frac > rbc_threshold or rbc_local > 0.08:
        return None

    # OD conversion
    od, od_tissue, tissue_mask = get_optical_density(img_raw, beta=beta)
    tissue_ratio = tissue_mask.mean()

    # Gate 3: too little tissue
    if tissue_ratio < 0.20 or od_tissue.shape[0] < 50:
        return None

    mean_od = od_tissue.mean(axis=0)

    return {
        "mean_od":      mean_od,
        "tissue_ratio": float(tissue_ratio),
        "rbc_frac":     rbc_frac,
    }


# ------------------------------------------------------------------
# Diagnostic: check a single image and explain why it passes/fails
# ------------------------------------------------------------------

def diagnose_image(path: Path, beta: float = 0.15, rbc_threshold: float = 0.03):
    """
    Run all quality gates on a single image and print a full report.
    Useful for understanding why a specific image was accepted or rejected.
    """
    img_raw = np.array(Image.open(path).convert("RGB"))

    print(f"\nDiagnosing: {path.name}")
    print(f"  Image size : {img_raw.shape}")

    white_frac = float((np.mean(img_raw, axis=2) > 245).mean())
    print(f"  White frac : {white_frac*100:.1f}%  (gate: >60% = reject)")
    if white_frac > 0.60:
        print("  → REJECTED at Gate 1 (too much background)")
        return

    rbc_frac, rbc_local = erythrocyte_fraction(img_raw)
    print(f"  RBC global : {rbc_frac*100:.2f}%  (gate: >{rbc_threshold*100:.0f}% = reject)")
    print(f"  RBC local  : {rbc_local*100:.1f}%  (gate: >8% in any 32x32 window = reject)")
    if rbc_frac > rbc_threshold or rbc_local > 0.10:
        print("  → REJECTED at Gate 2 (RBC contamination: global or clustered)")
        return

    od, od_tissue, tissue_mask = get_optical_density(img_raw, beta=beta)
    tissue_ratio = float(tissue_mask.mean())
    print(f"  Tissue     : {tissue_ratio*100:.1f}%  (gate: <20% = reject)")
    if tissue_ratio < 0.20 or od_tissue.shape[0] < 50:
        print("  → REJECTED at Gate 3 (too little tissue)")
        return

    print("  → PASSED all gates ✓")


# ------------------------------------------------------------------
# Main selection: medoid in OD space
# ------------------------------------------------------------------

def find_best_reference(
    folder: Path,
    ext: str,
    top_n: int = 5,
    beta: float = 0.15,
    rbc_threshold: float = 0.03,
    limit: int = None,
):
    # Accept both .jpg and .jpeg regardless of what was passed
    exts = [".jpg", ".jpeg"] if ext.lower() in (".jpg", ".jpeg") else [ext]
    image_paths = sorted(p for e in exts for p in folder.rglob(f"*{e}"))
    image_paths = sorted(set(image_paths))  # deduplicate, keep sorted
    if not image_paths:
        raise FileNotFoundError(f"No {ext} images found in {folder}")
    if limit:
        image_paths = image_paths[:limit]
        print(f"\n--limit {limit}: scanning first {len(image_paths)} images only.")

    print(f"\nAnalyzing {len(image_paths)} images...")

    valid_paths = []
    features    = []
    meta        = []
    rejected    = 0
    reject_log  = {"white": 0, "rbc_rgb": 0, "tissue": 0}

    for path in tqdm(image_paths, unit="img", desc="Scanning"):
        try:
            img_raw = np.array(Image.open(path).convert("RGB"))

            if (np.mean(img_raw, axis=2) > 245).mean() > 0.60:
                reject_log["white"] += 1; rejected += 1; continue

            rbc_frac, rbc_local = erythrocyte_fraction(img_raw)
            if rbc_frac > rbc_threshold or rbc_local > 0.08:
                reject_log["rbc_rgb"] += 1; rejected += 1; continue

            od, od_tissue, tissue_mask = get_optical_density(img_raw, beta=beta)
            tissue_ratio = tissue_mask.mean()

            if tissue_ratio < 0.20 or od_tissue.shape[0] < 50:
                reject_log["tissue"] += 1; rejected += 1; continue

            valid_paths.append(path)
            features.append(od_tissue.mean(axis=0))
            meta.append((rbc_frac, float(tissue_ratio)))

        except Exception:
            rejected += 1

    scanned      = len(image_paths)
    rbc_rejected = reject_log["rbc_rgb"]
    rbc_rate     = rbc_rejected / scanned if scanned > 0 else 0

    # If >60% of the sample was rejected purely for RBC contamination,
    # the sample is unrepresentative — the class folder may have a run of
    # RBC-heavy images at the start (files are sorted alphabetically, and
    # some scanner sessions produce more blood artifacts than others).
    # Better to reject the whole sample and ask for a larger limit.
    if rbc_rate > 0.60:
        total_in_folder = scanned  # minimum known
        suggested_limit = (limit or scanned) * 3
        raise ValueError(
            f"\n{'='*60}\n"
            f"  SAMPLE REJECTED — RBC contamination too pervasive\n"
            f"{'='*60}\n"
            f"  {rbc_rejected}/{scanned} images ({rbc_rate*100:.0f}%) failed the RBC gate.\n"
            f"  This likely means the first {scanned} images in this folder\n"
            f"  happen to be from a high-RBC scanner session.\n"
            f"\n"
            f"  Action: increase --limit to sample more images\n"
            f"  Suggested: --limit {suggested_limit}\n"
            f"\n"
            f"  Alternatively, if RBC contamination is genuinely\n"
            f"  pervasive in your dataset, relax the threshold:\n"
            f"  --rbc-threshold 0.06\n"
            f"{'='*60}"
        )

    if not valid_paths:
        raise ValueError(
            f"\n{'='*60}\n"
            f"  NO VALID CANDIDATES FOUND\n"
            f"{'='*60}\n"
            f"  Rejection breakdown:\n"
            f"    white    : {reject_log['white']} images (too much background)\n"
            f"    rbc_rgb  : {reject_log['rbc_rgb']} images (RBC contamination)\n"
            f"    tissue   : {reject_log['tissue']} images (too little tissue)\n"
            f"\n"
            f"  Try:\n"
            f"    --limit {(limit or scanned) * 3}   (scan more images)\n"
            f"    --rbc-threshold 0.06              (relax RBC filter)\n"
            f"    --beta 0.10                       (lower tissue threshold)\n"
            f"{'='*60}"
        )

    # Warn (but don't fail) if passed pool is very small relative to sample
    pass_rate = len(valid_paths) / scanned
    if pass_rate < 0.05 and len(valid_paths) < 20:
        print(
            f"\nWarning: only {len(valid_paths)}/{scanned} images passed filters "
            f"({pass_rate*100:.1f}%). Reference may not be representative.\n"
            f"Consider: --limit {(limit or scanned) * 3} for a larger sample.\n"
        )

    features_arr = np.array(features)
    centroid     = features_arr.mean(axis=0)
    std          = features_arr.std(axis=0) + 1e-8
    distances    = np.linalg.norm((features_arr - centroid) / std, axis=1)

    order    = np.argsort(distances)
    top_list = [
        (valid_paths[i], float(distances[i]),
         meta[i][0], meta[i][1])
        for i in order[:top_n]
    ]

    return top_list[0][0], top_list, rejected, reject_log


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Select the best Macenko reference image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--folder",   required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--ext",      default=".jpeg",
                   help="Image extension (default: .jpeg). Both .jpg and .jpeg are accepted.")
    p.add_argument("--top",      type=int,   default=5)
    p.add_argument("--beta",     type=float, default=0.15)
    p.add_argument("--rbc-threshold", type=float, default=0.02, dest="rbc_threshold",
                   help="Max RBC pixel fraction (default: 0.02 = 2%%)")
    p.add_argument("--limit",    type=int, default=None,
                   help="Scan only first N images (for quick test runs)")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--diagnose", default=None,
                   help="Path to a single image to diagnose (explain pass/fail gates)")
    args = p.parse_args()

    # Diagnose mode: explain a single image
    if args.diagnose:
        diagnose_image(Path(args.diagnose), beta=args.beta, rbc_threshold=args.rbc_threshold)
        return

    folder = Path(args.folder).resolve()
    output = Path(args.output).resolve()

    try:
        best_path, top_candidates, rejected, reject_log = find_best_reference(
            folder, args.ext, top_n=args.top,
            beta=args.beta, rbc_threshold=args.rbc_threshold,
            limit=args.limit,
        )
    except ValueError as e:
        print(str(e))
        raise SystemExit(1)

    _exts = [".jpg", ".jpeg"] if args.ext.lower() in (".jpg", ".jpeg") else [args.ext]
    total = len(set(p for e in _exts for p in folder.rglob(f"*{e}")))

    print(f"\n{'─'*72}")
    print(f"  Total: {total}  |  Rejected: {rejected}  |  Passed: {total - rejected}")
    print(f"  Rejection breakdown → white:{reject_log['white']}  "
          f"rbc_rgb:{reject_log['rbc_rgb']}  tissue:{reject_log['tissue']}")
    print(f"{'─'*72}")
    print(f"  {'Rk':<3} {'Filename':<36} {'Dist':>6}  {'RBC%':>6}  {'Tissue':>6}")
    print(f"{'─'*60}")
    for rank, (path, dist, rbc, tissue) in enumerate(top_candidates, 1):
        marker = " <- SELECTED" if rank == 1 else ""
        print(f"  {rank:<3} {path.name:<36} {dist:>6.4f}  "
              f"{rbc*100:>5.1f}%  {tissue*100:>5.1f}%{marker}")
    print(f"{'─'*72}\n")

    if args.dry_run:
        print("--dry-run: no file copied.")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, output)
    print(f"Reference saved to: {output}")


if __name__ == "__main__":
    main()
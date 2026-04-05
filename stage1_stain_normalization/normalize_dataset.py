"""
normalize_dataset.py
--------------------
Batch normalize an entire folder of H&E patches using Macenko normalization.

Usage:
    # Normalize full dataset
    python normalize_dataset.py \
        --input  data/raw/ \
        --output data/normalized/ \
        --reference data/sample/reference.png \
        --ext .png \
        --workers 4

    # Quick test: normalize only first N images
    python normalize_dataset.py \
        --input  data/raw/ \
        --output data/normalized_test/ \
        --reference data/sample/reference.png \
        --limit 10

Folder structure mirrored from input to output:
    data/raw/class_01/patch_001.png  →  data/normalized/class_01/patch_001.png
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from macenko import MacenkoNormalizer

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Worker (top-level for multiprocessing pickle)
# ------------------------------------------------------------------

def _normalize_one(args):
    """
    Normalize a single image file.

    Each worker builds and fits its own MacenkoNormalizer instance —
    the fitted HERef/maxCRef arrays are passed in as plain numpy arrays
    so they pickle cleanly.

    Args:
        args : (src_path, dst_path, he_ref, max_c_ref, Io, alpha, beta)

    Returns:
        (src_path_str, success: bool, error: str | None)
    """
    src_path, dst_path, he_ref, max_c_ref, Io, alpha, beta = args

    try:
        normalizer         = MacenkoNormalizer(Io=Io, alpha=alpha, beta=beta)
        normalizer.HERef   = he_ref
        normalizer.maxCRef = max_c_ref
        normalizer._fitted = True

        image      = Image.open(src_path).convert("RGB")
        normalized = normalizer.transform(image)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        normalized.save(dst_path)

        return str(src_path), True, None

    except Exception as e:
        return str(src_path), False, str(e)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def collect_image_paths(input_dir: Path, ext: str):
    # Accept both .jpg and .jpeg regardless of what was passed
    exts = [".jpg", ".jpeg"] if ext.lower() in (".jpg", ".jpeg") else [ext]
    return sorted(set(p for e in exts for p in input_dir.rglob(f"*{e}")))


def find_reference_image(input_dir: Path, ext: str) -> Path:
    images = collect_image_paths(input_dir, ext)
    if not images:
        log.error(f"No images found in {input_dir} with extension {ext}")
        sys.exit(1)
    log.info(f"Auto-selected reference image: {images[0]}")
    return images[0]


def build_dst_path(src: Path, input_dir: Path, output_dir: Path) -> Path:
    return output_dir / src.relative_to(input_dir)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run(args):
    input_dir  = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    ext        = args.ext if args.ext.startswith(".") else f".{args.ext}"

    if not input_dir.exists():
        log.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # ------ Reference image ------
    if args.reference:
        ref_path = Path(args.reference).resolve()
        if not ref_path.exists():
            log.error(f"Reference image not found: {ref_path}")
            sys.exit(1)
    else:
        ref_path = find_reference_image(input_dir, ext)

    reference_img = Image.open(ref_path).convert("RGB")
    log.info(f"Reference: {ref_path}  size={reference_img.size}")

    # Fit normalizer once in the main process; workers receive fitted arrays
    log.info("Fitting Macenko normalizer to reference image...")
    normalizer = MacenkoNormalizer(Io=args.Io, alpha=args.alpha, beta=args.beta)
    try:
        normalizer.fit(reference_img)
        log.info("Fit complete. HERef estimated from reference image.")
    except ValueError as e:
        log.warning(f"fit() failed ({e}). Falling back to fixed MATLAB reference vectors.")

    # ------ Collect images ------
    all_images = collect_image_paths(input_dir, ext)
    log.info(f"Found {len(all_images)} images in {input_dir}")

    if args.limit:
        all_images = all_images[: args.limit]
        log.info(f"--limit {args.limit}: processing first {len(all_images)} images only.")

    if not all_images:
        log.warning("Nothing to process. Exiting.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass fitted arrays directly (numpy arrays are pickleable)
    tasks = [
        (
            src,
            build_dst_path(src, input_dir, output_dir),
            normalizer.HERef,
            normalizer.maxCRef,
            normalizer.Io,
            normalizer.alpha,
            normalizer.beta,
        )
        for src in all_images
    ]

    # ------ Process ------
    failed   = []
    succeeded = 0

    log.info(f"Starting normalization with {args.workers} worker(s)...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_normalize_one, t): t[0] for t in tasks}

        with tqdm(total=len(tasks), unit="img", desc="Normalizing") as pbar:
            for future in as_completed(futures):
                src_path, success, error = future.result()
                if success:
                    succeeded += 1
                else:
                    failed.append((src_path, error))
                pbar.update(1)

    # ------ Summary ------
    log.info(f"Done.  Succeeded: {succeeded}  Failed: {len(failed)}")

    if failed:
        log.warning("Failed images:")
        for path, err in failed[:20]:
            log.warning(f"  {path}  →  {err}")
        if len(failed) > 20:
            log.warning(f"  ... and {len(failed) - 20} more")

    log.info(f"Output saved to: {output_dir}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch Macenko stain normalization for H&E patches."
    )
    p.add_argument("--input",     required=True,       help="Root folder of raw patches")
    p.add_argument("--output",    required=True,       help="Root folder for normalized output")
    p.add_argument("--reference", default=None,        help="Reference image path (optional)")
    p.add_argument("--ext",       default=".jpeg",     help="Image extension (default: .jpeg). Both .jpg and .jpeg are accepted.")
    p.add_argument("--workers",   type=int, default=4, help="Parallel workers (default: 4)")
    p.add_argument("--limit",     type=int, default=None,
                   help="Process only first N images (for quick test runs)")
    p.add_argument("--Io",        type=int,   default=240,  help="Transmitted light intensity")
    p.add_argument("--alpha",     type=float, default=1.0,  help="Angle percentile tolerance")
    p.add_argument("--beta",      type=float, default=0.15, help="OD background threshold")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
"""
extract_patches.py
------------------
Extracts fixed-size patches from histopathology images.

This module works in TWO modes:

MODE 1 — WSI Mode (production/Colab):
    Reads gigapixel .svs/.ndpi Whole Slide Images using OpenSlide,
    extracts non-overlapping 224x224 patches at a specified magnification,
    filters out background tiles, saves tissue patches organized by class.

MODE 2 — Standard Image Mode (local development):
    Works on regular images (JPEG/PNG/TIF) by treating them as
    high-resolution source images and extracting sub-patches.
    Used for demonstration and testing without WSI files.

Usage (Mode 2 — local development):
    extractor = PatchExtractor(patch_size=224, stride=224)
    patches   = extractor.extract_from_image("path/to/image.jpeg")

Usage (Mode 1 — WSI, requires openslide):
    extractor = PatchExtractor(patch_size=224, level=0)
    patches   = extractor.extract_from_wsi("path/to/slide.svs")

    # Or batch process entire folder:
    python extract_patches.py \\
        --input  data/raw/colon_image_sets \\
        --output data/patches \\
        --size   224 \\
        --stride 224 \\
        --mode   image
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from filter_background import BackgroundFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class PatchExtractor:
    """
    Extracts fixed-size patches from histopathology images.

    Parameters
    ----------
    patch_size : int
        Size of extracted patches (patch_size x patch_size). Default: 224.
    stride     : int
        Step between patches. stride == patch_size → no overlap.
        stride < patch_size → overlapping patches. Default: 224.
    min_tissue_ratio : float
        Minimum tissue coverage for a patch to be kept. Default: 0.15.
    brightness_threshold : int
        Pixel brightness above which pixels are considered background. Default: 210.
    """

    def __init__(
        self,
        patch_size           : int   = 224,
        stride               : int   = 224,
        min_tissue_ratio     : float = 0.05,
        brightness_threshold : int   = 230,
    ):
        self.patch_size  = patch_size
        self.stride      = stride
        self.bg_filter   = BackgroundFilter(
            brightness_threshold=brightness_threshold,
            min_tissue_ratio=min_tissue_ratio,
        )

    # ------------------------------------------------------------------
    # Mode 2: Standard image extraction
    # ------------------------------------------------------------------

    def extract_from_image(
        self,
        image_path,
        filter_background: bool = True,
    ) -> List[np.ndarray]:
        """
        Extract patches from a standard image file (JPEG/PNG/TIF).

        For images smaller than patch_size, returns the resized image as
        a single patch. For larger images, uses sliding window extraction.

        Args:
            image_path       : Path to image file or PIL Image
            filter_background: If True, remove blank/background patches

        Returns:
            List of numpy arrays (patch_size, patch_size, 3) uint8
        """
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        img_array = np.array(image)
        h, w      = img_array.shape[:2]

        # If image is smaller than patch size, just resize and return
        if h < self.patch_size or w < self.patch_size:
            resized = image.resize(
                (self.patch_size, self.patch_size),
                Image.LANCZOS
            )
            patch = np.array(resized)
            if filter_background and not self.bg_filter.is_tissue(patch):
                return []
            return [patch]

        # Sliding window extraction
        patches = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img_array[y:y+self.patch_size, x:x+self.patch_size]

                if filter_background and not self.bg_filter.is_tissue(patch):
                    continue

                patches.append(patch)

        return patches

    def extract_from_image_with_coords(
        self,
        image_path,
        filter_background: bool = True,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Same as extract_from_image but also returns (x, y) coordinates.
        Useful for reconstructing patch positions on the original image.

        Returns:
            List of (patch_array, x, y) tuples
        """
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path.convert("RGB")

        img_array = np.array(image)
        h, w      = img_array.shape[:2]

        results = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img_array[y:y+self.patch_size, x:x+self.patch_size]

                if filter_background and not self.bg_filter.is_tissue(patch):
                    continue

                results.append((patch, x, y))

        return results

    # ------------------------------------------------------------------
    # Mode 1: WSI extraction (requires openslide)
    # ------------------------------------------------------------------

    def extract_from_wsi(
        self,
        wsi_path,
        level            : int  = 0,
        filter_background: bool = True,
    ) -> List[np.ndarray]:
        """
        Extract patches from a Whole Slide Image (.svs, .ndpi, .tif).
        Requires openslide-python to be installed.

        Args:
            wsi_path         : Path to .svs or other WSI file
            level            : Magnification level (0 = highest resolution)
            filter_background: If True, skip background tiles

        Returns:
            List of numpy arrays (patch_size, patch_size, 3) uint8

        Raises:
            ImportError: if openslide is not installed
            FileNotFoundError: if WSI file doesn't exist
        """
        try:
            import openslide
        except ImportError:
            raise ImportError(
                "openslide-python is required for WSI extraction.\n"
                "Install with: pip install openslide-python\n"
                "Also install OpenSlide C library: https://openslide.org/download/"
            )

        wsi_path = Path(wsi_path)
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")

        slide    = openslide.OpenSlide(str(wsi_path))
        w, h     = slide.level_dimensions[level]

        log.info(f"WSI: {wsi_path.name}  size=({w}, {h})  level={level}")

        patches = []
        total_tiles = ((h - self.patch_size) // self.stride + 1) * \
                      ((w - self.patch_size) // self.stride + 1)

        with tqdm(total=total_tiles, desc=f"Extracting {wsi_path.stem}") as pbar:
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    # Read region from WSI
                    region = slide.read_region(
                        location=(x, y),
                        level=level,
                        size=(self.patch_size, self.patch_size)
                    ).convert("RGB")

                    patch = np.array(region)

                    if filter_background and not self.bg_filter.is_tissue(patch):
                        pbar.update(1)
                        continue

                    patches.append(patch)
                    pbar.update(1)

        slide.close()
        log.info(f"Extracted {len(patches)} tissue patches from {wsi_path.name}")
        return patches

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_folder(
        self,
        input_dir : Path,
        output_dir: Path,
        ext       : str  = ".jpeg",
        mode      : str  = "image",
        save      : bool = True,
    ) -> dict:
        """
        Process all images in a folder and save extracted patches.

        Preserves class folder structure:
            input_dir/colon_aca/*.jpeg → output_dir/colon_aca/patch_001.png

        Args:
            input_dir : Root folder containing class subfolders
            output_dir: Where to save patches
            ext       : Image extension
            mode      : "image" for standard images, "wsi" for WSI files
            save      : If True, save patches to disk

        Returns:
            dict with statistics per class
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)

        stats = {}

        # Find all class folders
        class_folders = [f for f in input_dir.iterdir() if f.is_dir()]

        if not class_folders:
            # No subfolders — treat input_dir itself as single class
            class_folders = [input_dir]

        for class_folder in sorted(class_folders):
            class_name  = class_folder.name
            image_files = sorted(class_folder.rglob(f"*{ext}"))

            if not image_files:
                continue

            out_class_dir = output_dir / class_name
            if save:
                out_class_dir.mkdir(parents=True, exist_ok=True)

            n_patches = 0
            n_images  = 0

            log.info(f"Processing class: {class_name}  ({len(image_files)} images)")

            for img_path in tqdm(image_files, desc=class_name, unit="img"):
                try:
                    if mode == "wsi":
                        patches = self.extract_from_wsi(img_path)
                    else:
                        patches = self.extract_from_image(img_path)

                    if save:
                        stem = img_path.stem
                        for i, patch in enumerate(patches):
                            patch_path = out_class_dir / f"{stem}_p{i:04d}.png"
                            Image.fromarray(patch).save(patch_path)

                    n_patches += len(patches)
                    n_images  += 1

                except Exception as e:
                    log.warning(f"Failed: {img_path.name} → {e}")

            stats[class_name] = {
                "images" : n_images,
                "patches": n_patches,
                "avg_patches_per_image": n_patches / max(n_images, 1),
            }
            log.info(f"  {class_name}: {n_patches} patches from {n_images} images")

        return stats


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract patches from histopathology images or WSIs."
    )
    p.add_argument("--input",  required=True, help="Input folder (with class subfolders)")
    p.add_argument("--output", required=True, help="Output folder for patches")
    p.add_argument("--size",   type=int, default=224,  help="Patch size (default: 224)")
    p.add_argument("--stride", type=int, default=224,  help="Stride (default: 224 = no overlap)")
    p.add_argument("--ext",    default=".jpeg",        help="Image extension (default: .jpeg)")
    p.add_argument("--mode",   default="image",
                   choices=["image", "wsi"],
                   help="Extraction mode: 'image' for JPEG/PNG, 'wsi' for .svs files")
    p.add_argument("--min-tissue", type=float, default=0.15,
                   help="Minimum tissue ratio to keep a patch (default: 0.15)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extractor = PatchExtractor(
        patch_size=args.size,
        stride=args.stride,
        min_tissue_ratio=args.min_tissue,
    )

    stats = extractor.process_folder(
        input_dir =Path(args.input),
        output_dir=Path(args.output),
        ext=args.ext,
        mode=args.mode,
    )

    print("\n=== Extraction Summary ===")
    for class_name, s in stats.items():
        print(f"{class_name:20s}  images={s['images']:5d}  "
              f"patches={s['patches']:6d}  "
              f"avg={s['avg_patches_per_image']:.1f}")

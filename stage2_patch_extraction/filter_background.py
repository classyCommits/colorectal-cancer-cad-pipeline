"""
filter_background.py
--------------------
Filters out background (white/blank) patches from histopathology images.

In WSI patch extraction, many extracted tiles are blank white regions
(glass slide with no tissue). These carry zero information and hurt
model training. This module detects and removes them.

Strategy:
    A patch is considered background if:
    1. Mean pixel intensity > threshold (too bright = white/empty)
    2. Tissue coverage < min_tissue_ratio (not enough dark pixels)
    3. Standard deviation < std_threshold (too uniform = blank)

Usage:
    filter = BackgroundFilter()
    is_tissue = filter.is_tissue(patch)        # single patch
    patches   = filter.filter_patches(patches) # list of patches
"""

import numpy as np
from PIL import Image


class BackgroundFilter:
    """
    Detects and filters background patches from H&E histopathology images.

    Parameters
    ----------
    brightness_threshold : int
        Mean pixel value above which a patch is considered background.
        H&E tissue is darker than blank glass. Default: 210.
    min_tissue_ratio : float
        Minimum fraction of pixels that must be tissue (dark enough).
        Default: 0.15 (at least 15% tissue coverage required).
    std_threshold : float
        Minimum standard deviation of pixel values.
        Very low std = uniform blank patch. Default: 10.0.
    """

    def __init__(
        self,
        brightness_threshold: int   = 230,
        min_tissue_ratio: float     = 0.05,
        std_threshold: float        = 8.0,
    ):
        self.brightness_threshold = brightness_threshold
        self.min_tissue_ratio     = min_tissue_ratio
        self.std_threshold        = std_threshold

    def _to_numpy_gray(self, patch) -> np.ndarray:
        """Convert patch to grayscale numpy array."""
        if isinstance(patch, Image.Image):
            return np.array(patch.convert("L"), dtype=np.float32)
        # If RGB numpy array, convert to grayscale
        if patch.ndim == 3:
            return np.mean(patch, axis=2).astype(np.float32)
        return patch.astype(np.float32)

    def is_tissue(self, patch) -> bool:
        """
        Returns True if patch contains enough tissue, False if background.

        Args:
            patch: PIL Image or numpy array (H, W) or (H, W, 3)

        Returns:
            bool: True = tissue patch, False = background/blank
        """
        gray = self._to_numpy_gray(patch)

        # Check 1: Overall brightness — too bright = blank glass
        mean_brightness = gray.mean()
        if mean_brightness > self.brightness_threshold:
            return False

        # Check 2: Tissue coverage — count dark pixels
        tissue_pixels = np.sum(gray < self.brightness_threshold)
        tissue_ratio  = tissue_pixels / gray.size
        if tissue_ratio < self.min_tissue_ratio:
            return False

        # Check 3: Uniformity — very low std = blank or near-blank
        if gray.std() < self.std_threshold:
            return False

        return True

    def tissue_ratio(self, patch) -> float:
        """
        Returns the fraction of pixels classified as tissue (0.0 to 1.0).
        Useful for ranking patches by tissue content.
        """
        gray = self._to_numpy_gray(patch)
        tissue_pixels = np.sum(gray < self.brightness_threshold)
        return float(tissue_pixels / gray.size)

    def filter_patches(self, patches: list) -> list:
        """
        Filter a list of patches, keeping only tissue patches.

        Args:
            patches: list of PIL Images or numpy arrays

        Returns:
            list of patches that passed the tissue filter
        """
        return [p for p in patches if self.is_tissue(p)]

    def filter_with_stats(self, patches: list) -> dict:
        """
        Filter patches and return detailed statistics.

        Returns:
            dict with keys:
                tissue    : list of tissue patches
                background: list of background patches
                total     : int
                n_tissue  : int
                n_background: int
                tissue_pct: float
        """
        tissue     = []
        background = []

        for patch in patches:
            if self.is_tissue(patch):
                tissue.append(patch)
            else:
                background.append(patch)

        total = len(patches)
        return {
            "tissue"      : tissue,
            "background"  : background,
            "total"       : total,
            "n_tissue"    : len(tissue),
            "n_background": len(background),
            "tissue_pct"  : len(tissue) / total * 100 if total > 0 else 0.0,
        }

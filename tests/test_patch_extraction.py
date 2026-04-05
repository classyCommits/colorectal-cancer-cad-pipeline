"""
tests/test_patch_extraction.py
------------------------------
Unit tests for Stage 2 — Patch Extraction.

Run with:
    pytest tests/test_patch_extraction.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "stage2_patch_extraction"))
from extract_patches import PatchExtractor
from filter_background import BackgroundFilter


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_tissue_patch(size=300, seed=42) -> np.ndarray:
    """Synthetic tissue-like patch — dark enough to pass background filter."""
    rng = np.random.default_rng(seed)
    patch = rng.integers(50, 180, size=(size, size, 3), dtype=np.uint8)
    return patch


def make_background_patch(size=300) -> np.ndarray:
    """Synthetic background patch — very bright/white."""
    return np.full((size, size, 3), 240, dtype=np.uint8)


@pytest.fixture
def tissue_image():
    return Image.fromarray(make_tissue_patch(size=300))


@pytest.fixture
def background_image():
    return Image.fromarray(make_background_patch(size=300))


@pytest.fixture
def extractor():
    return PatchExtractor(patch_size=100, stride=100)


@pytest.fixture
def bg_filter():
    return BackgroundFilter()


# ------------------------------------------------------------------
# BackgroundFilter tests
# ------------------------------------------------------------------

class TestBackgroundFilter:
    def test_tissue_patch_passes(self, bg_filter):
        patch = make_tissue_patch()
        assert bg_filter.is_tissue(patch) is True

    def test_background_patch_fails(self, bg_filter):
        patch = make_background_patch()
        assert bg_filter.is_tissue(patch) is False

    def test_accepts_pil_image(self, bg_filter):
        pil = Image.fromarray(make_tissue_patch())
        assert bg_filter.is_tissue(pil) is True

    def test_tissue_ratio_tissue(self, bg_filter):
        patch = make_tissue_patch()
        ratio = bg_filter.tissue_ratio(patch)
        assert ratio > 0.5, f"Expected >0.5 for tissue, got {ratio:.2f}"

    def test_tissue_ratio_background(self, bg_filter):
        patch = make_background_patch()
        ratio = bg_filter.tissue_ratio(patch)
        assert ratio < 0.1, f"Expected <0.1 for background, got {ratio:.2f}"

    def test_filter_patches_removes_background(self, bg_filter):
        patches = [make_tissue_patch(seed=i) for i in range(3)] + \
                  [make_background_patch() for _ in range(2)]
        filtered = bg_filter.filter_patches(patches)
        assert len(filtered) == 3

    def test_filter_with_stats(self, bg_filter):
        patches = [make_tissue_patch(seed=i) for i in range(4)] + \
                  [make_background_patch() for _ in range(1)]
        stats = bg_filter.filter_with_stats(patches)
        assert stats["total"]       == 5
        assert stats["n_tissue"]    == 4
        assert stats["n_background"] == 1
        assert stats["tissue_pct"]  == 80.0


# ------------------------------------------------------------------
# PatchExtractor tests
# ------------------------------------------------------------------

class TestPatchExtractor:
    def test_extract_returns_list(self, extractor, tissue_image):
        patches = extractor.extract_from_image(tissue_image)
        assert isinstance(patches, list)

    def test_patch_shape(self, extractor, tissue_image):
        patches = extractor.extract_from_image(tissue_image)
        assert len(patches) > 0
        for p in patches:
            assert p.shape == (100, 100, 3), f"Expected (100,100,3) got {p.shape}"

    def test_patch_dtype(self, extractor, tissue_image):
        patches = extractor.extract_from_image(tissue_image)
        for p in patches:
            assert p.dtype == np.uint8

    def test_correct_number_of_patches(self, extractor, tissue_image):
        """300x300 image with 100x100 patches and stride 100 → 3x3 = 9 patches max."""
        patches = extractor.extract_from_image(tissue_image, filter_background=False)
        assert len(patches) == 9

    def test_background_filtering_works(self, extractor, background_image):
        """Background image should yield 0 patches when filtering is on."""
        patches = extractor.extract_from_image(background_image, filter_background=True)
        assert len(patches) == 0

    def test_no_filter_keeps_background(self, extractor, background_image):
        """With filter off, patches are returned even from background."""
        patches = extractor.extract_from_image(background_image, filter_background=False)
        assert len(patches) > 0

    def test_small_image_returns_one_patch(self):
        """Image smaller than patch_size should return exactly 1 resized patch."""
        extractor = PatchExtractor(patch_size=224, stride=224)
        small_img = Image.fromarray(make_tissue_patch(size=100))
        patches   = extractor.extract_from_image(small_img)
        assert len(patches) == 1
        assert patches[0].shape == (224, 224, 3)

    def test_extract_with_coords_returns_tuples(self, extractor, tissue_image):
        results = extractor.extract_from_image_with_coords(tissue_image)
        assert len(results) > 0
        for item in results:
            assert len(item) == 3
            patch, x, y = item
            assert isinstance(x, (int, np.integer))
            assert isinstance(y, (int, np.integer))

    def test_coords_are_valid(self, extractor, tissue_image):
        results = extractor.extract_from_image_with_coords(
            tissue_image, filter_background=False
        )
        img_arr = np.array(tissue_image)
        h, w    = img_arr.shape[:2]
        for patch, x, y in results:
            assert 0 <= x <= w - extractor.patch_size
            assert 0 <= y <= h - extractor.patch_size

    def test_stride_overlap(self):
        """Stride smaller than patch_size should produce more patches."""
        img = Image.fromarray(make_tissue_patch(size=300))

        e_no_overlap  = PatchExtractor(patch_size=100, stride=100)
        e_with_overlap = PatchExtractor(patch_size=100, stride=50)

        p_no_overlap   = e_no_overlap.extract_from_image(img, filter_background=False)
        p_with_overlap = e_with_overlap.extract_from_image(img, filter_background=False)

        assert len(p_with_overlap) > len(p_no_overlap)

    def test_wsi_import_error_without_openslide(self, extractor):
        """Should raise ImportError if openslide not installed."""
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"openslide": None}):
            with pytest.raises((ImportError, TypeError)):
                extractor.extract_from_wsi("fake_slide.svs")

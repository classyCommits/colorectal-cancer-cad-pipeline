"""
tests/test_normalization.py
---------------------------
Unit tests for MacenkoNormalizer.

Run with:
    pytest tests/test_normalization.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add stage1 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "stage1_stain_normalization"))
from macenko.py import MacenkoNormalizer


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def make_synthetic_he_patch(seed: int = 42, size: int = 64) -> np.ndarray:
    """
    Creates a synthetic H&E-like patch with realistic OD values.
    Not a real H&E image but sufficient to test the math.
    """
    rng = np.random.default_rng(seed)
    # Simulate tissue colors: purple/blue (hematoxylin) + pink (eosin)
    patch = rng.integers(80, 230, size=(size, size, 3), dtype=np.uint8)
    # Add some darker (tissue) pixels to ensure foreground exists
    patch[10:40, 10:40] = rng.integers(40, 130, size=(30, 30, 3), dtype=np.uint8)
    return patch


@pytest.fixture
def reference_array():
    return make_synthetic_he_patch(seed=0)


@pytest.fixture
def target_array():
    return make_synthetic_he_patch(seed=99)


@pytest.fixture
def fitted_normalizer(reference_array):
    n = MacenkoNormalizer()
    n.fit(reference_array)
    return n


# ------------------------------------------------------------------
# Tests: fit()
# ------------------------------------------------------------------

class TestFit:
    def test_fit_sets_stain_matrix(self, reference_array):
        n = MacenkoNormalizer()
        n.fit(reference_array)
        assert n.stain_matrix_ref is not None

    def test_stain_matrix_shape(self, reference_array):
        n = MacenkoNormalizer()
        n.fit(reference_array)
        assert n.stain_matrix_ref.shape == (2, 3), (
            f"Expected (2, 3), got {n.stain_matrix_ref.shape}"
        )

    def test_fit_sets_max_concentrations(self, reference_array):
        n = MacenkoNormalizer()
        n.fit(reference_array)
        assert n.max_concentrations_ref is not None
        assert n.max_concentrations_ref.shape == (2,)

    def test_fit_accepts_pil_image(self, reference_array):
        n = MacenkoNormalizer()
        pil_img = Image.fromarray(reference_array)
        n.fit(pil_img)
        assert n.stain_matrix_ref is not None

    def test_fit_returns_self(self, reference_array):
        n = MacenkoNormalizer()
        result = n.fit(reference_array)
        assert result is n, "fit() should return self for chaining"


# ------------------------------------------------------------------
# Tests: transform()
# ------------------------------------------------------------------

class TestTransform:
    def test_transform_output_is_pil_image(self, fitted_normalizer, target_array):
        result = fitted_normalizer.transform(target_array)
        assert isinstance(result, Image.Image)

    def test_output_shape_matches_input(self, fitted_normalizer, target_array):
        result = fitted_normalizer.transform(target_array)
        result_array = np.array(result)
        assert result_array.shape == target_array.shape, (
            f"Shape mismatch: input {target_array.shape}, output {result_array.shape}"
        )

    def test_output_dtype_uint8(self, fitted_normalizer, target_array):
        result = np.array(fitted_normalizer.transform(target_array))
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, fitted_normalizer, target_array):
        result = np.array(fitted_normalizer.transform(target_array))
        assert result.min() >= 0
        assert result.max() <= 255

    def test_transform_accepts_pil_image(self, fitted_normalizer, target_array):
        pil_input = Image.fromarray(target_array)
        result = fitted_normalizer.transform(pil_input)
        assert isinstance(result, Image.Image)

    def test_transform_without_fit_raises(self, target_array):
        n = MacenkoNormalizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            n.transform(target_array)

    def test_transform_reference_is_stable(self, reference_array):
        """Normalizing the reference image against itself should produce stable output."""
        n = MacenkoNormalizer()
        n.fit(reference_array)
        result = np.array(n.transform(reference_array))
        # Output should be close to input (not identical due to float precision, but close)
        diff = np.abs(result.astype(int) - reference_array.astype(int))
        assert diff.mean() < 30, (
            f"Reference self-normalization diverged too much: mean diff = {diff.mean():.2f}"
        )


# ------------------------------------------------------------------
# Tests: fit_transform()
# ------------------------------------------------------------------

class TestFitTransform:
    def test_fit_transform_convenience(self, reference_array, target_array):
        n = MacenkoNormalizer()
        result = n.fit_transform(reference_array, target_array)
        assert isinstance(result, Image.Image)
        assert n.stain_matrix_ref is not None

    def test_fit_transform_equivalent_to_separate_calls(self, reference_array, target_array):
        n1 = MacenkoNormalizer()
        r1 = np.array(n1.fit_transform(reference_array, target_array))

        n2 = MacenkoNormalizer()
        n2.fit(reference_array)
        r2 = np.array(n2.transform(target_array))

        np.testing.assert_array_equal(r1, r2)


# ------------------------------------------------------------------
# Tests: edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_mostly_white_image_raises(self):
        """A blank white patch has no foreground pixels — should raise ValueError."""
        white = np.full((64, 64, 3), 240, dtype=np.uint8)
        n = MacenkoNormalizer(beta=0.15)
        with pytest.raises(ValueError, match="No foreground pixels"):
            n.fit(white)

    def test_different_sized_images(self, reference_array, fitted_normalizer):
        """Normalizer should handle images of any size, not just the reference size."""
        big_patch = make_synthetic_he_patch(seed=7, size=128)
        result = fitted_normalizer.transform(big_patch)
        assert np.array(result).shape == (128, 128, 3)

    def test_chaining(self, reference_array, target_array):
        """fit() returns self so chaining works."""
        result = MacenkoNormalizer().fit(reference_array).transform(target_array)
        assert isinstance(result, Image.Image)
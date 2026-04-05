"""
macenko.py
----------
Macenko stain normalization for H&E histopathology images.

Implements:
    Macenko et al., "A method for normalizing histology slides for
    quantitative analysis", IEEE ISBI 2009.
    DOI: 10.1109/ISBI.2009.5193250

Two modes of operation
----------------------
1. Fixed-reference mode (default, no fit needed):
       normalizer = MacenkoNormalizer()
       out = normalizer.transform(image)

2. Fit-to-reference mode (recommended when you have a good reference):
       normalizer = MacenkoNormalizer()
       normalizer.fit(reference_image)   # estimates HERef + maxCRef from ref
       out = normalizer.transform(image)

   In this mode the reference image's stain appearance becomes the target,
   giving tighter intra-dataset consistency than the MATLAB fixed vectors.
"""

import numpy as np
from PIL import Image


class MacenkoNormalizer:
    """
    Normalizes H&E stained images using the Macenko (2009) method.

    Parameters
    ----------
    Io    : int   Transmitted light intensity (default 240)
    alpha : float Tolerance for pseudo-min/max angle (default 1.0)
    beta  : float OD threshold — pixels below this are background (default 0.15)
    """

    # Fallback fixed reference (MATLAB paper values).
    # Used only when fit() has not been called.
    _HERef_fixed = np.array([
        [0.5626, 0.2159],
        [0.7201, 0.8012],
        [0.4062, 0.5581],
    ], dtype=np.float64)

    _maxCRef_fixed = np.array([1.9705, 1.0308], dtype=np.float64).reshape(2, 1)

    def __init__(self, Io: int = 240, alpha: float = 1.0, beta: float = 0.15):
        self.Io    = Io
        self.alpha = alpha
        self.beta  = beta

        # Will be overwritten by fit(); fall back to fixed values until then
        self.HERef   = self._HERef_fixed.copy()
        self.maxCRef = self._maxCRef_fixed.copy()
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_numpy(self, image) -> np.ndarray:
        """Accept PIL Image or numpy array; always return (H, W, 3) uint8."""
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"), dtype=np.uint8)
        return np.asarray(image, dtype=np.uint8)

    def _compute_od(self, I_flat: np.ndarray) -> np.ndarray:
        """
        Convert flattened (N, 3) uint8 RGB to optical density.
        +1 guard avoids log(0) on pure-black pixels.
        """
        return -np.log((I_flat.astype(np.float64) + 1) / self.Io)

    def _estimate_he_matrix(self, OD_hat: np.ndarray) -> np.ndarray:
        """
        Estimate the H&E stain matrix from foreground OD pixels.

        Uses eigh (symmetric eigensolver) — safer and faster than eig
        on a covariance matrix (always real, sorted eigenvalues).

        Args:
            OD_hat : (N, 3) foreground optical densities

        Returns:
            HE : (3, 2) — column 0 = hematoxylin, column 1 = eosin
        """
        cov = np.cov(OD_hat.T)                          # (3, 3) symmetric
        eigenvalues, eigenvectors = np.linalg.eigh(cov) # eigh: real, ascending

        # Two largest eigenvectors (eigh returns ascending → take last two)
        # Reshape to (3, 2) for projection
        V = eigenvectors[:, -2:]                         # (3, 2)

        # Project OD onto the 2-principal-component plane
        That = OD_hat @ V                                # (N, 2)

        # Angle of each projected point
        phi = np.arctan2(That[:, 1], That[:, 0])        # (N,)

        # Robust extremes via percentile (pseudo-min/max)
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)

        # Map angles back to OD-space direction vectors
        v_min = V @ np.array([np.cos(min_phi), np.sin(min_phi)])  # (3,)
        v_max = V @ np.array([np.cos(max_phi), np.sin(max_phi)])  # (3,)

        # Hematoxylin has stronger absorption in red channel (index 0)
        if v_min[0] > v_max[0]:
            HE = np.stack([v_min, v_max], axis=1)
        else:
            HE = np.stack([v_max, v_min], axis=1)

        return HE  # (3, 2)

    def _get_foreground_od(self, I: np.ndarray):
        """
        Convert image to OD and return foreground pixels + full OD array.

        Returns
        -------
        OD     : (H*W, 3) full optical density array
        OD_hat : (N, 3)   foreground-only (background removed)
        """
        I_flat = I.reshape(-1, 3)
        OD     = self._compute_od(I_flat)
        mask   = ~np.any(OD < self.beta, axis=1)
        return OD, OD[mask]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, reference_image) -> "MacenkoNormalizer":
        """
        Fit the normalizer to a reference image.

        Estimates HERef and maxCRef from the reference image's stain
        appearance. After fit(), all subsequent transform() calls will
        map images toward this reference's color space.

        Args:
            reference_image : PIL Image or numpy array (H, W, 3) uint8

        Returns:
            self  (for method chaining)
        """
        ref = self._to_numpy(reference_image)
        OD, OD_hat = self._get_foreground_od(ref)

        if OD_hat.shape[0] < 10:
            raise ValueError(
                "Reference image has too few foreground pixels "
                f"(found {OD_hat.shape[0]}, need ≥ 10). "
                "Check that --reference points to a tissue-rich image."
            )

        # Estimate stain matrix from reference
        self.HERef = self._estimate_he_matrix(OD_hat)   # (3, 2)

        # Solve concentrations for the reference itself
        Y    = OD.T                                       # (3, H*W)
        C, _, _, _ = np.linalg.lstsq(self.HERef, Y, rcond=None)  # (2, H*W)

        # 99th-percentile concentration = reference max
        self.maxCRef = np.percentile(C, 99, axis=1, keepdims=True)  # (2, 1)
        self.maxCRef = np.maximum(self.maxCRef, 1e-6)

        self._fitted = True
        return self

    def transform(self, image, return_he: bool = False):
        """
        Normalize an H&E image to the fitted (or default fixed) reference.

        Args:
            image     : PIL Image (RGB) or numpy array (H, W, 3) uint8
            return_he : If True, also return separate H and E channel images

        Returns:
            If return_he is False : normalized PIL Image (RGB)
            If return_he is True  : (normalized, H_image, E_image) — all PIL
        """
        I = self._to_numpy(image)
        h, w = I.shape[:2]

        OD, OD_hat = self._get_foreground_od(I)

        if OD_hat.shape[0] < 10:
            # Barely any tissue — return original unchanged
            pil = Image.fromarray(I)
            if return_he:
                return pil, pil, pil
            return pil

        # Estimate this image's stain matrix
        HE = self._estimate_he_matrix(OD_hat)            # (3, 2)

        # Solve stain concentrations for every pixel
        Y = OD.T                                          # (3, H*W)
        C, _, _, _ = np.linalg.lstsq(HE, Y, rcond=None) # (2, H*W)

        # Normalize concentrations to reference range
        maxC = np.percentile(C, 99, axis=1, keepdims=True)
        maxC = np.maximum(maxC, 1e-6)
        C_norm = C / maxC * self.maxCRef                  # (2, H*W)

        # Reconstruct using reference stain matrix
        I_norm_flat = self.Io * np.exp(-self.HERef @ C_norm)  # (3, H*W)
        I_norm_flat = np.clip(I_norm_flat, 0, 255).astype(np.uint8)
        I_norm      = I_norm_flat.T.reshape(h, w, 3)

        normalized = Image.fromarray(I_norm, mode="RGB")

        if not return_he:
            return normalized

        # Separate H channel
        H_flat = self.Io * np.exp(-self.HERef[:, 0:1] * C_norm[0:1, :])
        H_flat = np.clip(H_flat, 0, 255).astype(np.uint8)
        H_img  = Image.fromarray(H_flat.T.reshape(h, w, 3), mode="RGB")

        # Separate E channel
        E_flat = self.Io * np.exp(-self.HERef[:, 1:2] * C_norm[1:2, :])
        E_flat = np.clip(E_flat, 0, 255).astype(np.uint8)
        E_img  = Image.fromarray(E_flat.T.reshape(h, w, 3), mode="RGB")

        return normalized, H_img, E_img

    def fit_transform(self, reference_image, image) -> Image.Image:
        """Convenience: fit to reference, then transform image."""
        return self.fit(reference_image).transform(image)

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been called; False if using fixed fallback values."""
        return self._fitted
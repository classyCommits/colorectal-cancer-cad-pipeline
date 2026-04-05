"""
visualize_normalization.py
--------------------------
Generates a before/after comparison grid to visually verify that
Macenko stain normalization is working correctly.

TWO modes
---------
Mode A — Paired folders (recommended after a batch normalize run):
    Pass --raw-folder and --norm-folder.
    The script matches files by filename across the two folders.
    "Original" column   <- data/raw/colon_image_sets/colon_aca/
    "Normalized" column <- data/normalized_test/colon_image_sets/colon_aca/

    python visualize_normalization.py \
        --reference   data/sample/reference.png \
        --raw-folder  data/raw/colon_image_sets/colon_aca/ \
        --norm-folder data/normalized_test/colon_image_sets/colon_aca/ \
        --n 6 \
        --output outputs/comparison.png \
        --histograms

Mode B — Live normalization (no prior batch run needed):
    Pass --folder pointing to the raw images.
    The script normalizes on-the-fly and shows before/after.

    python visualize_normalization.py \
        --reference data/sample/reference.png \
        --folder    data/raw/colon_image_sets/colon_aca/ \
        --n 6 \
        --output outputs/comparison.png
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from macenko import MacenkoNormalizer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _build_normalizer(reference_path: Path) -> MacenkoNormalizer:
    """Fit normalizer to reference; fall back to fixed vectors on failure."""
    ref = load_rgb(reference_path)
    normalizer = MacenkoNormalizer()
    try:
        normalizer.fit(ref)
        print(f"Normalizer fitted to: {reference_path.name}")
    except ValueError as e:
        print(f"Warning: fit() failed ({e}). Using fixed MATLAB reference vectors.")
    return normalizer


def collect_random(folder: Path, n: int, ext: str) -> list:
    exts = [".jpg", ".jpeg"] if ext.lower() in (".jpg", ".jpeg") else [ext]
    all_images = list(set(p for e in exts for p in folder.rglob(f"*{e}")))
    if not all_images:
        raise FileNotFoundError(f"No {ext} images found in {folder}")
    random.shuffle(all_images)
    return all_images[:n]


def pair_raw_and_normalized(
    raw_folder: Path,
    norm_folder: Path,
    n: int,
    ext: str,
) -> list:
    """
    Match raw images to their normalized counterparts by mirroring
    the subfolder structure.

      raw_folder/sub/file.png  <->  norm_folder/sub/file.png

    Returns list of (raw_path, norm_path) tuples for files found in both.
    """
    exts = [".jpg", ".jpeg"] if ext.lower() in (".jpg", ".jpeg") else [ext]
    raw_images = sorted(set(p for e in exts for p in raw_folder.rglob(f"*{e}")))
    if not raw_images:
        raise FileNotFoundError(f"No {ext} images found in {raw_folder}")

    pairs   = []
    missing = []

    for raw_path in raw_images:
        relative  = raw_path.relative_to(raw_folder)
        norm_path = norm_folder / relative

        if norm_path.exists():
            pairs.append((raw_path, norm_path))
        else:
            missing.append(raw_path.name)

    if missing:
        print(f"Warning: {len(missing)} raw image(s) had no normalized counterpart. "
              f"First few: {missing[:5]}")
        print("  Have you run normalize_dataset.py yet?")

    if not pairs:
        raise FileNotFoundError(
            f"No matched pairs found between:\n"
            f"  raw-folder : {raw_folder}\n"
            f"  norm-folder: {norm_folder}\n"
            f"Run normalize_dataset.py first, or use --folder for live normalization."
        )

    print(f"Matched {len(pairs)} raw/normalized pairs.")
    random.shuffle(pairs)
    return pairs[:n]


# ------------------------------------------------------------------
# Before / After comparison grid
# ------------------------------------------------------------------

def make_comparison_grid(
    reference_path: Path,
    pairs: list,
    output_path: Path,
    live_normalizer=None,
):
    """
    3-column grid: Original | Reference | Normalized

    pairs is a list of (raw_path, norm_path_or_None).
      - If norm_path is a Path  -> Mode A: load the saved normalized file.
      - If norm_path is None    -> Mode B: normalize on-the-fly with live_normalizer.
    """
    reference = load_rgb(reference_path)
    n = len(pairs)

    if n == 0:
        print("No images to visualize.")
        return

    fig, axes = plt.subplots(
        nrows=n, ncols=3,
        figsize=(12, 4 * n),
        dpi=100,
        squeeze=False,
    )

    col_titles = ["Original", "Reference", "Normalized"]
    col_colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for col_idx, (title, color) in enumerate(zip(col_titles, col_colors)):
        axes[0, col_idx].set_title(
            title, fontsize=14, fontweight="bold", color=color, pad=10
        )

    for row_idx, (raw_path, norm_path) in enumerate(pairs):
        original    = load_rgb(raw_path)
        norm_failed = False
        err_msg     = ""

        if norm_path is not None:
            # Mode A: load the already-saved normalized file
            try:
                normalized = load_rgb(norm_path)
            except Exception as e:
                normalized  = np.zeros_like(original)
                norm_failed = True
                err_msg     = str(e)
        else:
            # Mode B: normalize on-the-fly
            try:
                normalized = np.array(
                    live_normalizer.transform(Image.fromarray(original))
                )
            except Exception as e:
                normalized  = np.zeros_like(original)
                norm_failed = True
                err_msg     = str(e)

        axes[row_idx, 0].imshow(original)
        axes[row_idx, 0].set_ylabel(
            raw_path.stem[:22],
            fontsize=8, rotation=0, labelpad=65, va="center",
        )
        axes[row_idx, 1].imshow(reference)
        axes[row_idx, 2].imshow(normalized)

        if norm_failed:
            axes[row_idx, 2].text(
                0.5, 0.5, f"Failed:\n{err_msg}",
                transform=axes[row_idx, 2].transAxes,
                ha="center", va="center", fontsize=8, color="red",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
            )

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Macenko Stain Normalization — Before / After",
        fontsize=16, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved comparison grid -> {output_path}")


# ------------------------------------------------------------------
# Channel histogram comparison
# ------------------------------------------------------------------

def plot_channel_histograms(
    reference_path: Path,
    pairs: list,
    output_path: Path,
    live_normalizer=None,
):
    """
    RGB channel histograms (reference / original / normalized) per image.
    Shows how well normalization aligns channel distributions to the reference.
    """
    reference      = load_rgb(reference_path)
    channel_names  = ["Red", "Green", "Blue"]
    channel_colors = ["#e74c3c", "#2ecc71", "#3498db"]

    n = len(pairs)
    if n == 0:
        print("No images for histogram plot.")
        return

    fig, axes = plt.subplots(
        nrows=n, ncols=3,
        figsize=(15, 4 * n),
        dpi=100,
        squeeze=False,
    )

    for col_idx, (ch_name, ch_color) in enumerate(zip(channel_names, channel_colors)):
        axes[0, col_idx].set_title(
            f"{ch_name} Channel", fontsize=12, fontweight="bold", color=ch_color
        )

    hist_kw = dict(bins=64, range=(0, 255), density=True)

    for row_idx, (raw_path, norm_path) in enumerate(pairs):
        original = load_rgb(raw_path)

        if norm_path is not None:
            try:
                normalized = load_rgb(norm_path)
            except Exception:
                normalized = original
        else:
            try:
                normalized = np.array(
                    live_normalizer.transform(Image.fromarray(original))
                )
            except Exception:
                normalized = original

        for ch_idx in range(3):
            ax    = axes[row_idx, ch_idx]
            color = channel_colors[ch_idx]

            ax.hist(reference[:, :, ch_idx].ravel(),  color="#95a5a6", alpha=0.5,
                    label="Reference", **hist_kw)
            ax.hist(original[:, :, ch_idx].ravel(),   color=color,     alpha=0.5,
                    label="Original",  **hist_kw)
            ax.hist(normalized[:, :, ch_idx].ravel(), color=color,     alpha=0.9,
                    histtype="step", linewidth=2, label="Normalized", **hist_kw)

            ax.set_xlim(0, 255)
            ax.set_ylabel(raw_path.stem[:15], fontsize=7)

            if row_idx == 0 and ch_idx == 2:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Channel Histograms: Original vs Normalized vs Reference",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved histogram comparison -> {output_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize Macenko stain normalization results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  Mode A - compare raw vs already-normalized files (recommended):
    python visualize_normalization.py \\
        --reference   data/sample/reference.png \\
        --raw-folder  data/raw/colon_image_sets/colon_aca/ \\
        --norm-folder data/normalized_test/colon_image_sets/colon_aca/ \\
        --n 6 --output outputs/comparison.png --histograms

  Mode B - normalize on-the-fly (no prior batch run needed):
    python visualize_normalization.py \\
        --reference data/sample/reference.png \\
        --folder    data/raw/colon_image_sets/colon_aca/ \\
        --n 6 --output outputs/comparison.png
        """
    )
    p.add_argument("--reference",  required=True, help="Reference H&E image path")
    p.add_argument("--output",     default="outputs/comparison.png",
                   help="Output path for comparison grid PNG")
    p.add_argument("--n",          type=int, default=6,
                   help="Number of images to sample (default: 6)")
    p.add_argument("--ext",        default=".jpeg",
                   help="Image file extension (default: .jpeg). Both .jpg and .jpeg are accepted.")
    p.add_argument("--histograms", action="store_true",
                   help="Also generate channel histogram comparison plot")
    p.add_argument("--seed",       type=int, default=None,
                   help="Random seed for reproducible sampling")

    # Mutually exclusive: Mode A (--raw-folder) vs Mode B (--folder)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--raw-folder",  dest="raw_folder",
                      help="[Mode A] Folder of original raw images")
    mode.add_argument("--folder",
                      help="[Mode B] Raw folder — normalize on-the-fly")

    # Only used with Mode A
    p.add_argument("--norm-folder", dest="norm_folder", default=None,
                   help="[Mode A] Folder of already-normalized images (required with --raw-folder)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    ref_path    = Path(args.reference)
    output_path = Path(args.output)

    if args.raw_folder:
        # ---- Mode A: load raw + load normalized from disk ----
        if not args.norm_folder:
            raise SystemExit(
                "Error: --norm-folder is required when using --raw-folder.\n"
                "Example:\n"
                "  --raw-folder  data/raw/colon_image_sets/colon_aca/\n"
                "  --norm-folder data/normalized_test/colon_image_sets/colon_aca/"
            )
        pairs           = pair_raw_and_normalized(
            raw_folder  = Path(args.raw_folder),
            norm_folder = Path(args.norm_folder),
            n           = args.n,
            ext         = args.ext,
        )
        live_normalizer = None

    else:
        # ---- Mode B: live normalization from raw folder ----
        raw_paths       = collect_random(Path(args.folder), args.n, args.ext)
        pairs           = [(p, None) for p in raw_paths]
        live_normalizer = _build_normalizer(ref_path)

    make_comparison_grid(ref_path, pairs, output_path, live_normalizer)

    if args.histograms:
        hist_path = output_path.parent / (output_path.stem + "_histograms.png")
        plot_channel_histograms(ref_path, pairs, hist_path, live_normalizer)
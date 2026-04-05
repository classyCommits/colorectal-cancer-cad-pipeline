"""
visualize_patches.py
--------------------
Visualizes extracted patches in a grid view to verify extraction quality.

Usage:
    # Visualize patches from a folder
    python visualize_patches.py \
        --folder data/patches/colon_aca \
        --n 25 \
        --output outputs/patch_grid.png

    # Show patch extraction from a single source image
    python visualize_patches.py \
        --source data/raw/colon_image_sets/colon_aca/colonca1.jpeg \
        --output outputs/patch_extraction_demo.png
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw

from extract_patches import PatchExtractor
from filter_background import BackgroundFilter


# ------------------------------------------------------------------
# Grid visualization
# ------------------------------------------------------------------

def visualize_patch_grid(
    patches     : list,
    title       : str  = "Extracted Patches",
    cols        : int  = 5,
    output_path : Path = None,
    class_name  : str  = "",
):
    """
    Display patches in a grid layout.

    Args:
        patches    : List of numpy arrays or PIL Images
        title      : Plot title
        cols       : Number of columns in grid
        output_path: Where to save the figure
        class_name : Class label shown in subtitle
    """
    n    = len(patches)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=100)
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < n:
            patch = patches[i]
            if isinstance(patch, Image.Image):
                patch = np.array(patch)
            ax.imshow(patch)
            ax.set_title(f"#{i+1}", fontsize=7)
        ax.axis("off")

    fig.suptitle(
        title + (f"\nClass: {class_name}" if class_name else ""),
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=120)
        print(f"Saved patch grid → {output_path}")

    plt.close(fig)


# ------------------------------------------------------------------
# Single image extraction demo
# ------------------------------------------------------------------

def visualize_extraction_demo(
    source_path : Path,
    output_path : Path,
    patch_size  : int  = 224,
    stride      : int  = 224,
    max_patches : int  = 20,
):
    """
    Shows a single source image with patch extraction overlay,
    alongside the extracted tissue patches.

    Args:
        source_path : Path to source image
        output_path : Where to save the visualization
        patch_size  : Patch size in pixels
        stride      : Extraction stride
        max_patches : Maximum patches to show in grid
    """
    extractor = PatchExtractor(patch_size=patch_size, stride=stride)
    bg_filter = BackgroundFilter()

    source_image = Image.open(source_path).convert("RGB")
    img_array    = np.array(source_image)
    h, w         = img_array.shape[:2]

    # Extract patches with coordinates
    results = extractor.extract_from_image_with_coords(
        source_path, filter_background=False
    )

    # Separate tissue vs background
    tissue_results     = [(p, x, y) for p, x, y in results if bg_filter.is_tissue(p)]
    background_results = [(p, x, y) for p, x, y in results if not bg_filter.is_tissue(p)]

    # Draw overlay on source image
    overlay = source_image.copy().convert("RGBA")
    draw    = ImageDraw.Draw(overlay, "RGBA")

    for patch, x, y in results:
        is_tissue = bg_filter.is_tissue(patch)
        color = (0, 200, 0, 60) if is_tissue else (255, 0, 0, 40)
        draw.rectangle([x, y, x+patch_size, y+patch_size], fill=color, outline=color[:3]+(180,), width=1)

    overlay_rgb = Image.alpha_composite(
        source_image.convert("RGBA"), overlay
    ).convert("RGB")

    # Layout: source + overlay | patch grid
    tissue_patches = [p for p, x, y in tissue_results[:max_patches]]
    n_show = min(len(tissue_patches), max_patches)
    cols   = 5
    rows   = max(1, (n_show + cols - 1) // cols)

    fig = plt.figure(figsize=(18, max(6, rows * 2.8)), dpi=100)

    # Source image
    ax_src = fig.add_subplot(1, 3, 1)
    ax_src.imshow(img_array)
    ax_src.set_title("Source Image", fontweight="bold", fontsize=12)
    ax_src.axis("off")

    # Overlay
    ax_ov = fig.add_subplot(1, 3, 2)
    ax_ov.imshow(overlay_rgb)
    ax_ov.set_title(
        f"Patch Grid Overlay\n"
        f"✅ Tissue: {len(tissue_results)}  ❌ Background: {len(background_results)}",
        fontweight="bold", fontsize=10
    )
    green_patch = mpatches.Patch(color=(0, 0.78, 0, 0.6), label="Tissue")
    red_patch   = mpatches.Patch(color=(1, 0, 0, 0.4),    label="Background")
    ax_ov.legend(handles=[green_patch, red_patch], loc="upper right", fontsize=8)
    ax_ov.axis("off")

    # Patch grid
    ax_grid = fig.add_subplot(1, 3, 3)
    ax_grid.axis("off")

    if tissue_patches:
        grid_cols = min(cols, n_show)
        grid_rows = (n_show + grid_cols - 1) // grid_cols
        grid_h    = grid_rows * patch_size
        grid_w    = grid_cols * patch_size
        grid_img  = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

        for i, patch in enumerate(tissue_patches[:n_show]):
            r = i // grid_cols
            c = i % grid_cols
            grid_img[r*patch_size:(r+1)*patch_size,
                     c*patch_size:(c+1)*patch_size] = patch

        ax_grid.imshow(grid_img)
        ax_grid.set_title(
            f"Extracted Tissue Patches (first {n_show})",
            fontweight="bold", fontsize=10
        )

    fig.suptitle(
        f"Patch Extraction Demo — {Path(source_path).name}\n"
        f"Patch size: {patch_size}×{patch_size}px  |  Stride: {stride}px",
        fontsize=13, fontweight="bold"
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"Saved extraction demo → {output_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualize patch extraction results.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--folder", help="Folder of already-extracted patches to show as grid")
    mode.add_argument("--source", help="Single source image to demonstrate extraction on")

    p.add_argument("--output", required=True, help="Output path for visualization image")
    p.add_argument("--n",      type=int, default=25,  help="Number of patches to show (grid mode)")
    p.add_argument("--cols",   type=int, default=5,   help="Grid columns (default: 5)")
    p.add_argument("--size",   type=int, default=224, help="Patch size (demo mode)")
    p.add_argument("--stride", type=int, default=224, help="Stride (demo mode)")
    p.add_argument("--ext",    default=".png",        help="Extension for grid mode (default: .png)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.source:
        visualize_extraction_demo(
            source_path=Path(args.source),
            output_path=Path(args.output),
            patch_size=args.size,
            stride=args.stride,
        )
    else:
        folder = Path(args.folder)
        images = list(folder.rglob(f"*{args.ext}"))
        if not images:
            print(f"No {args.ext} images found in {folder}")
            exit(1)

        sample = random.sample(images, min(args.n, len(images)))
        patches = [np.array(Image.open(p).convert("RGB")) for p in sample]

        visualize_patch_grid(
            patches=patches,
            title="Extracted Tissue Patches",
            cols=args.cols,
            output_path=Path(args.output),
            class_name=folder.name,
        )

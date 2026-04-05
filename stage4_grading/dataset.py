"""
dataset.py
----------
Dataset classes for cancer grading.

Supported datasets:
    1. TCGA-COAD (recommended)
       - Source: GDC Data Portal (https://portal.gdc.cancer.gov/)
       - Contains WSI slides with pathologist grade annotations
       - Grades extracted from clinical metadata (XML/JSON)
       - Requires Stage 2 patch extraction first

    2. Kather et al. Graded CRC
       - Source: https://zenodo.org/record/53169
       - 5000 patches, 8 tissue classes
       - Not directly graded but can be used for multi-class training

    3. Custom folder structure (most flexible):
       data/graded/
           grade1/   ← well-differentiated patches
           grade2/   ← moderately-differentiated patches
           grade3/   ← poorly-differentiated patches

Usage:
    # Option 1 — ImageFolder (simplest, use if you have grade folders)
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder('data/graded/', transform=transform)

    # Option 2 — GradingDataset (more control)
    dataset = GradingDataset('data/graded/', transform=transform)
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ------------------------------------------------------------------
# Grade definitions
# ------------------------------------------------------------------

GRADE_TO_IDX = {
    'grade1': 0,
    'grade2': 1,
    'grade3': 2,
    'well_differentiated'       : 0,
    'moderately_differentiated' : 1,
    'poorly_differentiated'     : 2,
    'low'    : 0,
    'medium' : 1,
    'high'   : 2,
}

IDX_TO_GRADE = {
    0: 'Grade 1 (Well-differentiated)',
    1: 'Grade 2 (Moderately-differentiated)',
    2: 'Grade 3 (Poorly-differentiated)',
}


# ------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------

class GradingDataset(Dataset):
    """
    Dataset for cancer grading from a folder of grade subfolders.

    Expected folder structure:
        root/
            grade1/  or  well_differentiated/  or  low/
                patch_001.png
                patch_002.png
            grade2/  or  moderately_differentiated/  or  medium/
                ...
            grade3/  or  poorly_differentiated/  or  high/
                ...

    Parameters
    ----------
    root        : Path to root folder containing grade subfolders
    transform   : torchvision transforms
    extensions  : Accepted image file extensions
    """

    def __init__(
        self,
        root       : str,
        transform  : Optional[Callable] = None,
        extensions : Tuple[str, ...]    = ('.png', '.jpg', '.jpeg', '.tif', '.tiff'),
    ):
        self.root      = Path(root)
        self.transform = transform
        self.samples   : List[Tuple[Path, int]] = []
        self.classes   : List[str] = []

        self._load_samples(extensions)

    def _load_samples(self, extensions):
        """Scan root folder and build sample list."""
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        grade_folders = sorted([
            d for d in self.root.iterdir() if d.is_dir()
        ])

        if not grade_folders:
            raise ValueError(f"No subfolders found in {self.root}")

        for folder in grade_folders:
            folder_name = folder.name.lower()

            # Map folder name to grade index
            grade_idx = GRADE_TO_IDX.get(folder_name)
            if grade_idx is None:
                print(f"Warning: Skipping unknown folder '{folder.name}' "
                      f"(expected: grade1/grade2/grade3)")
                continue

            self.classes.append(folder.name)

            # Collect all image files
            for ext in extensions:
                for img_path in folder.glob(f'*{ext}'):
                    self.samples.append((img_path, grade_idx))

        if not self.samples:
            raise ValueError(
                f"No images found in {self.root}. "
                f"Expected subfolders: grade1/, grade2/, grade3/"
            )

        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} grades")

        # Print class distribution
        from collections import Counter
        dist = Counter(label for _, label in self.samples)
        for grade_idx, count in sorted(dist.items()):
            print(f"  {IDX_TO_GRADE[grade_idx]:40s}: {count:5d} patches")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, grade_idx = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, grade_idx

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        Returns weights inversely proportional to class frequency.
        Used with WeightedRandomSampler or CrossEntropyLoss weight param.
        """
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        total  = len(self.samples)
        weights = torch.zeros(3)
        for grade_idx, count in counts.items():
            weights[grade_idx] = total / (len(counts) * count)
        return weights

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Returns a WeightedRandomSampler for balanced batch sampling.
        Use this when grades are imbalanced.
        """
        class_weights = self.get_class_weights()
        sample_weights = [
            class_weights[label].item()
            for _, label in self.samples
        ]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


# ------------------------------------------------------------------
# Standard transforms for grading
# ------------------------------------------------------------------

def get_grading_transforms(image_size: int = 224) -> dict:
    """
    Returns train and val transforms for grading task.
    More aggressive augmentation than Stage 3 because
    grading is a harder task requiring more generalization.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {'train': train_transform, 'val': val_transform}


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------

def get_grading_dataloaders(
    root        : str,
    batch_size  : int   = 32,
    image_size  : int   = 224,
    val_split   : float = 0.15,
    test_split  : float = 0.15,
    num_workers : int   = 4,
    balanced    : bool  = True,
) -> dict:
    """
    Creates train/val/test DataLoaders from a graded dataset folder.

    Args:
        root       : Path to folder with grade1/, grade2/, grade3/ subfolders
        batch_size : Batch size
        image_size : Image resize target
        val_split  : Fraction for validation
        test_split : Fraction for test
        num_workers: DataLoader workers
        balanced   : Use WeightedRandomSampler for balanced training

    Returns:
        dict with 'train', 'val', 'test' DataLoaders and 'class_weights'
    """
    from torch.utils.data import random_split

    transforms_dict = get_grading_transforms(image_size)

    full_dataset = GradingDataset(root, transform=transforms_dict['train'])
    class_weights = full_dataset.get_class_weights()

    n_total = len(full_dataset)
    n_val   = int(val_split * n_total)
    n_test  = int(test_split * n_total)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    val_ds.dataset.transform  = transforms_dict['val']
    test_ds.dataset.transform = transforms_dict['val']

    # Balanced sampler for training
    sampler = full_dataset.get_weighted_sampler() if balanced else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    return {
        'train'         : train_loader,
        'val'           : val_loader,
        'test'          : test_loader,
        'class_weights' : class_weights,
        'n_train'       : n_train,
        'n_val'         : n_val,
        'n_test'        : n_test,
    }

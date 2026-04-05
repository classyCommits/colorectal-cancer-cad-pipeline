"""
evaluate.py
-----------
Evaluation script for trained cancer grading model.

Key metric for grading: Quadratic Weighted Kappa
    - Standard metric for ordinal classification tasks like cancer grading
    - Penalizes predictions that are far from true grade more than nearby ones
    - Grade 1 vs Grade 3 error is worse than Grade 1 vs Grade 2
    - Score > 0.8 is considered strong agreement in medical grading

Usage:
    python evaluate.py \
        --checkpoint checkpoints/grading_model_best.pth \
        --data       data/graded \
        --output     outputs/grading_evaluation.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from dataset import GradingDataset, get_grading_transforms, IDX_TO_GRADE
from model import CancerGradingModel
from torch.utils.data import DataLoader


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = outputs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


def plot_evaluation(preds, labels, probs, output_path):
    grade_names = [IDX_TO_GRADE[i] for i in range(3)]
    short_names = ['Grade 1\n(Well)', 'Grade 2\n(Moderate)', 'Grade 3\n(Poor)']

    cm     = confusion_matrix(labels, preds)
    f1     = f1_score(labels, preds, average='weighted')
    kappa  = cohen_kappa_score(labels, preds, weights='quadratic')
    acc    = (preds == labels).mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=short_names,
        yticklabels=short_names,
        ax=axes[0],
    )
    axes[0].set_xlabel('Predicted Grade', fontsize=11)
    axes[0].set_ylabel('True Grade',      fontsize=11)
    axes[0].set_title(
        f'Confusion Matrix\n'
        f'Accuracy: {acc*100:.2f}%  |  '
        f'F1: {f1:.4f}  |  '
        f'Kappa: {kappa:.4f}',
        fontsize=11, fontweight='bold'
    )

    # Per-class confidence distribution
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for i in range(3):
        mask = labels == i
        if mask.sum() > 0:
            axes[1].hist(
                probs[mask, i],
                bins=20, alpha=0.6,
                color=colors[i],
                label=f'Grade {i+1} (n={mask.sum()})',
            )
    axes[1].set_xlabel('Predicted Confidence', fontsize=11)
    axes[1].set_ylabel('Count',                fontsize=11)
    axes[1].set_title('Confidence Distribution by True Grade', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        'Cancer Grading Model — Evaluation Results',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Evaluation plot saved → {output_path}')

    print('\n' + '='*60)
    print('GRADING EVALUATION RESULTS')
    print('='*60)
    print(f'Accuracy         : {acc*100:.2f}%')
    print(f'Weighted F1      : {f1:.4f}')
    print(f'Quadratic Kappa  : {kappa:.4f}')
    print('\nClassification Report:')
    print(classification_report(labels, preds, target_names=grade_names))


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate cancer grading model.')
    p.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    p.add_argument('--data',       required=True, help='Path to graded dataset folder')
    p.add_argument('--output',     default='outputs/grading_evaluation.png')
    p.add_argument('--batch',      type=int, default=32)
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CancerGradingModel(num_grades=3, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Load dataset
    transforms = get_grading_transforms()
    dataset    = GradingDataset(args.data, transform=transforms['val'])
    loader     = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=2)

    # Evaluate
    preds, labels, probs = run_evaluation(model, loader, device)
    plot_evaluation(preds, labels, probs, args.output)

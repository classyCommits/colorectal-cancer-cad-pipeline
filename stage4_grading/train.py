"""
train.py
--------
Training script for cancer grading model.

NOTE: This script requires a graded colorectal cancer dataset.
      See dataset.py for supported datasets and folder structure.

      Until a graded dataset is available, this script demonstrates
      the complete training pipeline architecture. The model, loss
      function, optimizer, and evaluation logic are all production-ready.

Required folder structure:
    data/graded/
        grade1/   ← well-differentiated patches (low severity)
        grade2/   ← moderately-differentiated patches
        grade3/   ← poorly-differentiated patches (high severity)

Usage (when dataset is available):
    python train.py \
        --data   data/graded \
        --output stage4_grading/checkpoints \
        --epochs 20 \
        --batch  32 \
        --lr     1e-4

Recommended dataset:
    TCGA-COAD from GDC Data Portal (https://portal.gdc.cancer.gov/)
    After downloading WSIs, use Stage 2 patch extractor to generate patches,
    then organize by pathologist-annotated grade into grade1/grade2/grade3.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
)

from dataset import get_grading_dataloaders, IDX_TO_GRADE
from model import CancerGradingModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Training functions
# ------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct      += (outputs.argmax(1) == labels).sum().item()
        total        += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds    = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------ Data ------
    log.info(f'Loading graded dataset from: {args.data}')
    dataloaders = get_grading_dataloaders(
        root=args.data,
        batch_size=args.batch,
        val_split=0.15,
        test_split=0.15,
        balanced=True,
    )
    log.info(f"Train: {dataloaders['n_train']}  "
             f"Val: {dataloaders['n_val']}  "
             f"Test: {dataloaders['n_test']}")

    # ------ Model ------
    model = CancerGradingModel(num_grades=3, pretrained=True).to(device)
    log.info(f"Parameters: {model.parameter_count()}")

    # ------ Loss — weighted for class imbalance ------
    class_weights = dataloaders['class_weights'].to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ------ Optimizer ------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )

    # ------ Training loop ------
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss'  : [], 'val_acc'  : [],
    }

    best_val_acc    = 0.0
    patience_count  = 0
    best_model_path = output_dir / 'grading_model_best.pth'

    log.info(f'Starting training for {args.epochs} epochs...')
    log.info('=' * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Unfreeze backbone after epoch 5
        if epoch == 6:
            model.unfreeze_backbone()
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr / 10
            log.info('Backbone unfrozen, LR reduced for fine-tuning')

        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, dataloaders['val'], criterion, device
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        log.info(
            f'Epoch {epoch:3d}/{args.epochs}  '
            f'Train Loss:{train_loss:.4f} Acc:{train_acc:.4f}  '
            f'Val Loss:{val_loss:.4f} Acc:{val_acc:.4f}  '
            f'Time:{time.time()-t0:.1f}s'
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch'           : epoch,
                'model_state_dict': model.state_dict(),
                'val_acc'         : val_acc,
                'val_loss'        : val_loss,
                'args'            : vars(args),
            }, best_model_path)
            log.info(f'  --> Best model saved (val_acc={val_acc:.4f})')
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info(f'Early stopping at epoch {epoch}')
                break

    # ------ Final evaluation on test set ------
    log.info('=' * 60)
    log.info('Loading best model for test evaluation...')

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, dataloaders['test'], criterion, device
    )

    f1     = f1_score(test_labels, test_preds, average='weighted')
    kappa  = cohen_kappa_score(test_labels, test_preds,
                               weights='quadratic')  # standard for grading
    cm     = confusion_matrix(test_labels, test_preds)

    log.info(f'Test Accuracy    : {test_acc*100:.2f}%')
    log.info(f'Weighted F1      : {f1:.4f}')
    log.info(f'Quadratic Kappa  : {kappa:.4f}')  # key metric for grading
    log.info('\n' + classification_report(
        test_labels, test_preds,
        target_names=[IDX_TO_GRADE[i] for i in range(3)]
    ))

    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'test_acc'    : test_acc,
        'f1_weighted' : f1,
        'kappa'       : kappa,
        'history'     : history,
        'confusion_matrix': cm.tolist(),
    }
    with open(output_dir / 'grading_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    log.info(f'Results saved to {output_dir}')
    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Train cancer grading model on graded H&E patches.'
    )
    p.add_argument('--data',    required=True,
                   help='Path to graded dataset (with grade1/grade2/grade3 subfolders)')
    p.add_argument('--output',  default='checkpoints',
                   help='Output directory for weights and results')
    p.add_argument('--epochs',  type=int,   default=20)
    p.add_argument('--batch',   type=int,   default=32)
    p.add_argument('--lr',      type=float, default=1e-4)
    p.add_argument('--patience',type=int,   default=5)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)

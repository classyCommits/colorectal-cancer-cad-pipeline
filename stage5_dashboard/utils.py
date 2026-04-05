"""
utils.py
--------
Utility functions for the Stage 5 dashboard.

Covers:
    - Model loading and caching
    - Image preprocessing
    - Prediction with confidence
    - Result formatting
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add stage3 to path for model import
sys.path.insert(0, str(Path(__file__).parent.parent / 'stage3_classification'))


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CLASS_NAMES = ['colon_aca', 'colon_n']
CLASS_LABELS = {
    'colon_aca': 'Adenocarcinoma (Cancer)',
    'colon_n'  : 'Benign Tissue (Normal)',
}
CLASS_COLORS = {
    'colon_aca': '#e74c3c',   # red for cancer
    'colon_n'  : '#2ecc71',   # green for normal
}
CLASS_ICONS = {
    'colon_aca': '🔴',
    'colon_n'  : '🟢',
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def get_transform() -> transforms.Compose:
    """Standard preprocessing transform for model inference."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL Image for model inference.

    Args:
        image: PIL Image (any mode, any size)

    Returns:
        tensor: (1, 3, 224, 224) float tensor
    """
    transform = get_transform()
    image_rgb = image.convert('RGB')
    tensor    = transform(image_rgb).unsqueeze(0)  # add batch dim
    return tensor


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: (3, H, W) normalized tensor

    Returns:
        numpy array (H, W, 3) uint8
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = 'cpu'):
    """
    Load trained ColonClassifier from checkpoint.

    Args:
        checkpoint_path: Path to .pth file
        device         : 'cpu' or 'cuda'

    Returns:
        model: loaded and eval-mode ColonClassifier
    """
    try:
        from model import ColonClassifier
    except ImportError:
        # Fallback: define inline if stage3 not in path
        import timm
        import torch.nn as nn

        class ColonClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone   = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
                n_features      = self.backbone.num_features
                self.classifier = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Linear(256, 2),
                )
            def forward(self, x):
                return self.classifier(self.backbone(x))

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please download efficientnet_b0_colon.pth from Google Drive "
            f"and place it in stage3_classification/checkpoints/"
        )

    device    = torch.device(device)
    model     = ColonClassifier()
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

@torch.no_grad()
def predict(
    model  : torch.nn.Module,
    tensor : torch.Tensor,
    device : str = 'cpu',
) -> dict:
    """
    Run inference and return structured prediction result.

    Args:
        model : loaded ColonClassifier
        tensor: (1, 3, 224, 224) preprocessed tensor
        device: device string

    Returns:
        dict with class_idx, class_name, label, confidence, probabilities
    """
    tensor  = tensor.to(device)
    logits  = model(tensor)                          # (1, 2)
    probs   = torch.softmax(logits, dim=1)[0]        # (2,)
    cls_idx = int(probs.argmax().item())
    cls_name = CLASS_NAMES[cls_idx]

    return {
        'class_idx'    : cls_idx,
        'class_name'   : cls_name,
        'label'        : CLASS_LABELS[cls_name],
        'color'        : CLASS_COLORS[cls_name],
        'icon'         : CLASS_ICONS[cls_name],
        'confidence'   : float(probs[cls_idx].item()),
        'probabilities': {
            CLASS_NAMES[i]: float(probs[i].item())
            for i in range(len(CLASS_NAMES))
        },
        'is_cancer'    : cls_name == 'colon_aca',
    }


# ------------------------------------------------------------------
# Image validation
# ------------------------------------------------------------------

def validate_image(image: Image.Image) -> tuple:
    """
    Validate uploaded image for histopathology analysis.

    Returns:
        (is_valid: bool, message: str)
    """
    if image is None:
        return False, "No image provided."

    w, h = image.size

    if w < 32 or h < 32:
        return False, f"Image too small ({w}×{h}px). Minimum 32×32px required."

    if w > 10000 or h > 10000:
        return False, f"Image too large ({w}×{h}px). Maximum 10000×10000px."

    if image.mode not in ('RGB', 'RGBA', 'L', 'P'):
        return False, f"Unsupported image mode: {image.mode}"

    return True, "Image valid."


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string."""
    return f"{confidence * 100:.1f}%"


def get_risk_level(confidence: float, is_cancer: bool) -> str:
    """
    Determine clinical risk level based on prediction and confidence.

    Returns:
        Risk level string: 'High', 'Moderate', 'Low', 'Very Low'
    """
    if is_cancer:
        if confidence >= 0.90:
            return 'High'
        elif confidence >= 0.70:
            return 'Moderate'
        else:
            return 'Low'
    else:
        if confidence >= 0.90:
            return 'Very Low'
        elif confidence >= 0.70:
            return 'Low'
        else:
            return 'Moderate'

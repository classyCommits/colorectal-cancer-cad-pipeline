"""
model.py
--------
EfficientNet-B0 classifier for binary colon cancer detection.

Classes:
    0 — colon_aca (adenocarcinoma / cancer)
    1 — colon_n   (benign / normal)
"""

import timm
import torch
import torch.nn as nn


class ColonClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for binary colon cancer classification.

    Architecture:
        EfficientNet-B0 backbone (pretrained ImageNet)
        → GlobalAvgPool (built into backbone)
        → Linear(1280 → 256) → ReLU → Dropout(0.3)
        → Linear(256 → 2)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
        )

        n_features = self.backbone.num_features  # 1280

        self.classifier = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

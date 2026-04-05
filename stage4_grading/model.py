"""
model.py
--------
EfficientNet-B0 model for colorectal cancer grading.

Grading Task:
    Given an H&E histopathology patch, predict the cancer grade:
        Grade 1 — Well-differentiated   (low severity)
        Grade 2 — Moderately-differentiated (medium severity)
        Grade 3 — Poorly-differentiated  (high severity)

Architecture:
    Same EfficientNet-B0 backbone as Stage 3 classifier,
    but with a 3-class output head instead of 2-class.

    EfficientNet-B0 → GlobalAvgPool → Linear(1280→256)
                    → ReLU → Dropout(0.4) → Linear(256→3)

Note:
    This model requires a graded colorectal cancer dataset with
    pathologist-annotated severity grades. See dataset.py for
    supported datasets and how to prepare them.
"""

import torch
import torch.nn as nn
import timm


class CancerGradingModel(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 3-class cancer grading.

    Parameters
    ----------
    num_grades : int
        Number of grade classes (default 3: Grade1, Grade2, Grade3)
    pretrained : bool
        Use ImageNet pretrained weights (default True)
    dropout    : float
        Dropout rate in classifier head (default 0.4)
    """

    # Grade labels
    GRADE_NAMES = {
        0: 'Grade 1 (Well-differentiated)',
        1: 'Grade 2 (Moderately-differentiated)',
        2: 'Grade 3 (Poorly-differentiated)',
    }

    GRADE_SEVERITY = {
        0: 'Low',
        1: 'Medium',
        2: 'High',
    }

    def __init__(
        self,
        num_grades : int   = 3,
        pretrained : bool  = True,
        dropout    : float = 0.4,
    ):
        super().__init__()

        self.num_grades = num_grades

        # EfficientNet-B0 backbone — same as Stage 3
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,       # Remove original classifier
        )

        n_features = self.backbone.num_features  # 1280

        # Grading head — slightly more dropout than Stage 3
        # because grading is a harder task than binary detection
        self.grading_head = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_grades),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print('Backbone frozen.')

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print('Backbone unfrozen.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input tensor

        Returns:
            logits: (B, num_grades) raw class scores
        """
        features = self.backbone(x)          # (B, 1280)
        logits   = self.grading_head(features)  # (B, 3)
        return logits

    def predict_grade(self, x: torch.Tensor) -> dict:
        """
        Convenience method — returns predicted grade with confidence.

        Args:
            x: (1, 3, H, W) single image tensor

        Returns:
            dict with grade_idx, grade_name, severity, confidence, probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)[0]
            grade_idx = int(probs.argmax().item())

        return {
            'grade_idx'    : grade_idx,
            'grade_name'   : self.GRADE_NAMES[grade_idx],
            'severity'     : self.GRADE_SEVERITY[grade_idx],
            'confidence'   : float(probs[grade_idx].item()),
            'probabilities': {
                self.GRADE_NAMES[i]: float(probs[i].item())
                for i in range(self.num_grades)
            }
        }

    def parameter_count(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


if __name__ == '__main__':
    # Quick sanity check
    model  = CancerGradingModel(num_grades=3, pretrained=False)
    dummy  = torch.randn(4, 3, 224, 224)
    output = model(dummy)
    print(f'Input  shape : {dummy.shape}')
    print(f'Output shape : {output.shape}')   # (4, 3)
    print(f'Parameters   : {model.parameter_count()}')

    # Test predict_grade
    result = model.predict_grade(dummy[:1])
    print(f'Prediction   : {result}')

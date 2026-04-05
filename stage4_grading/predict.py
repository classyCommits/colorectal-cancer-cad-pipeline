"""
predict.py
----------
Single image cancer grade prediction.

Usage:
    python predict.py \
        --checkpoint checkpoints/grading_model_best.pth \
        --image      path/to/patch.png
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import CancerGradingModel


def predict_single(checkpoint_path: str, image_path: str) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CancerGradingModel(num_grades=3, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    image  = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    result = model.predict_grade(tensor)

    print('\n' + '='*50)
    print('CANCER GRADE PREDICTION')
    print('='*50)
    print(f"Image      : {image_path}")
    print(f"Grade      : {result['grade_name']}")
    print(f"Severity   : {result['severity']}")
    print(f"Confidence : {result['confidence']*100:.1f}%")
    print('\nProbabilities:')
    for grade_name, prob in result['probabilities'].items():
        bar = '█' * int(prob * 30)
        print(f"  {grade_name[:35]:35s} {prob*100:5.1f}% {bar}")
    print('='*50)

    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image',      required=True)
    args = p.parse_args()
    predict_single(args.checkpoint, args.image)

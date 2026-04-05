"""
gradcam.py
----------
Grad-CAM implementation for EfficientNet-B0 visualization.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights
which regions of an input image the model focused on when making
its prediction. Essential for clinical AI explainability.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", IEEE ICCV 2017.
    DOI: 10.1109/ICCV.2017.74

Usage:
    cam = GradCAM(model)
    heatmap     = cam.generate(image_tensor, class_idx)
    overlay_img = cam.overlay(original_pil_image, heatmap)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM for EfficientNet-B0.

    Hooks into the last convolutional layer (conv_head) to capture
    activations and gradients during the forward/backward pass.

    Parameters
    ----------
    model      : ColonClassifier — the trained EfficientNet model
    target_layer: str — layer name to hook (default: 'conv_head')
    """

    def __init__(self, model, target_layer: str = 'conv_head'):
        self.model        = model
        self.model.eval()
        self.activations  = None
        self.gradients    = None
        self._hooks       = []
        self._register_hooks(target_layer)

    def _register_hooks(self, target_layer: str):
        """Register forward and backward hooks on the target layer."""
        target = getattr(self.model.backbone, target_layer)

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(
        self,
        image_tensor : torch.Tensor,
        class_idx    : int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image_tensor: (1, 3, H, W) preprocessed image tensor
            class_idx   : Target class index. If None, uses predicted class.

        Returns:
            heatmap: (H, W) numpy array, values in [0, 1]
        """
        self.model.zero_grad()

        # Forward pass
        output = self.model(image_tensor)     # (1, num_classes)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for target class
        score = output[0, class_idx]
        score.backward()

        # Global average pooling of gradients → weights
        # gradients shape: (1, C, H, W)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # only positive influences

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.astype(np.float32)

    def overlay(
        self,
        original_image : Image.Image,
        heatmap        : np.ndarray,
        alpha          : float = 0.4,
        colormap       : str   = 'jet',
    ) -> Image.Image:
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            original_image: PIL Image (RGB)
            heatmap       : (H, W) array from generate(), values in [0, 1]
            alpha         : Heatmap opacity (0=invisible, 1=opaque)
            colormap      : Matplotlib colormap name

        Returns:
            PIL Image with heatmap overlay
        """
        import matplotlib.cm as cm

        # Resize heatmap to match image
        img_w, img_h = original_image.size
        heatmap_resized = np.array(
            Image.fromarray(np.uint8(heatmap * 255)).resize(
                (img_w, img_h), Image.BILINEAR
            )
        ) / 255.0

        # Apply colormap
        cmap       = cm.get_cmap(colormap)
        heatmap_rgb = cmap(heatmap_resized)[:, :, :3]  # drop alpha
        heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_rgb)

        # Blend with original
        original_rgb = original_image.convert('RGB')
        blended      = Image.blend(original_rgb, heatmap_pil, alpha=alpha)

        return blended

    def run(
        self,
        image_tensor   : torch.Tensor,
        original_image : Image.Image,
        class_idx      : int   = None,
        alpha          : float = 0.4,
    ) -> tuple:
        """
        Convenience: generate heatmap and overlay in one call.

        Returns:
            (heatmap_array, overlay_pil_image, predicted_class_idx)
        """
        with torch.enable_grad():
            heatmap   = self.generate(image_tensor, class_idx)
            pred_idx  = self.model(image_tensor).argmax(dim=1).item()

        overlay = self.overlay(original_image, heatmap, alpha=alpha)
        return heatmap, overlay, pred_idx

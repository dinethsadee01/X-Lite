"""
Grad-CAM Heatmap Generation (PyTorch)
Generates class-activation heatmaps to show where the model is looking.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image


def _find_target_layer(model):
    """Find the last convolutional/BN layer in the CNN backbone for Grad-CAM."""
    target = None
    for module in model.backbone.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            target = module
    if target is None:
        # fallback: use the backbone itself
        target = model.backbone
    return target


def generate_gradcam(model, input_tensor, target_class, device):
    """
    Generate Grad-CAM heatmap for a given class.
    
    Returns: numpy heatmap [H, W] normalized to [0, 1]
    """
    model.eval()
    activations = []
    gradients = []

    target_layer = _find_target_layer(model)

    def fwd_hook(m, inp, out):
        activations.append(out.detach())

    def bwd_hook(m, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        inp = input_tensor.to(device).requires_grad_(True)
        logits = model(inp)
        model.zero_grad()
        logits[0, target_class].backward()

        if not activations or not gradients:
            # Fallback: uniform heatmap
            return np.ones((inp.shape[2], inp.shape[3]), dtype=np.float32) * 0.5

        act = activations[0]  # [1, C, h, w]
        grad = gradients[0]   # [1, C, h, w]

        # Grad-CAM: weight activations by mean gradient per channel
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inp.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam
    finally:
        h1.remove()
        h2.remove()


def save_heatmap_overlay(original_image, heatmap, save_path, alpha=0.4):
    """Overlay heatmap on original image and save as PNG."""
    img = np.array(original_image.convert('RGB'))
    h, w = img.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = ((1 - alpha) * img + alpha * heatmap_colored).clip(0, 255).astype(np.uint8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(str(save_path), format='PNG')

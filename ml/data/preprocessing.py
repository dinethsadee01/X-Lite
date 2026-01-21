"""
Image Preprocessing Transforms
Handles normalization, resizing, and conversion to tensors
Includes CLAHE for contrast enhancement in medical images
"""

import torch
from torchvision import transforms
from typing import Callable
import cv2
import numpy as np
from PIL import Image

from config import Config


def get_transforms(
    image_size: int = None,
    is_training: bool = True,
    mean: list = None,
    std: list = None
) -> Callable:
    """
    Get preprocessing transforms for images
    
    Args:
        image_size (int): Target image size (square)
        is_training (bool): Whether for training (includes augmentation)
        mean (list): Normalization mean
        std (list): Normalization std
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if image_size is None:
        image_size = Config.IMAGE_SIZE
    
    if mean is None:
        mean = Config.IMAGE_MEAN
    
    if std is None:
        std = Config.IMAGE_STD
    
    if is_training:
        # Training transforms (basic - augmentation handled separately)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation/Test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform


def denormalize(tensor: torch.Tensor, mean: list = None, std: list = None) -> torch.Tensor:
    """
    Denormalize an image tensor for visualization
    
    Args:
        tensor (torch.Tensor): Normalized image tensor (C, H, W)
        mean (list): Normalization mean used
        std (list): Normalization std used
    
    Returns:
        torch.Tensor: Denormalized tensor
    """
    if mean is None:
        mean = Config.IMAGE_MEAN
    if std is None:
        std = Config.IMAGE_STD
    
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    return tensor * std + mean


def get_inference_transforms(image_size: int = None) -> Callable:
    """
    Get transforms for inference (single image prediction)
    
    Args:
        image_size (int): Target image size
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    return get_transforms(image_size=image_size, is_training=False)


class CLAHEPreprocessor:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for chest X-ray preprocessing
    
    CLAHE enhances local contrast without over-amplifying noise
    """
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit (float): Threshold for contrast limiting
            tile_grid_size (tuple): Size of grid for histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply CLAHE to PIL Image
        
        Args:
            image (PIL.Image): Input image (RGB)
        
        Returns:
            PIL.Image: CLAHE-enhanced image
        """
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Convert to grayscale if RGB (X-rays are grayscale)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Apply CLAHE
        enhanced = self.clahe.apply(gray)
        
        # Convert back to RGB for compatibility with models
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Convert back to PIL
        return Image.fromarray(enhanced_rgb)


class GaussianDenoiser:
    """
    Apply Gaussian filtering to reduce noise in X-ray images
    """
    
    def __init__(self, kernel_size=5, sigma=1.0):
        """
        Args:
            kernel_size (int): Size of Gaussian kernel (must be odd)
            sigma (float): Standard deviation of Gaussian
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply Gaussian denoising to PIL Image
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            PIL.Image: Denoised image
        """
        # Convert PIL to numpy
        img_np = np.array(image)
        
        # Apply Gaussian blur
        denoised = cv2.GaussianBlur(
            img_np,
            (self.kernel_size, self.kernel_size),
            self.sigma
        )
        
        # Convert back to PIL
        return Image.fromarray(denoised)


def get_medical_transforms(
    image_size: int = None,
    use_clahe: bool = True,
    use_denoising: bool = False,
    mean: list = None,
    std: list = None
) -> Callable:
    """
    Get preprocessing transforms optimized for medical images
    
    Args:
        image_size (int): Target image size
        use_clahe (bool): Apply CLAHE for contrast enhancement
        use_denoising (bool): Apply Gaussian denoising
        mean (list): Normalization mean
        std (list): Normalization std
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if image_size is None:
        image_size = Config.IMAGE_SIZE
    
    if mean is None:
        mean = Config.IMAGE_MEAN
    
    if std is None:
        std = Config.IMAGE_STD
    
    transform_list = []
    
    # Medical preprocessing
    if use_denoising:
        transform_list.append(GaussianDenoiser(kernel_size=5, sigma=1.0))
    
    if use_clahe:
        transform_list.append(CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8)))
    
    # Standard preprocessing
    transform_list.extend([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transforms.Compose(transform_list)

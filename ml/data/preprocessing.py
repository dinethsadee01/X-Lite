"""
Image Preprocessing Transforms
Handles normalization, resizing, and conversion to tensors
"""

import torch
from torchvision import transforms
from typing import Callable

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

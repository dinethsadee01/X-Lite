"""
Advanced Data Augmentation Pipeline
Uses Albumentations for medical imaging augmentation
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

from config import Config


def get_augmentation_pipeline(
    image_size: int = None,
    augmentation_strength: str = 'medium'
) -> A.Compose:
    """
    Get Albumentations augmentation pipeline
    
    Args:
        image_size (int): Target image size
        augmentation_strength (str): 'light', 'medium', 'heavy'
    
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    if image_size is None:
        image_size = Config.IMAGE_SIZE
    
    # Base transforms (always applied)
    base_transforms = [
        A.Resize(image_size, image_size),
    ]
    
    # Augmentation transforms based on strength
    if augmentation_strength == 'light':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
        ]
    elif augmentation_strength == 'medium':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.3
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
        ]
    elif augmentation_strength == 'heavy':
        aug_transforms = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=20,
                p=0.4
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 80.0), p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.4
            ),
            A.OneOf([
                A.GridDistortion(p=1.0),
                A.ElasticTransform(p=1.0),
            ], p=0.2),
        ]
    else:
        aug_transforms = []
    
    # Normalization (always applied last)
    normalization = [
        A.Normalize(
            mean=Config.IMAGE_MEAN,
            std=Config.IMAGE_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]
    
    # Combine all transforms
    all_transforms = base_transforms + aug_transforms + normalization
    
    return A.Compose(all_transforms)


class AlbumentationsTransform:
    """
    Wrapper to use Albumentations with PyTorch Dataset
    Converts PIL Image to numpy array before applying transforms
    """
    
    def __init__(self, augmentation_pipeline: A.Compose):
        self.augmentation = augmentation_pipeline
    
    def __call__(self, image: Image.Image):
        """
        Apply augmentation to PIL Image
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            torch.Tensor: Augmented image tensor
        """
        # Convert PIL to numpy array
        image_np = np.array(image)
        
        # Apply augmentation
        augmented = self.augmentation(image=image_np)
        
        return augmented['image']


def get_training_transforms(
    image_size: int = None,
    augmentation_strength: str = 'medium'
):
    """
    Get training transforms with augmentation
    
    Args:
        image_size (int): Target image size
        augmentation_strength (str): Augmentation strength
    
    Returns:
        AlbumentationsTransform: Transform callable
    """
    pipeline = get_augmentation_pipeline(image_size, augmentation_strength)
    return AlbumentationsTransform(pipeline)


def get_validation_transforms(image_size: int = None):
    """
    Get validation/test transforms (no augmentation)
    
    Args:
        image_size (int): Target image size
    
    Returns:
        AlbumentationsTransform: Transform callable
    """
    if image_size is None:
        image_size = Config.IMAGE_SIZE
    
    pipeline = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=Config.IMAGE_MEAN,
            std=Config.IMAGE_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    return AlbumentationsTransform(pipeline)

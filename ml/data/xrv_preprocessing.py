"""
Official TorchXRayVision Preprocessing
======================================
This module provides preprocessing ONLY for the teacher model using official
TorchXRayVision transforms. DO NOT use this for student model preprocessing.

Student model uses: ml.data.preprocessing.get_medical_transforms()
Teacher model uses: This module
"""

import torch
import torchvision
import numpy as np
import skimage.io
try:
    import torchxrayvision as xrv
    HAS_XRV = True
except ImportError:
    HAS_XRV = False
    

class XRVTeacherPreprocessor:
    """
    Preprocessor for TorchXRayVision teacher model.
    Uses official XRV transforms as documented in https://mlmed.org/torchxrayvision/
    """
    
    def __init__(self, image_size: int = 224):
        """
        Args:
            image_size: Target image size (default 224 for DenseNet)
        """
        if not HAS_XRV:
            raise ImportError("torchxrayvision not installed")
        
        self.image_size = image_size
        
        # Official TorchXRayVision transforms
        self.transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(image_size),
        ])
    
    def preprocess_single_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image for teacher model.
        
        Args:
            image_path: Path to image file
        
        Returns:
            torch.Tensor: Preprocessed image [1, 1, H, W]
        """
        # Load image using skimage (as per official example)
        img = skimage.io.imread(str(image_path))
        
        # Normalize to [-1024, 1024] using official XRV function
        img = xrv.datasets.normalize(img, 255)
        
        # Add channel dimension if needed (X-rays are grayscale)
        if len(img.shape) == 2:
            img = img[None, ...]  # [1, H, W]
        elif len(img.shape) == 3:
            # RGB to grayscale
            img = img.mean(2)[None, ...]
        
        # Apply official transforms
        img = self.transform(img)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img)
        
        return img_tensor[None, ...]  # Add batch dimension [1, 1, H, W]
    
    def preprocess_batch(self, image_paths: list) -> torch.Tensor:
        """
        Preprocess a batch of images for teacher model.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            torch.Tensor: Batch of preprocessed images [B, 1, H, W]
        """
        batch = []
        for img_path in image_paths:
            img = self.preprocess_single_image(img_path)
            batch.append(img)
        
        return torch.cat(batch, dim=0)
    
    def preprocess_from_numpy(self, img_array: np.ndarray) -> torch.Tensor:
        """
        Preprocess from numpy array (for use in DataLoader).
        Assumes image is already loaded from disk.
        
        Args:
            img_array: Numpy array [H, W] or [H, W, C]
        
        Returns:
            torch.Tensor: Preprocessed image [1, H, W]
        """
        # Normalize to [-1024, 1024]
        img = xrv.datasets.normalize(img_array, 255)
        
        # Handle channel dimension
        if len(img.shape) == 2:
            img = img[None, ...]  # [1, H, W]
        elif len(img.shape) == 3:
            img = img.mean(2)[None, ...]  # RGB to grayscale
        
        # Apply transforms
        img = self.transform(img)
        
        # Convert to tensor
        return torch.from_numpy(img)


def get_xrv_teacher_preprocessor(image_size: int = 224) -> XRVTeacherPreprocessor:
    """
    Factory function to create XRV preprocessor for teacher model.
    
    Args:
        image_size: Target image size
    
    Returns:
        XRVTeacherPreprocessor instance
    """
    return XRVTeacherPreprocessor(image_size=image_size)

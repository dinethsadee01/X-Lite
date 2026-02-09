"""
Teacher Model for Knowledge Distillation
=========================================
Implements DenseNet121 teacher models trained on chest X-ray data.

Options:
- CheXNet (RECOMMENDED): Stanford's model trained on ChestX-ray14 dataset (exact match!)
- TorchXRayVision: Trained on multiple chest X-ray datasets (NIH + others)
- ImageNet pretrained: Standard DenseNet121 from ImageNet

CheXNet is recommended as it's trained on the exact same ChestX-ray14 dataset
with the exact same 14 classes, providing perfect alignment for knowledge distillation.

This model will be used to distill knowledge into lightweight student models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Optional, Literal
import urllib.request


class TeacherModel(nn.Module):
    """
    Medical imaging teacher model for knowledge distillation.
    
    Supports pretrained backends:
    - CheXNet (RECOMMENDED): Stanford's ChestX-ray14 trained model (14 classes, exact match)
    - Standard DenseNet121 (ImageNet): Generic image classification features
    
    Args:
        num_classes (int): Number of output classes (14 for ChestX-ray14)
        pretrained (bool): Load pretrained weights
        model_type (str): 'chexnet' | 'densenet'
        freeze_backbone (bool): Freeze backbone weights during distillation
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        model_type: str = "chexnet",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_type = model_type
        
        if model_type == "chexnet":
            # CheXNet: DenseNet121 trained on ChestX-ray14
            # Architecture: features + single Linear(1024, 14)
            self.backbone = models.densenet121(pretrained=False)
            num_features = self.backbone.classifier.in_features  # 1024
            
            # Replace classifier to match CheXNet structure
            self.backbone.classifier = nn.Linear(num_features, num_classes)
            
        elif model_type == "densenet":
            # Standard ImageNet-pretrained DenseNet121
            if pretrained:
                try:
                    weights = models.DenseNet121_Weights.DEFAULT
                    self.backbone = models.densenet121(weights=weights)
                except Exception:
                    self.backbone = models.densenet121(pretrained=True)
            else:
                self.backbone = models.densenet121(pretrained=False)
            
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                "Choose from: 'chexnet', 'densenet'"
            )
        
        # Freeze backbone if requested (useful during KD to keep teacher fixed)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images (batch_size, 3, 224, 224) - RGB chest X-rays
        
        Returns:
            torch.Tensor: Logits (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        size = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)
        return size


def create_teacher_model(
    num_classes: int = 14,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    model_type: str = "chexnet",
    chexnet_weights_path: Optional[Path] = None,
) -> TeacherModel:
    """
    Factory function to create teacher model with optimal defaults.
    
    Args:
        num_classes (int): Number of output classes (default: 14)
        pretrained (bool): Load pretrained weights (default: True)
        device (torch.device): Device to load on (default: cuda if available, else cpu)
        model_type (str): 'chexnet' (RECOMMENDED) | 'densenet'
        chexnet_weights_path (Path): Path to CheXNet checkpoint (auto-downloads if missing)
    
    Returns:
        TeacherModel: Initialized model on specified device with pretrained weights
    
    Examples:
        # Best choice: CheXNet (exact ChestX-ray14 match)
        model = create_teacher_model()  # Uses chexnet by default
        
        # Alternative: ImageNet-pretrained DenseNet121 (not recommended)
        model = create_teacher_model(model_type='densenet')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base model
    model = TeacherModel(
        num_classes=num_classes,
        pretrained=False,  # We'll load weights manually for CheXNet
        model_type=model_type,
        freeze_backbone=True,  # Keep teacher frozen during KD
    )
    
    # Load CheXNet weights if requested
    if model_type == "chexnet" and pretrained:
        if chexnet_weights_path is None:
            weights_dir = Path('data') / 'weights' / 'chexnet'
            weights_dir.mkdir(parents=True, exist_ok=True)
            chexnet_weights_path = weights_dir / 'model.pth.tar'
        
        # Auto-download if missing
        chexnet_weights_url = 'https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'
        if not chexnet_weights_path.exists():
            print(f"Downloading CheXNet weights to: {chexnet_weights_path}")
            urllib.request.urlretrieve(chexnet_weights_url, chexnet_weights_path)
            print("✓ Download complete")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(chexnet_weights_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(chexnet_weights_path, map_location=device)
        
        # Extract state dict (CheXNet uses DataParallel naming)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix from DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            # module.densenet121.features.conv0.weight -> features.conv0.weight
            name = k.replace('module.densenet121.', '')
            new_state_dict[name] = v
        
        # Load into model
        model.backbone.load_state_dict(new_state_dict, strict=True)
        print("✓ CheXNet weights loaded successfully")
    
    return model.to(device)

"""
Teacher Model for Knowledge Distillation
=========================================
Implements CheXNet-inspired teacher model (DenseNet121 backbone with medical adaptations).

This model will be used to distill knowledge into lightweight student models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Optional
import urllib.request


class TeacherModel(nn.Module):
    """
    Medical imaging teacher model for knowledge distillation.
    
    Architecture:
    - Backbone: DenseNet121 (pretrained on ImageNet)
    - Head: Custom medical imaging classification head
    - Output: Multi-label sigmoid predictions (14 disease classes)
    
    Args:
        num_classes (int): Number of output classes (14 for CheXNet)
        pretrained (bool): Load ImageNet pretrained weights
        freeze_backbone (bool): Freeze backbone weights during training
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        chexnet_compatible: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Load DenseNet121 backbone
        if pretrained:
            try:
                weights = models.DenseNet121_Weights.DEFAULT
                self.backbone = models.densenet121(weights=weights)
            except Exception:
                self.backbone = models.densenet121(pretrained=True)
        else:
            self.backbone = models.densenet121(pretrained=False)

        num_features = self.backbone.classifier.in_features

        if chexnet_compatible:
            # CheXNet-compatible head (matches original checkpoint)
            self.backbone.classifier = nn.Linear(num_features, num_classes)
            self.head = None
        else:
            # Custom medical imaging head
            self.backbone.classifier = nn.Identity()
            self.head = nn.Sequential(
                nn.BatchNorm1d(num_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Logits (batch_size, num_classes)
        """
        features = self.backbone(x)
        if self.head is None:
            return features
        logits = self.head(features)
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        size = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)
        return size
    
    @staticmethod
    def load_pretrained(checkpoint_path: Path, device: torch.device) -> 'TeacherModel':
        """
        Load pretrained teacher model
        
        Args:
            checkpoint_path (Path): Path to checkpoint file
            device (torch.device): Device to load on
        
        Returns:
            TeacherModel: Loaded model
        """
        model = TeacherModel(num_classes=14, pretrained=False, chexnet_compatible=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        return model


def create_teacher_model(
    num_classes: int = 14,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    chexnet_weights_path: Optional[Path] = None,
    chexnet_weights_url: Optional[str] = None
) -> TeacherModel:
    """
    Factory function to create teacher model
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Load ImageNet pretrained weights
        device (torch.device): Device to create on
    
    Returns:
        TeacherModel: Initialized model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If CheXNet weights provided, build a CheXNet-compatible head
    chexnet_mode = chexnet_weights_path is not None or chexnet_weights_url is not None

    model = TeacherModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=False,
        chexnet_compatible=chexnet_mode
    )

    if chexnet_mode:
        if chexnet_weights_path is None:
            weights_dir = Path('data') / 'weights' / 'chexnet'
            weights_dir.mkdir(parents=True, exist_ok=True)
            chexnet_weights_path = weights_dir / 'model.pth.tar'

        if chexnet_weights_url is None:
            chexnet_weights_url = 'https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'

        if not chexnet_weights_path.exists():
            print(f"Downloading CheXNet weights to: {chexnet_weights_path}")
            urllib.request.urlretrieve(chexnet_weights_url, chexnet_weights_path)

        try:
            checkpoint = torch.load(chexnet_weights_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(chexnet_weights_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    return model.to(device)

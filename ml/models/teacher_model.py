"""
Teacher Model for Knowledge Distillation
=========================================
Implements DenseNet121 teacher models trained on chest X-ray data.

Options:
- TorchXRayVision NIH-pretrained (RECOMMENDED): Trained on NIH Chest X-ray 14 dataset
- CheXNet-compatible: Original CheXNet checkpoint from Stanford
- ImageNet pretrained: Standard DenseNet121 from ImageNet

The TorchXRayVision variant is recommended as it's pretrained on actual chest X-ray data,
providing much better domain-specific representations for knowledge distillation.

This model will be used to distill knowledge into lightweight student models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from typing import Optional, Literal
import urllib.request

# Optional: Try to import torchxrayvision for better chest X-ray models
try:
    import torchxrayvision as xrv
    HAS_TORCHXRAYVISION = True
except ImportError:
    HAS_TORCHXRAYVISION = False


class TeacherModel(nn.Module):
    """
    Medical imaging teacher model for knowledge distillation.
    
    Supports multiple pretrained backends:
    - TorchXRayVision DenseNet121 (NIH): Best choice - chest X-ray specific
    - Standard DenseNet121 (ImageNet): Generic image classification features
    - CheXNet: Original Stanford checkpoint
    
    Args:
        num_classes (int): Number of output classes (14 for medical imaging)
        pretrained (bool): Load pretrained weights
        model_type (str): 'torchxrayvision' | 'densenet' | 'chexnet'
        freeze_backbone (bool): Freeze backbone weights during training
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        model_type: str = "torchxrayvision",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model_type = model_type
        
        if model_type == "torchxrayvision":
            if not HAS_TORCHXRAYVISION:
                raise ImportError(
                    "torchxrayvision not installed. "
                    "Install with: pip install torchxrayvision"
                )
            # Load TorchXRayVision's NIH-pretrained DenseNet121 (trained on grayscale X-rays)
            self.backbone = xrv.models.DenseNet(weights="densenet121-res224-nih")
            # Adapt first conv layer to accept 3-channel RGB input (instead of 1-channel grayscale)
            # The pretrained weights from the single-channel conv will be expanded via channel replication
            original_conv = self.backbone.features[0]
            self.backbone.features[0] = nn.Conv2d(
                3, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            # Expand the pretrained weights from 1-channel to 3-channel by replication
            with torch.no_grad():
                self.backbone.features[0].weight.copy_(
                    original_conv.weight.repeat(1, 3, 1, 1) / 3.0
                )
            # Feature extraction from backbone (after features, before classifier)
            # TorchXRayVision outputs 1024 features from DenseNet backbone
            num_features = 1024
            
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
            self.backbone.classifier = nn.Identity()
            
        elif model_type == "chexnet":
            # CheXNet-compatible checkpoint
            self.backbone = models.densenet121(pretrained=False)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                "Choose from: 'torchxrayvision', 'densenet', 'chexnet'"
            )
        
        # Medical imaging head with appropriate capacity
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
            x (torch.Tensor): Input images (batch_size, 3, 224, 224) - RGB chest X-rays
        
        Returns:
            torch.Tensor: Logits (batch_size, num_classes)
        """
        if self.model_type == "torchxrayvision":
            # TorchXRayVision DenseNet: extract features from backbone
            # Input: [batch, 3, 224, 224] RGB X-rays
            # Output of features: [batch, 1024, 7, 7]
            features = self.backbone.features(x)  # [batch, 1024, 7, 7]
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [batch, 1024, 1, 1]
            features = features.view(features.size(0), -1)  # [batch, 1024]
        else:
            # Standard DenseNet forward (ImageNet or CheXNet)
            features = self.backbone(x)
        
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
    def load_pretrained(checkpoint_path: Path, device: torch.device, model_type: str = "torchxrayvision") -> 'TeacherModel':
        """
        Load pretrained teacher model from checkpoint
        
        Args:
            checkpoint_path (Path): Path to checkpoint file
            device (torch.device): Device to load on
            model_type (str): 'torchxrayvision' | 'densenet' | 'chexnet'
        
        Returns:
            TeacherModel: Loaded model in eval mode
        """
        model = TeacherModel(num_classes=14, pretrained=False, model_type=model_type)
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
    model_type: str = "torchxrayvision",
    chexnet_weights_path: Optional[Path] = None,
) -> TeacherModel:
    """
    Factory function to create teacher model with optimal defaults.
    
    Args:
        num_classes (int): Number of output classes (default: 14)
        pretrained (bool): Load pretrained weights (default: True)
        device (torch.device): Device to load on (default: cuda if available, else cpu)
        model_type (str): 'torchxrayvision' (RECOMMENDED) | 'densenet' | 'chexnet'
        chexnet_weights_path (Path): Path to CheXNet checkpoint (for model_type='chexnet')
    
    Returns:
        TeacherModel: Initialized model on specified device
    
    Examples:
        # Best choice: TorchXRayVision NIH-pretrained DenseNet121
        model = create_teacher_model()  # Uses torchxrayvision by default
        
        # Alternative: ImageNet-pretrained DenseNet121
        model = create_teacher_model(model_type='densenet')
        
        # Alternative: CheXNet checkpoint
        model = create_teacher_model(
            model_type='chexnet',
            chexnet_weights_path=Path('data/weights/chexnet/model.pth.tar')
        )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base model
    model = TeacherModel(
        num_classes=num_classes,
        pretrained=pretrained,
        model_type=model_type,
        freeze_backbone=False,
    )
    
    # Load CheXNet specific weights if requested
    if model_type == "chexnet":
        if chexnet_weights_path is None:
            weights_dir = Path('data') / 'weights' / 'chexnet'
            weights_dir.mkdir(parents=True, exist_ok=True)
            chexnet_weights_path = weights_dir / 'model.pth.tar'
        
        # Auto-download if missing
        chexnet_weights_url = 'https://github.com/arnoweng/CheXNet/raw/master/model.pth.tar'
        if not chexnet_weights_path.exists():
            print(f"Downloading CheXNet weights to: {chexnet_weights_path}")
            urllib.request.urlretrieve(chexnet_weights_url, chexnet_weights_path)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(chexnet_weights_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(chexnet_weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    return model.to(device)

"""
Teacher Model for Knowledge Distillation
=========================================
Implements DenseNet121 teacher models trained on chest X-ray data.

SELECTED: TorchXRayVision DenseNet121 (NIH pretrained)
- Trained on: Multiple chest X-ray datasets (ChestX-ray14 + PC + CX + RSNA)
- Classes: 18 total (includes all 14 ChestX-ray14 diseases + 4 extras)
- Architecture: DenseNet121
- Key feature: Uses pretrained classifier (not random initialization)
- Extraction: We extract 14 ChestX-ray14 classes from 18 total output

Why TorchXRayVision over CheXNet:
- Modern PyTorch compatibility (no framework issues)
- More diverse training data (better generalization)
- Actively maintained
- Already installed in environment
- All 14 required classes have learned representations
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    import torchxrayvision as xrv
    HAS_TORCHXRAYVISION = True
except ImportError:
    HAS_TORCHXRAYVISION = False


class TeacherModel(nn.Module):
    """
    TorchXRayVision DenseNet121 teacher for Knowledge Distillation.
    
    Uses the PRETRAINED classifier from TorchXRayVision (not random initialization).
    Outputs logits for 14 ChestX-ray14 classes by extracting from the 18-class output.
    
    Args:
        num_classes (int): Number of output classes to extract (should be 14)
        extract_from_18 (bool): Extract 14 from 18 classes (True for ChestX-ray14)
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        extract_from_18: bool = True,
    ):
        super().__init__()
        
        if not HAS_TORCHXRAYVISION:
            raise ImportError(
                "torchxrayvision not installed. "
                "Install with: pip install torchxrayvision"
            )
        
        self.num_classes = num_classes
        self.extract_from_18 = extract_from_18
        
        # Load full pretrained model with trained classifier
        # This has 18 classes (ChestX-ray14 14 classes + 4 additional pathologies)
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        # Freeze all parameters (teacher is not trained during KD)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Log model structure
        print(f"✓ Loaded TorchXRayVision DenseNet121 (pretrained, frozen)")
        print(f"  Model: densenet121-res224-all (trained on multiple datasets)")
        print(f"  Total output classes: 18 (all pathologies)")
        print(f"  Will extract: {num_classes} (ChestX-ray14 subset)")
        print(f"  Input: Expects correctly preprocessed 1-channel images via XRV transforms")
        
        # XRV disease order (18 classes)
        self.xrv_diseases_18 = [
            'Atelectasis',           # 0
            'Consolidation',         # 1
            'Infiltration',          # 2
            'Pneumothorax',          # 3
            'Edema',                 # 4
            'Emphysema',             # 5
            'Fibrosis',              # 6
            'Effusion',              # 7
            'Pneumonia',             # 8
            'Pleural_Thickening',    # 9
            'Cardiomegaly',          # 10
            'Nodule',                # 11
            'Mass',                  # 12
            'Hernia',                # 13
            'Lung Lesion',           # 14 - NOT in ChestX-ray14
            'Fracture',              # 15 - NOT in ChestX-ray14
            'Lung Opacity',          # 16 - NOT in ChestX-ray14
            'Enlarged Cardiomediastinum',  # 17 - NOT in ChestX-ray14
        ]
        
        # ChestX-ray14 subset (indices in XRV's 18-class output)
        self.chestxray14_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        # These are the 14 diseases we care about (not 14, 15, 16, 17)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns logits for 14 ChestX-ray14 classes.
        
        IMPORTANT: Input must be preprocessed using official TorchXRayVision transforms:
        - xrv.datasets.normalize(img, 255)
        - xrv.datasets.XRayCenterCrop()
        - xrv.datasets.XRayResizer(224)
        
        Args:
            x (torch.Tensor): Preprocessed images [batch_size, 1, 224, 224] (grayscale)
        
        Returns:
            torch.Tensor: Logits [batch_size, 14] for ChestX-ray14 diseases
        """
        # Get full 18-class output from pretrained model
        logits_18 = self.model(x)  # [batch_size, 18]
        
        # Extract only the 14 ChestX-ray14 classes
        logits_14 = logits_18[:, self.chestxray14_indices]  # [batch_size, 14]
        
        return logits_14
    
    def to(self, device):
        """Override to() to ensure all model parts are on the correct device"""
        self.model = self.model.to(device)
        return super().to(device)
    
    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB"""
        size = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)
        return size


def create_teacher_model(
    num_classes: int = 14,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> TeacherModel:
    """
    Factory function to create TorchXRayVision teacher model.
    
    Args:
        num_classes (int): Number of output classes (must be 14 for ChestX-ray14)
        pretrained (bool): Always True - uses XRV's pretrained weights
        device (torch.device): Device to load on (default: cuda if available)
    
    Returns:
        TeacherModel: Initialized model in eval mode on specified device
    
    Example:
        >>> teacher = create_teacher_model()
        >>> teacher.eval()
        >>> logits = teacher(images)  # [batch_size, 14]
    """
    if num_classes != 14:
        raise ValueError(
            f"Only 14 classes supported (ChestX-ray14). Got {num_classes}"
        )
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = TeacherModel(num_classes=num_classes, extract_from_18=True)
    
    # Set to eval mode (frozen, no gradients)
    model.eval()
    
    return model.to(device)

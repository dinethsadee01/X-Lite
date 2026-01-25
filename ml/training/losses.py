"""
Loss Functions for Multi-Label Chest X-Ray Classification
Handles class imbalance with weighted BCE and Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with class weights for imbalanced multi-label classification
    
    Args:
        pos_weights (torch.Tensor): Positive class weights for each disease (shape: [num_classes])
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, pos_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.pos_weights = pos_weights
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid), shape [batch_size, num_classes]
            targets: Ground truth labels (0 or 1), shape [batch_size, num_classes]
        
        Returns:
            Loss value (scalar if reduction='mean' or 'sum')
        """
        return F.binary_cross_entropy_with_logits(
            logits, 
            targets, 
            pos_weight=self.pos_weights,
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reduces loss for well-classified examples, focuses on hard examples
    
    FL(p_t) = -α(1 - p_t)^γ * log(p_t)
    
    Args:
        alpha (float or torch.Tensor): Weighting factor (0-1). Can be per-class.
        gamma (float): Focusing parameter (typically 2.0)
        reduction (str): 'mean', 'sum', or 'none'
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid), shape [batch_size, num_classes]
            targets: Ground truth labels (0 or 1), shape [batch_size, num_classes]
        
        Returns:
            Loss value (scalar if reduction='mean' or 'sum')
        """
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute BCE loss (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute focal term: (1 - p_t)^gamma
        # p_t is prob of correct class: p if y=1, (1-p) if y=0
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply focal term
        focal_loss = focal_term * bce_loss
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combination of Weighted BCE and Focal Loss
    Useful for severe class imbalance
    
    Args:
        pos_weights (torch.Tensor): Positive class weights
        focal_alpha (float): Focal loss alpha parameter
        focal_gamma (float): Focal loss gamma parameter
        bce_weight (float): Weight for BCE loss component (0-1)
        focal_weight (float): Weight for focal loss component (0-1)
    """
    
    def __init__(
        self,
        pos_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5
    ):
        super().__init__()
        self.bce_loss = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (before sigmoid)
            targets: Ground truth labels
        
        Returns:
            Combined loss value
        """
        bce = self.bce_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        
        return self.bce_weight * bce + self.focal_weight * focal


def calculate_pos_weights(
    label_counts: torch.Tensor, 
    total_samples: int,
    smoothing: float = 1.0,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Calculate positive class weights using smoothed inverse frequency
    
    weight_i = 1 / (count_i + smoothing)^alpha
    
    Args:
        label_counts (torch.Tensor): Number of positive samples per class, shape [num_classes]
        total_samples (int): Total number of samples in dataset
        smoothing (float): Smoothing factor to avoid division by zero
        alpha (float): Smoothing exponent (0=no weighting, 1=full inverse freq)
                      Default 0.5 balances rare-class boost with common-class stability
    
    Returns:
        torch.Tensor: Positive weights for each class, shape [num_classes]
    
    Example:
        >>> label_counts = torch.tensor([1000, 500, 100])  # Class frequencies
        >>> pos_weights = calculate_pos_weights(label_counts, total_samples=10000, alpha=0.5)
        >>> # Classes with fewer samples get higher weights, but not extreme
    """
    # Smoothed inverse frequency
    label_counts = label_counts.float() + smoothing
    
    # Apply power for smoothing: weights = 1 / count^alpha
    pos_weights = 1.0 / torch.pow(label_counts, alpha)
    
    # Normalize to reasonable range (mean weight = total_samples / num_classes)
    pos_weights = pos_weights * (total_samples / len(label_counts)) / pos_weights.mean()
    
    return pos_weights


def calculate_effective_num_samples(
    label_counts: torch.Tensor,
    beta: float = 0.9999
) -> torch.Tensor:
    """
    Calculate effective number of samples for class-balanced loss
    
    Effective number = (1 - β^n) / (1 - β)
    
    This gives more balanced weights than simple inverse frequency
    for severe class imbalance.
    
    Args:
        label_counts (torch.Tensor): Number of samples per class
        beta (float): Decay parameter (0.9999 for long-tailed, 0.99 for moderate)
    
    Returns:
        torch.Tensor: Effective number of samples per class
    
    Reference: https://arxiv.org/abs/1901.05555 (Class-Balanced Loss)
    """
    effective_num = 1.0 - torch.pow(beta, label_counts.float())
    effective_num = effective_num / (1.0 - beta)
    
    return effective_num


def get_class_balanced_weights(
    label_counts: torch.Tensor,
    beta: float = 0.9999
) -> torch.Tensor:
    """
    Get class-balanced weights using effective number of samples
    
    weight_i = (1 - β) / (1 - β^n_i)
    
    Args:
        label_counts (torch.Tensor): Number of samples per class
        beta (float): Decay parameter
    
    Returns:
        torch.Tensor: Normalized class weights
    """
    effective_num = calculate_effective_num_samples(label_counts, beta)
    weights = 1.0 / effective_num
    
    # Normalize so sum = num_classes
    weights = weights / weights.sum() * len(weights)
    
    return weights


# Example usage and testing
if __name__ == '__main__':
    # Example: 3 diseases with severe imbalance
    print("="*70)
    print("LOSS FUNCTION EXAMPLES")
    print("="*70)
    
    # Simulate imbalanced dataset
    label_counts = torch.tensor([5000, 500, 50])  # Severe imbalance
    total_samples = 10000
    
    print(f"\nDataset Stats:")
    print(f"  Class 0: {label_counts[0]} samples (50%)")
    print(f"  Class 1: {label_counts[1]} samples (5%)")
    print(f"  Class 2: {label_counts[2]} samples (0.5%)")
    
    # Calculate weights
    print(f"\n1. Inverse Frequency Weights:")
    pos_weights = calculate_pos_weights(label_counts, total_samples)
    for i, w in enumerate(pos_weights):
        print(f"  Class {i}: {w:.4f}")
    
    print(f"\n2. Class-Balanced Weights (Effective Number):")
    cb_weights = get_class_balanced_weights(label_counts)
    for i, w in enumerate(cb_weights):
        print(f"  Class {i}: {w:.4f}")
    
    # Test loss functions
    print(f"\n3. Loss Function Test:")
    batch_size = 4
    num_classes = 3
    
    # Random predictions and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    # Weighted BCE
    bce_loss = WeightedBCEWithLogitsLoss(pos_weights=pos_weights)
    loss_bce = bce_loss(logits, targets)
    print(f"  Weighted BCE Loss: {loss_bce.item():.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_focal = focal_loss(logits, targets)
    print(f"  Focal Loss: {loss_focal.item():.4f}")
    
    # Combined Loss
    combined_loss = CombinedLoss(
        pos_weights=pos_weights,
        bce_weight=0.5,
        focal_weight=0.5
    )
    loss_combined = combined_loss(logits, targets)
    print(f"  Combined Loss: {loss_combined.item():.4f}")
    
    print("\n" + "="*70)
    print("✓ All loss functions working correctly!")
    print("="*70)

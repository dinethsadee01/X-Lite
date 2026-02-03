"""
Knowledge Distillation Loss Functions
======================================
Implements loss functions for knowledge distillation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    
    Combines:
    - KL divergence loss (soft targets from teacher at temperature T)
    - Cross-entropy loss (hard targets from ground truth)
    
    Formula:
        L = alpha * L_KD + (1 - alpha) * L_CE
        
        Where:
        L_KD = KL(softmax(teacher_logits/T), softmax(student_logits/T))
        L_CE = CrossEntropy(student_logits, ground_truth_labels)
    
    Args:
        temperature (float): Temperature for soft targets (higher = softer)
        alpha (float): Weight for KD loss (0-1). 1.0 = pure KD, 0.0 = pure CE
        num_classes (int): Number of output classes
        reduction (str): Loss reduction ('mean', 'sum', 'none')
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        num_classes: int = 14,
        reduction: str = 'mean',
        pos_weights: torch.Tensor = None
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.num_classes = num_classes
        self.reduction = reduction
        self.pos_weights = pos_weights
        
        # Hard loss (cross-entropy with class weights)
        if pos_weights is not None:
            self.ce_loss = nn.BCEWithLogitsLoss(
                pos_weight=pos_weights,
                reduction=reduction
            )
        else:
            self.ce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        
        # Soft loss (KL divergence)
        self.kl_loss = nn.KLDivLoss(reduction=reduction)
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> dict:
        """
        Compute distillation loss
        
        Args:
            student_logits (torch.Tensor): Student model output (batch_size, num_classes)
            teacher_logits (torch.Tensor): Teacher model output (batch_size, num_classes)
            ground_truth (torch.Tensor): Ground truth labels (batch_size, num_classes)
        
        Returns:
            dict: Loss components {
                'total_loss': Total loss,
                'kd_loss': KL divergence loss,
                'ce_loss': Cross-entropy loss
            }
        """
        # Hard loss (ground truth labels)
        ce_loss_val = self.ce_loss(student_logits, ground_truth)
        
        # Soft loss (teacher guidance)
        # Compute soft probabilities at temperature T
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        student_probs = torch.sigmoid(student_logits / self.temperature)
        
        # KL divergence between soft probabilities
        # KL(P||Q) = P * log(P/Q) = P * (log(P) - log(Q))
        kd_loss_val = torch.mean(
            torch.sum(
                teacher_probs * (
                    torch.log(teacher_probs + 1e-8) - torch.log(student_probs + 1e-8)
                ),
                dim=1
            )
        )
        
        # Weighted combination
        total_loss = self.alpha * kd_loss_val + (1 - self.alpha) * ce_loss_val
        
        return {
            'total_loss': total_loss,
            'kd_loss': kd_loss_val,
            'ce_loss': ce_loss_val,
            'alpha': self.alpha,
            'temperature': self.temperature
        }


class FocalDistillationLoss(nn.Module):
    """
    Focal Knowledge Distillation Loss
    
    Combines Focal Loss with KD for handling class imbalance.
    
    Args:
        temperature (float): Temperature for soft targets
        alpha (float): Weight for KD vs CE
        gamma (float): Focusing parameter for Focal Loss (higher = focus on hard examples)
        num_classes (int): Number of output classes
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        gamma: float = 2.0,
        num_classes: int = 14,
        pos_weights: torch.Tensor = None
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.pos_weights = pos_weights
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> dict:
        """
        Compute focal distillation loss
        """
        # Hard loss: Focal loss variant
        student_probs = torch.sigmoid(student_logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            student_logits,
            ground_truth,
            reduction='none'
        )
        
        if self.pos_weights is not None:
            ce_loss = ce_loss * self.pos_weights.unsqueeze(0)
        
        # Focal weighting
        pt = student_probs * ground_truth + (1 - student_probs) * (1 - ground_truth)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = (focal_weight * ce_loss).mean()
        
        # Soft loss (KD)
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        student_probs_soft = torch.sigmoid(student_logits / self.temperature)
        
        kd_loss = torch.mean(
            torch.sum(
                teacher_probs * (
                    torch.log(teacher_probs + 1e-8) - torch.log(student_probs_soft + 1e-8)
                ),
                dim=1
            )
        )
        
        # Weighted combination
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * focal_loss
        
        return {
            'total_loss': total_loss,
            'kd_loss': kd_loss,
            'ce_loss': focal_loss,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'gamma': self.gamma
        }

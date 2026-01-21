"""
Evaluation Metrics for Multi-Label Classification
Computes per-class and macro-averaged metrics
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, hamming_loss,
    coverage_error, label_ranking_average_precision_score,
    roc_curve, auc
)
from typing import Dict, Tuple, List


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    disease_labels: List[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-label classification
    
    Args:
        predictions (torch.Tensor): Model predictions (probabilities), shape [batch_size, num_classes]
        targets (torch.Tensor): Ground truth labels (0 or 1), shape [batch_size, num_classes]
        threshold (float): Decision threshold for predictions
        disease_labels (List[str]): Names of disease classes for per-class metrics
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Convert to numpy
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Binary predictions
    binary_preds = (preds_np >= threshold).astype(float)
    
    num_classes = targets_np.shape[1]
    
    metrics = {}
    
    # Overall metrics
    try:
        # AUC-ROC (macro-averaged across classes)
        auc_scores = []
        for i in range(num_classes):
            if targets_np[:, i].sum() > 0 and targets_np[:, i].sum() < len(targets_np):
                try:
                    auc = roc_auc_score(targets_np[:, i], preds_np[:, i])
                    auc_scores.append(auc)
                except:
                    pass
        
        if auc_scores:
            metrics['AUC_macro'] = np.mean(auc_scores)
            metrics['AUC_micro'] = roc_auc_score(targets_np, preds_np, average='micro')
        else:
            metrics['AUC_macro'] = 0.0
            metrics['AUC_micro'] = 0.0
    except:
        metrics['AUC_macro'] = 0.0
        metrics['AUC_micro'] = 0.0
    
    # F1 Score (macro and micro)
    try:
        metrics['F1_macro'] = f1_score(targets_np, binary_preds, average='macro', zero_division=0)
        metrics['F1_micro'] = f1_score(targets_np, binary_preds, average='micro', zero_division=0)
    except:
        metrics['F1_macro'] = 0.0
        metrics['F1_micro'] = 0.0
    
    # Precision and Recall (macro and micro)
    try:
        metrics['Precision_macro'] = precision_score(targets_np, binary_preds, average='macro', zero_division=0)
        metrics['Precision_micro'] = precision_score(targets_np, binary_preds, average='micro', zero_division=0)
        metrics['Recall_macro'] = recall_score(targets_np, binary_preds, average='macro', zero_division=0)
        metrics['Recall_micro'] = recall_score(targets_np, binary_preds, average='micro', zero_division=0)
    except:
        metrics['Precision_macro'] = 0.0
        metrics['Precision_micro'] = 0.0
        metrics['Recall_macro'] = 0.0
        metrics['Recall_micro'] = 0.0
    
    # Hamming Loss (lower is better)
    try:
        metrics['Hamming_Loss'] = hamming_loss(targets_np, binary_preds)
    except:
        metrics['Hamming_Loss'] = 1.0
    
    # Label ranking metrics
    try:
        metrics['Label_Ranking_AP'] = label_ranking_average_precision_score(targets_np, preds_np)
        metrics['Coverage_Error'] = coverage_error(targets_np, preds_np)
    except:
        metrics['Label_Ranking_AP'] = 0.0
        metrics['Coverage_Error'] = num_classes
    
    # Per-class metrics
    if disease_labels is not None:
        for i, label in enumerate(disease_labels):
            if targets_np[:, i].sum() > 0:  # Only if class appears in batch
                try:
                    auc = roc_auc_score(targets_np[:, i], preds_np[:, i])
                    metrics[f'AUC_{label}'] = auc
                except:
                    pass
                
                try:
                    f1 = f1_score(targets_np[:, i], binary_preds[:, i], zero_division=0)
                    metrics[f'F1_{label}'] = f1
                except:
                    pass
    
    return metrics


def calculate_auc_roc(
    predictions: np.ndarray,
    targets: np.ndarray,
    disease_labels: List[str] = None
) -> Dict[str, float]:
    """
    Calculate per-class and macro-averaged AUC-ROC scores
    
    Args:
        predictions (np.ndarray): Model predictions (probabilities), shape [num_samples, num_classes]
        targets (np.ndarray): Ground truth labels (0 or 1), shape [num_samples, num_classes]
        disease_labels (List[str]): Names of disease classes
    
    Returns:
        dict: AUC scores per class and macro-averaged
    """
    auc_scores = {}
    per_class_aucs = []
    
    num_classes = targets.shape[1]
    
    for i in range(num_classes):
        try:
            # Only calculate if class has both positive and negative samples
            if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                auc = roc_auc_score(targets[:, i], predictions[:, i])
                per_class_aucs.append(auc)
                
                if disease_labels is not None:
                    auc_scores[disease_labels[i]] = auc
                else:
                    auc_scores[f'Class_{i}'] = auc
        except:
            pass
    
    # Macro average
    if per_class_aucs:
        auc_scores['Macro_AUC'] = np.mean(per_class_aucs)
    else:
        auc_scores['Macro_AUC'] = 0.0
    
    return auc_scores


def get_roc_curves(
    predictions: np.ndarray,
    targets: np.ndarray,
    disease_labels: List[str] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Get ROC curves for all classes
    
    Args:
        predictions (np.ndarray): Model predictions (probabilities)
        targets (np.ndarray): Ground truth labels
        disease_labels (List[str]): Names of disease classes
    
    Returns:
        dict: Dictionary with keys as class names and values as (fpr, tpr, auc)
    """
    curves = {}
    num_classes = targets.shape[1]
    
    for i in range(num_classes):
        try:
            if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                fpr, tpr, _ = roc_curve(targets[:, i], predictions[:, i])
                auc_score = auc(fpr, tpr)
                
                class_name = disease_labels[i] if disease_labels is not None else f'Class_{i}'
                curves[class_name] = (fpr, tpr, auc_score)
        except:
            pass
    
    return curves


class MetricsTracker:
    """
    Tracks metrics across epochs during training
    """
    
    def __init__(self, num_classes: int, disease_labels: List[str] = None):
        self.num_classes = num_classes
        self.disease_labels = disease_labels or [f'Class_{i}' for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch
        
        Args:
            predictions (torch.Tensor): Batch predictions [batch_size, num_classes]
            targets (torch.Tensor): Batch targets [batch_size, num_classes]
        """
        self.predictions.append(predictions.cpu().numpy())
        self.targets.append(targets.cpu().numpy())
    
    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute all accumulated metrics
        
        Args:
            threshold (float): Decision threshold
        
        Returns:
            dict: All computed metrics
        """
        if len(self.predictions) == 0:
            return {}
        
        # Concatenate all batches
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        # Convert back to tensors for compute_metrics
        pred_tensor = torch.from_numpy(all_predictions)
        target_tensor = torch.from_numpy(all_targets)
        
        return compute_metrics(
            pred_tensor,
            target_tensor,
            threshold=threshold,
            disease_labels=self.disease_labels
        )
    
    def get_auc_scores(self) -> Dict[str, float]:
        """Get per-class AUC-ROC scores"""
        if len(self.predictions) == 0:
            return {}
        
        all_predictions = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        return calculate_auc_roc(all_predictions, all_targets, self.disease_labels)

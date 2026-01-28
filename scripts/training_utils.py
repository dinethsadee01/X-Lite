"""
Training Utilities for Baseline Training
=========================================
Helper functions for:
- Computing PR-AUC
- Resume capability after power outages
- Final comprehensive metrics evaluation
"""

import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_curve, auc as pr_auc

from ml.training.metrics import compute_metrics
from config.disease_labels import DISEASE_LABELS

project_root = Path(__file__).parent.parent


def compute_pr_auc_scores(model, loader, device):
    """
    Compute PR-AUC (Precision-Recall AUC) for all classes
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Device
    
    Returns:
        dict: PR-AUC scores per class and macro average
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    pr_auc_scores = []
    for i in range(len(DISEASE_LABELS)):
        if targets[:, i].sum() > 0:  # Only if class has positives
            try:
                precision, recall, _ = precision_recall_curve(targets[:, i], preds[:, i])
                pr_auc_score = pr_auc(recall, precision)
                pr_auc_scores.append(pr_auc_score)
            except:
                pr_auc_scores.append(0.0)
        else:
            pr_auc_scores.append(0.0)
    
    return {
        'pr_auc_macro': np.mean(pr_auc_scores),
        'pr_auc_per_class': pr_auc_scores
    }


def evaluate_final_metrics(model, loader, device):
    """
    Compute comprehensive final metrics including PR-AUC, precision, recall
    
    Args:
        model: Trained model
        loader: DataLoader  
        device: Device
    
    Returns:
        dict: All metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs)
            all_targets.append(targets)
    
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute standard metrics
    metrics = compute_metrics(predictions, targets, threshold=0.5, disease_labels=DISEASE_LABELS)
    
    # Compute PR-AUC
    pr_auc_results = compute_pr_auc_scores(model, loader, device)
    metrics['PR_AUC_macro'] = pr_auc_results['pr_auc_macro']
    
    return metrics


def load_training_progress():
    """
    Load record of completed model trainings
    
    Returns:
        set: Set of completed model names
    """
    progress_file = project_root / "experiments" / "training_progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get('completed_models', []))
    return set()


def save_training_progress(completed_models):
    """
    Save record of completed model trainings
    
    Args:
        completed_models (set): Set of completed model names
    """
    progress_file = project_root / "experiments" / "training_progress.json"
    progress_file.parent.mkdir(exist_ok=True)
    
    data = {
        'completed_models': list(completed_models),
        'last_updated': datetime.now().isoformat()
    }
    with open(progress_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Training progress saved to: {progress_file}")


def reset_training_progress():
    """
    Delete training progress file to restart all trainings
    """
    progress_file = project_root / "experiments" / "training_progress.json"
    if progress_file.exists():
        progress_file.unlink()
        print(f"✅ Training progress reset (deleted {progress_file})")
    else:
        print("⚠️  No training progress file found")

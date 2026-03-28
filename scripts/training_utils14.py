"""
Training utilities for 14-class setup (without No_Finding).
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc as pr_auc

from config.disease_labels14 import DISEASE_LABELS14
from ml.training.metrics import compute_metrics

project_root = Path(__file__).parent.parent


def compute_pr_auc_scores14(model, loader, device):
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
    for i in range(len(DISEASE_LABELS14)):
        if targets[:, i].sum() > 0:
            try:
                precision, recall, _ = precision_recall_curve(targets[:, i], preds[:, i])
                pr_auc_score = pr_auc(recall, precision)
                pr_auc_scores.append(pr_auc_score)
            except Exception:
                pr_auc_scores.append(0.0)
        else:
            pr_auc_scores.append(0.0)

    return {
        'pr_auc_macro': float(np.mean(pr_auc_scores)),
        'pr_auc_per_class': pr_auc_scores,
    }


def evaluate_final_metrics14(model, loader, device):
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

    metrics = compute_metrics(predictions, targets, threshold=0.5, disease_labels=DISEASE_LABELS14)
    pr_auc_results = compute_pr_auc_scores14(model, loader, device)
    metrics['PR_AUC_macro'] = pr_auc_results['pr_auc_macro']

    return metrics


def load_training_progress14():
    progress_file = project_root / 'experiments' / 'training_progress14.json'
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('completed_models', []))
    return set()


def save_training_progress14(completed_models):
    progress_file = project_root / 'experiments' / 'training_progress14.json'
    progress_file.parent.mkdir(exist_ok=True)

    data = {
        'completed_models': list(completed_models),
        'last_updated': datetime.now().isoformat(),
    }
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def reset_training_progress14():
    progress_file = project_root / 'experiments' / 'training_progress14.json'
    if progress_file.exists():
        progress_file.unlink()

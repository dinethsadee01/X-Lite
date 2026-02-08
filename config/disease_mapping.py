"""
Disease Order Mapping: Your Labels ↔ TorchXRayVision Order
===========================================================

CRITICAL: TorchXRayVision expects a specific disease order. If labels don't match,
the student learns wrong associations (e.g., "Pneumonia" features for "Atelectasis").

This module provides bidirectional mapping between:
- Your order (15 classes: 14 diseases + No_Finding, alphabetically sorted)
- TorchXRayVision order (14 classes: specific NIH order)
"""

# TorchXRayVision's canonical disease order (from NIH Chest X-ray 14 dataset)
XRV_DISEASE_ORDER = [
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
]

# Your current order (from disease_labels.py)
YOUR_DISEASE_ORDER = [
    'Atelectasis',           # 0
    'Cardiomegaly',          # 1
    'Effusion',              # 2
    'Infiltration',          # 3
    'Mass',                  # 4
    'Nodule',                # 5
    'Pneumonia',             # 6
    'Pneumothorax',          # 7
    'Consolidation',         # 8
    'Edema',                 # 9
    'Emphysema',             # 10
    'Fibrosis',              # 11
    'Pleural_Thickening',    # 12
    'Hernia',                # 13
    'No_Finding',            # 14 (not in XRV)
]

# Create mapping from your indices to XRV indices
# your_index -> xrv_index
YOUR_TO_XRV_MAPPING = {}
for your_idx, disease in enumerate(YOUR_DISEASE_ORDER):
    if disease != 'No_Finding':  # Skip No_Finding (not distilled)
        xrv_idx = XRV_DISEASE_ORDER.index(disease)
        YOUR_TO_XRV_MAPPING[your_idx] = xrv_idx

# Inverse mapping: XRV indices to your indices
# xrv_index -> your_index
XRV_TO_YOUR_MAPPING = {v: k for k, v in YOUR_TO_XRV_MAPPING.items()}


def reorder_labels_to_xrv(your_labels: list) -> list:
    """
    Reorder your labels/logits to TorchXRayVision order.
    
    Args:
        your_labels (list): Logits or labels in your order (14 or 15 elements)
    
    Returns:
        list: Reordered logits/labels in XRV order (14 elements)
    """
    if len(your_labels) == 15:
        # Remove No_Finding (index 14)
        your_labels = your_labels[:14]
    
    xrv_labels = [0] * 14
    for your_idx, xrv_idx in YOUR_TO_XRV_MAPPING.items():
        xrv_labels[xrv_idx] = your_labels[your_idx]
    return xrv_labels


def reorder_labels_from_xrv(xrv_labels: list) -> list:
    """
    Reorder XRV labels/logits back to your order.
    
    Args:
        xrv_labels (list): Logits or labels in XRV order (14 elements)
    
    Returns:
        list: Reordered logits/labels in your order (14 elements)
    """
    your_labels = [0] * 14
    for xrv_idx, your_idx in XRV_TO_YOUR_MAPPING.items():
        your_labels[your_idx] = xrv_labels[xrv_idx]
    return your_labels


def reorder_labels_batch_to_xrv(batch_labels, no_finding_idx=14):
    """
    Reorder a batch of labels from your order to XRV order.
    
    Args:
        batch_labels: Tensor/array of shape [batch, 14 or 15]
        no_finding_idx: Index of No_Finding in your order (default: 14)
    
    Returns:
        Reordered batch in XRV order [batch, 14]
    """
    import torch
    import numpy as np
    
    is_tensor = isinstance(batch_labels, torch.Tensor)
    
    # Convert to numpy for easier indexing
    if is_tensor:
        labels_np = batch_labels.cpu().numpy() if batch_labels.is_cuda else batch_labels.numpy()
    else:
        labels_np = np.array(batch_labels)
    
    batch_size = labels_np.shape[0]
    
    # Remove No_Finding if present
    if labels_np.shape[1] == 15:
        labels_np = labels_np[:, :14]
    
    # Reorder using mapping
    reordered = np.zeros((batch_size, 14), dtype=labels_np.dtype)
    for your_idx, xrv_idx in YOUR_TO_XRV_MAPPING.items():
        reordered[:, xrv_idx] = labels_np[:, your_idx]
    
    if is_tensor:
        device = batch_labels.device
        reordered = torch.from_numpy(reordered).to(device)
    
    return reordered


def reorder_labels_batch_from_xrv(batch_labels):
    """
    Reorder a batch of labels from XRV order back to your order.
    
    Args:
        batch_labels: Tensor/array of shape [batch, 14]
    
    Returns:
        Reordered batch in your order [batch, 14]
    """
    import torch
    import numpy as np
    
    is_tensor = isinstance(batch_labels, torch.Tensor)
    
    # Convert to numpy
    if is_tensor:
        labels_np = batch_labels.cpu().numpy() if batch_labels.is_cuda else batch_labels.numpy()
    else:
        labels_np = np.array(batch_labels)
    
    batch_size = labels_np.shape[0]
    
    # Reorder using inverse mapping
    reordered = np.zeros((batch_size, 14), dtype=labels_np.dtype)
    for xrv_idx, your_idx in XRV_TO_YOUR_MAPPING.items():
        reordered[:, your_idx] = labels_np[:, xrv_idx]
    
    if is_tensor:
        device = batch_labels.device
        reordered = torch.from_numpy(reordered).to(device)
    
    return reordered


if __name__ == '__main__':
    print("TorchXRayVision Disease Mapping")
    print("=" * 60)
    print("\nMapping from YOUR order → TorchXRayVision order:")
    print("-" * 60)
    for your_idx, xrv_idx in sorted(YOUR_TO_XRV_MAPPING.items()):
        your_disease = YOUR_DISEASE_ORDER[your_idx]
        xrv_disease = XRV_DISEASE_ORDER[xrv_idx]
        print(f"  [{your_idx:2d}] {your_disease:<25} → [{xrv_idx:2d}] {xrv_disease}")
    
    print("\nNo_Finding (index 14) is NOT distilled - only learned from hard labels")

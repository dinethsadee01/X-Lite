"""
ChestX-ray14 labels for 14-class setup (without No_Finding).
"""

DISEASE_LABELS14 = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia',
]

NUM_CLASSES14 = len(DISEASE_LABELS14)
LABEL_MAPPING14 = {label: idx for idx, label in enumerate(DISEASE_LABELS14)}
IDX_TO_LABEL14 = {idx: label for label, idx in LABEL_MAPPING14.items()}

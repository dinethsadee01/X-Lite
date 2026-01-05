"""
ChestX-ray14 Disease Labels and Metadata
14 thoracic disease classes from NIH Clinical Center dataset
"""

# 14 disease labels (official ChestX-ray14 classes)
DISEASE_LABELS = [
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
    'Hernia'
]

NUM_CLASSES = len(DISEASE_LABELS)

# Label to index mapping
LABEL_MAPPING = {label: idx for idx, label in enumerate(DISEASE_LABELS)}

# Index to label mapping
IDX_TO_LABEL = {idx: label for label, idx in LABEL_MAPPING.items()}

# Disease descriptions for report generation
DISEASE_DESCRIPTIONS = {
    'Atelectasis': 'Partial collapse of the lung',
    'Cardiomegaly': 'Enlargement of the heart',
    'Effusion': 'Fluid buildup around the lungs (pleural effusion)',
    'Infiltration': 'Substance denser than air in lung parenchyma',
    'Mass': 'Abnormal spot or area in the lungs',
    'Nodule': 'Small rounded opacity in the lung',
    'Pneumonia': 'Infection causing inflammation in the air sacs',
    'Pneumothorax': 'Collapsed lung due to air in pleural space',
    'Consolidation': 'Air spaces filled with fluid, pus, blood, or cells',
    'Edema': 'Excess fluid in the lungs (pulmonary edema)',
    'Emphysema': 'Damage to air sacs causing breathing difficulty',
    'Fibrosis': 'Thickening and scarring of lung tissue',
    'Pleural_Thickening': 'Thickening of pleura (lung lining)',
    'Hernia': 'Protrusion of organ through cavity wall'
}

# Risk levels for UI color coding
RISK_THRESHOLDS = {
    'low': 0.3,      # < 30%: Low probability
    'medium': 0.6,   # 30-60%: Medium probability
    'high': 1.0      # > 60%: High probability
}

def get_risk_level(probability):
    """
    Get risk level based on prediction probability
    
    Args:
        probability (float): Prediction probability [0, 1]
    
    Returns:
        str: 'low', 'medium', or 'high'
    """
    if probability < RISK_THRESHOLDS['low']:
        return 'low'
    elif probability < RISK_THRESHOLDS['medium']:
        return 'medium'
    else:
        return 'high'

def get_risk_color(probability):
    """
    Get color code for risk level (for UI visualization)
    
    Args:
        probability (float): Prediction probability [0, 1]
    
    Returns:
        str: Color hex code
    """
    risk = get_risk_level(probability)
    colors = {
        'low': '#4CAF50',     # Green
        'medium': '#FF9800',  # Orange
        'high': '#F44336'     # Red
    }
    return colors[risk]

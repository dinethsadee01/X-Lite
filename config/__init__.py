"""
Configuration module for X-Lite project.
Handles paths, hyperparameters, and disease labels.
"""

from .config import Config
from .disease_labels import (
    DISEASE_LABELS, 
    NUM_CLASSES, 
    LABEL_MAPPING,
    DISEASE_DESCRIPTIONS,
    get_risk_level,
    get_risk_color
)

__all__ = [
    'Config', 
    'DISEASE_LABELS', 
    'NUM_CLASSES', 
    'LABEL_MAPPING',
    'DISEASE_DESCRIPTIONS',
    'get_risk_level',
    'get_risk_color'
]

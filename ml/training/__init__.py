"""
Training package for teacher and student models
"""

from .teacher_trainer import TeacherTrainer
from .student_trainer import StudentTrainer
from .metrics import compute_metrics, calculate_auc_roc

__all__ = ['TeacherTrainer', 'StudentTrainer', 'compute_metrics', 'calculate_auc_roc']

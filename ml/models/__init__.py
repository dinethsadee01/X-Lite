"""
Model architectures package.

Teacher models will be added later; only student models are implemented now.
We export the student model factory and configs without importing nonexistent
TeacherModel or StudentModel symbols.
"""

from .student_model import (
	create_student_model,
	MODEL_CONFIGS,
	HybridStudentModel,
	StudentClassificationHead,
	MultiHeadSelfAttention,
	PerformerAttention,
)

__all__ = [
	'create_student_model',
	'MODEL_CONFIGS',
	'HybridStudentModel',
	'StudentClassificationHead',
	'MultiHeadSelfAttention',
	'PerformerAttention',
]

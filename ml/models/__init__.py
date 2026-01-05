"""
Model architectures package
Contains teacher, student models and components
"""

from .teacher_model import TeacherModel
from .student_model import StudentModel

__all__ = ['TeacherModel', 'StudentModel']

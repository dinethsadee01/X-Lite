"""
Backend routes package
"""

from . import health, upload, predict, report, auth

__all__ = ['health', 'upload', 'predict', 'report', 'auth']

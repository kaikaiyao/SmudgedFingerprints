"""
Attack evaluation module for fingerprint robustness testing.

This module provides functionality for evaluating attack effectiveness,
computing metrics, and analyzing results.
"""

from .evaluator import AttackEvaluator
from .metrics import AttackMetrics
from .target_selection import TargetSelector

__all__ = ['AttackEvaluator', 'AttackMetrics', 'TargetSelector']

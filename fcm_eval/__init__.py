"""
FCM Evaluation Module
=====================

So sánh FCM V1 và FCM V2 với cùng các test cases.
"""

from .quick_start import run_quick_start_comparison
from .demo import run_demo_comparison
from .locomo import run_locomo_comparison

__all__ = [
    "run_quick_start_comparison",
    "run_demo_comparison", 
    "run_locomo_comparison",
]

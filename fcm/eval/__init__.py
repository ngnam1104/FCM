"""
FCM Evaluation Module
=====================

So sánh FCM V1 và FCM V2 với các benchmark datasets.

Usage:
    from fcm.eval import run_locomo_comparison
    python -m fcm.eval locomo
"""

from fcm.eval.quick_start import run_quick_start_comparison
from fcm.eval.demo import run_demo_comparison
from fcm.eval.locomo import run_locomo_comparison

__all__ = [
    "run_quick_start_comparison",
    "run_demo_comparison", 
    "run_locomo_comparison",
]

"""
Common utilities for FCM Evaluation
===================================

Shared test data and helper functions for comparing V1 vs V2.
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("⚠️ WARNING: GROQ_API_KEY not found in .env or environment!")


@dataclass
class BenchmarkResult:
    """Kết quả benchmark của một kiến trúc"""
    version: str
    total_time: float
    messages_processed: int
    facts_extracted: int
    search_accuracy: float
    memory_stats: Dict[str, int]
    additional_metrics: Dict[str, Any]


# ============================================================
# SHARED TEST DATA
# ============================================================

QUICK_START_MESSAGES = [
    "Xin chào, tôi tên là Nam, sinh năm 2003.",
    "Tôi là sinh viên Bách Khoa Hà Nội, chuyên ngành KHMT.",
    "Tôi học ngành Công nghệ thông tin, hiện đang làm đồ án tốt nghiệp.",
    "Tôi thích lập trình Python và nghiên cứu về AI Agent.",
    "Cuối tuần này tôi sẽ đi Đà Lạt du lịch với bạn bè.",
]

# Format: (query, [list of acceptable answers])
# Validation passes if any answer is found in top search result
QUICK_START_QUERIES = [
    ("Nam sinh năm bao nhiêu?", ["2003", "sinh năm 2003"]),
    ("Nam học ngành gì?", ["Công nghệ thông tin", "CNTT", "KHMT", "chuyên ngành"]),
    ("Nam thích lập trình ngôn ngữ gì?", ["Python", "python", "lập trình Python"]),
]

DEMO_MESSAGES = [
    # Personal info segment
    "Xin chào, tôi là Minh, sinh viên năm 3 Bách Khoa Hà Nội",
    "Tôi đang học ngành Công nghệ thông tin, chuyên về AI",
    "Sở thích của tôi là chơi guitar và đọc sách về machine learning",
    # Plans segment (topic shift)
    "Cuối tuần này tôi dự định đi Đà Lạt với bạn bè",
    "Tôi thích cà phê sữa đá hơn là trà sữa",
    # Work info segment (topic shift)
    "Tôi từng làm intern ở FPT Software năm 2023",
    "Hiện tại tôi đang thực tập ở VinAI",
]

DEMO_QUERIES = [
    ("Minh học ngành gì?", "Công nghệ thông tin"),
    ("Minh thích uống gì?", "cà phê sữa đá"),
    ("Kế hoạch cuối tuần của Minh", "Đà Lạt"),
    ("Minh làm ở đâu?", "VinAI"),
    ("Minh thực tập ở đâu trước đây?", "FPT"),
]

CONFLICT_MESSAGES = [
    # Session 1
    ("Tôi tên là Lan, thích ăn phở bò", False),
    ("Tôi làm việc tại công ty ABC", False),
    # Session 2 (conflicts)
    ("Giờ tôi chuyển sang thích bún chả hơn phở", True),  # Conflict
    ("Tôi vừa nhảy việc sang công ty XYZ tuần trước", True),  # Conflict
]

CONFLICT_EXPECTED = {
    "food_preference": "bún chả",  # Should be updated
    "workplace": "XYZ",  # Should be updated
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title: str, char: str = "="):
    """Print formatted header"""
    width = 70
    print("\n" + char * width)
    print(f"  {title}".center(width))
    print(char * width)


def print_subheader(title: str):
    """Print subheader"""
    print(f"\n--- {title} ---")


def print_comparison_table(v1_result: BenchmarkResult, v2_result: BenchmarkResult):
    """Print comparison table between V1 and V2"""
    print("\n" + "=" * 70)
    print(" " * 20 + "COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Metric':<30} {'FCM V1':>18} {'FCM V2':>18}")
    print("-" * 70)
    
    # Time
    print(f"{'Total Time (s)':<30} {v1_result.total_time:>18.2f} {v2_result.total_time:>18.2f}")
    
    # Messages
    print(f"{'Messages Processed':<30} {v1_result.messages_processed:>18} {v2_result.messages_processed:>18}")
    
    # Facts
    print(f"{'Facts Extracted':<30} {v1_result.facts_extracted:>18} {v2_result.facts_extracted:>18}")
    
    # Accuracy
    print(f"{'Search Accuracy':<30} {v1_result.search_accuracy:>17.1%} {v2_result.search_accuracy:>17.1%}")
    
    # Memory stats
    for key in v1_result.memory_stats:
        v1_val = v1_result.memory_stats.get(key, 0)
        v2_val = v2_result.memory_stats.get(key, 0)
        print(f"{key:<30} {v1_val:>18} {v2_val:>18}")
    
    # Additional V2 metrics
    if v2_result.additional_metrics:
        print("-" * 70)
        print("V2 Additional Metrics:")
        for key, val in v2_result.additional_metrics.items():
            print(f"  • {key}: {val}")
    
    print("=" * 70)
    
    # Winner
    if v2_result.search_accuracy > v1_result.search_accuracy:
        print("🏆 WINNER: FCM V2 (Higher Accuracy)")
    elif v2_result.search_accuracy < v1_result.search_accuracy:
        print("🏆 WINNER: FCM V1 (Higher Accuracy)")
    else:
        if v2_result.total_time < v1_result.total_time:
            print("🏆 WINNER: FCM V2 (Same Accuracy, Faster)")
        else:
            print("🤝 TIE: Both architectures perform similarly")


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    return text.strip().lower()


def check_answer(retrieved: str, expected) -> bool:
    """
    Check if retrieved text contains expected answer(s).
    - expected: str hoặc list[str]
    """
    norm_retrieved = normalize_text(retrieved)
    if isinstance(expected, list):
        return any(normalize_text(ans) in norm_retrieved for ans in expected)
    return normalize_text(expected) in norm_retrieved


def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper

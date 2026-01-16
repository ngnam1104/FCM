"""
Frequency-based Crystallizing Memory (FCM)
==========================================

Kiến trúc bộ nhớ phân tầng theo tần số cập nhật, mô phỏng quá trình nhận thức của con người:
- Liquid Layer (High-Frequency): Lưu thông tin thô, cập nhật liên tục
- Crystal Layer (Mid-Frequency): Thông tin đã được lọc, có cấu trúc  
- Solid Layer (Low-Frequency): Kiến thức cốt lõi, bền vững

Dựa trên các nghiên cứu: Nested Learning, SeCom, A-Mem, MAPLE, InfLLM

Enhanced Features (v0.2.3):
- SeCom: LLM-based Topic Shift Detection for Semantic Segmentation
- SeCom: Auto-trigger Crystallize on Topic Shift (không cần đợi threshold)
- SeCom/COMEDY: Conversation Compression (Compressive Memory)
- MAPLE: Version Tracking với Linked List of Knowledge + Reflection Reasoning
- MAPLE: Proper Archive (Delete -> Insert with archived status)
- A-Mem: Zettelkasten Linking với keywords, context_tags, related_to
- G-Memory: Memory History Tracing
- Search: Client-side filtering for archived memories
- Modular Architecture: Separate layer modules for better maintainability
"""

from fcm.agent import FCMAgent
from fcm.config import FCMConfig, get_default_fcm_config

# Layer modules (new modular architecture)
from fcm.liquid import LiquidLayer
from fcm.crystal import CrystalLayer
from fcm.solid import SolidLayer

# Schemas
from fcm.schemas import (
    MemoryCategory,
    MemoryStatus,
    ConversationStats,
    LiquidMessage,
    CrystalFact,
    SolidKnowledge,
    CompressionResult,
    TopicShiftResult,
)

__all__ = [
    # Main Agent
    "FCMAgent",
    "FCMConfig", 
    "get_default_fcm_config",
    # Layer Modules
    "LiquidLayer",
    "CrystalLayer",
    "SolidLayer",
    # Schemas
    "MemoryCategory",
    "MemoryStatus",
    "ConversationStats",
    "LiquidMessage",
    "CrystalFact",
    "SolidKnowledge",
    "CompressionResult",
    "TopicShiftResult",
]

__version__ = "0.3.0"  # Major refactoring with modular architecture

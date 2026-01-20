"""
FCM V1 - Frequency-based Crystallizing Memory
==============================================

Kiến trúc bộ nhớ phân tầng theo tần số cập nhật:
- Liquid Layer (High-Frequency): Lưu thông tin thô
- Crystal Layer (Mid-Frequency): Thông tin đã lọc, có cấu trúc  
- Solid Layer (Low-Frequency): Kiến thức cốt lõi, bền vững

Usage:
    from fcm.v1 import FCMAgent, FCMConfig
"""

from fcm.v1.agent import FCMAgent
from fcm.v1.config import FCMConfig, get_default_fcm_config

# Layer modules
from fcm.v1.liquid import LiquidLayer
from fcm.v1.crystal import CrystalLayer
from fcm.v1.solid import SolidLayer

# Schemas
from fcm.v1.schemas import (
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

__version__ = "1.0.0"

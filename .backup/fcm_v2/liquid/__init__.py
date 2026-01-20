"""
FCM V2 Liquid Layer
===================

Liquid Layer với các cải tiến:
1. Attention Sinks: Luôn giữ K tin nhắn đầu tiên
2. Semantic Grouping: Tính embedding similarity trước khi gọi LLM check topic shift
"""

from fcm_v2.liquid.layer import LiquidLayer

__all__ = ["LiquidLayer"]

"""
FCM V2 - Enhanced Frequency-based Crystallizing Memory
=======================================================

Version 2.0 với các cải tiến:
1. Bi-Temporal Schema cho Crystal Layer
2. Attention Sinks & Semantic Grouping cho Liquid Layer
3. Active Forgetting (Ebbinghaus Curve) cho Solid Layer
4. Dynamic Persona
5. Weighted Ensemble Retrieval

Usage:
    from fcm.v2 import FCMAgentV2, FCMConfigV2
"""

__version__ = "2.0.0"

from fcm.v2.config import FCMConfigV2, get_default_config_v2
from fcm.v2.agent import FCMAgentV2
from fcm.v2.schemas.base import AtomicFact, MemoryStrength, UserPersona

__all__ = [
    "FCMConfigV2",
    "get_default_config_v2",
    "FCMAgentV2",
    "AtomicFact", 
    "MemoryStrength",
    "UserPersona",
]

"""
FCM V2 - Frequency-based Crystallizing Memory
==============================================

Version 2.0 với các cải tiến:
1. Bi-Temporal Schema cho Crystal Layer
2. Attention Sinks & Semantic Grouping cho Liquid Layer
3. Active Forgetting (Ebbinghaus Curve) cho Solid Layer
4. Dynamic Persona
5. Weighted Ensemble Retrieval

Cấu trúc module:
- liquid/: Liquid Layer (High-Frequency)
- crystal/: Crystal Layer (Mid-Frequency)  
- solid/: Solid Layer (Low-Frequency)
- retrieval/: Hybrid Retrieval System
- schemas/: Pydantic data models
"""

__version__ = "2.0.0"

from fcm_v2.config import FCMConfigV2, get_default_config_v2
from fcm_v2.agent import FCMAgentV2
from fcm_v2.schemas.base import AtomicFact, MemoryStrength, UserPersona

__all__ = [
    "FCMConfigV2",
    "get_default_config_v2",
    "FCMAgentV2",
    "AtomicFact", 
    "MemoryStrength",
    "UserPersona",
]

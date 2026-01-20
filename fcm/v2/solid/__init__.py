"""
FCM V2 Solid Layer
==================

Solid Layer với các cải tiến:
1. Active Forgetting (Ebbinghaus Curve)
2. Dynamic Persona extraction
3. Cold Storage cho memories yếu
"""

from fcm.v2.solid.layer import SolidLayer

__all__ = ["SolidLayer"]

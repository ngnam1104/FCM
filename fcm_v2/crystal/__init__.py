"""
FCM V2 Crystal Layer
====================

Crystal Layer với cải tiến Bi-Temporal:
- Trích xuất valid_at (thời gian sự kiện xảy ra)
- Ưu tiên khi search với temporal context
"""

from fcm_v2.crystal.layer import CrystalLayer

__all__ = ["CrystalLayer"]

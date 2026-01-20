"""
FCM Package - Frequency-based Crystallizing Memory
===================================================

Unified package containing both FCM V1 and V2 implementations.

Usage:
    from fcm.v1 import FCMAgent, FCMConfig
    from fcm.v2 import FCMAgentV2, FCMConfigV2
    from fcm.eval import run_locomo_comparison
"""

__version__ = "2.1.0"

# Re-export commonly used classes from v1
from fcm.v1 import FCMAgent, FCMConfig, get_default_fcm_config

# Re-export commonly used classes from v2
from fcm.v2 import FCMAgentV2, FCMConfigV2, get_default_config_v2

__all__ = [
    # V1
    "FCMAgent",
    "FCMConfig", 
    "get_default_fcm_config",
    # V2
    "FCMAgentV2",
    "FCMConfigV2",
    "get_default_config_v2",
]

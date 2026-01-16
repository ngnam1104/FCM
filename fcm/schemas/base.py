"""
FCM V1 Base Schemas
===================

Data structures cho FCM V1.
Compatible với FCM V2 để dễ so sánh.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class MemoryCategory(str, Enum):
    """Phân loại memory"""
    PERSONAL_INFO = "personal_info"
    PREFERENCE = "preference"
    FACT = "fact"
    PLAN = "plan"
    RELATIONSHIP = "relationship"
    EXPERIENCE = "experience"


class MemoryStatus(str, Enum):
    """Trạng thái memory"""
    RAW = "raw"
    ACTIVE = "active"
    PROCESSED = "processed"
    ARCHIVED = "archived"


@dataclass
class ConversationStats:
    """Theo dõi statistics của conversation"""
    total_messages: int = 0
    liquid_count: int = 0
    crystal_count: int = 0
    solid_count: int = 0
    last_crystallize: int = 0  # Message count tại lần crystallize cuối
    last_evolve: int = 0  # Message count tại lần evolve cuối


@dataclass
class LiquidMessage:
    """Message trong Liquid Layer"""
    content: str
    role: str = "user"
    index: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    topic_shifted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "role": self.role,
            "index": self.index,
            "timestamp": self.timestamp,
            "topic_shifted": self.topic_shifted
        }


@dataclass
class CrystalFact:
    """Atomic Fact trong Crystal Layer"""
    content: str
    category: str = "fact"
    confidence: float = 0.8
    keywords: List[str] = field(default_factory=list)
    context_tags: List[str] = field(default_factory=list)
    related_to: List[str] = field(default_factory=list)
    source_message_ids: List[str] = field(default_factory=list)
    crystallized_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_metadata(self) -> Dict[str, Any]:
        return {
            "fcm_type": "crystal",
            "fcm_frequency": 2,
            "fcm_status": "active",
            "category": self.category,
            "confidence": self.confidence,
            "keywords": self.keywords,
            "context_tags": self.context_tags,
            "related_to": self.related_to,
            "source_messages": self.source_message_ids,
            "crystallized_at": self.crystallized_at,
        }


@dataclass
class SolidKnowledge:
    """Consolidated Knowledge trong Solid Layer"""
    content: str
    category: str = "fact"
    confidence: float = 0.9
    action: str = "NEW"  # NEW, UPDATE, IGNORE
    supersedes: Optional[str] = None
    reason: str = ""
    evolved_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_metadata(self) -> Dict[str, Any]:
        return {
            "fcm_type": "solid",
            "fcm_frequency": 3,
            "fcm_status": "active",
            "category": self.category,
            "confidence": self.confidence,
            "action": self.action,
            "supersedes": self.supersedes,
            "reason": self.reason,
            "evolved_at": self.evolved_at,
        }


@dataclass
class CompressionResult:
    """Kết quả nén hội thoại (SeCom)"""
    compressed_narrative: str
    key_entities: List[str] = field(default_factory=list)
    noise_count: int = 0
    original_length: int = 0
    compressed_length: int = 0
    
    @property
    def compression_ratio(self) -> float:
        if self.original_length == 0:
            return 1.0
        return self.compressed_length / self.original_length


@dataclass 
class TopicShiftResult:
    """Kết quả phát hiện Topic Shift"""
    is_new_topic: bool = False
    confidence: float = 0.0
    reason: str = ""

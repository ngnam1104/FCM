"""
FCM V2 Base Schemas
===================

Pydantic models cho data structures với các cải tiến:
1. Bi-Temporal support (valid_at + observed_at)
2. Memory Strength tracking (Active Forgetting)
3. Dynamic Persona
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum
import math


class MemoryCategory(str, Enum):
    """Phân loại memory"""
    PERSONAL_INFO = "personal_info"
    PREFERENCE = "preference"
    FACT = "fact"
    PLAN = "plan"
    RELATIONSHIP = "relationship"
    EXPERIENCE = "experience"
    INTERACTION_STYLE = "interaction_style"


class ChangeType(str, Enum):
    """Loại thay đổi (MAPLE)"""
    SUPPLEMENT = "supplement"
    REPLACEMENT = "replacement"
    CORRECTION = "correction"
    EVOLUTION = "evolution"


class MemoryStatus(str, Enum):
    """Trạng thái memory"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    COLD_STORAGE = "cold_storage"  # Cho Active Forgetting


# =============================================================================
# CẢI TIẾN 3: MEMORY STRENGTH (Active Forgetting - Ebbinghaus Curve)
# =============================================================================

class MemoryStrength(BaseModel):
    """
    Tracking sức mạnh ký ức theo công thức Ebbinghaus cải biên:
    
    S(t) = S_0 * e^(-Δt/τ) + R * N_access
    
    Trong đó:
    - S_0: Sức mạnh ban đầu (initial_strength)
    - Δt: Thời gian trôi qua kể từ lần cập nhật cuối
    - τ: Hằng số suy giảm (decay_constant, mặc định 7 ngày)
    - R: Hệ số củng cố (reinforcement_factor)
    - N_access: Số lần truy xuất (access_count)
    """
    initial_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    current_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    decay_constant: float = Field(default=7.0, description="Hằng số suy giảm (ngày)")
    reinforcement_factor: float = Field(default=0.1, description="Hệ số củng cố mỗi lần truy xuất")
    access_count: int = Field(default=0, ge=0)
    last_access_at: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def calculate_strength(self, current_time: Optional[datetime] = None) -> float:
        """
        Tính sức mạnh ký ức tại thời điểm hiện tại.
        
        Công thức: S(t) = S_0 * e^(-Δt/τ) + R * N_access
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Tính Δt (số ngày)
        delta_days = (current_time - self.last_access_at).total_seconds() / 86400
        
        # Công thức Ebbinghaus cải biên
        decay_term = self.initial_strength * math.exp(-delta_days / self.decay_constant)
        reinforcement_term = self.reinforcement_factor * self.access_count
        
        # Clamp về [0, 1]
        strength = min(1.0, max(0.0, decay_term + reinforcement_term))
        return strength
    
    def on_access(self) -> None:
        """Được gọi khi memory được truy xuất (retrieved)"""
        self.access_count += 1
        self.last_access_at = datetime.now()
        # Reset strength về 1.0 khi được truy xuất
        self.current_strength = 1.0
    
    def should_forget(self, threshold: float = 0.2) -> bool:
        """
        Kiểm tra có nên "quên" (move to cold storage) không.
        
        Args:
            threshold: Ngưỡng quên (mặc định 0.2)
        """
        current = self.calculate_strength()
        return current < threshold


# =============================================================================
# CẢI TIẾN 1: BI-TEMPORAL SCHEMA CHO CRYSTAL
# =============================================================================

class AtomicFact(BaseModel):
    """
    Atomic Fact với Bi-Temporal support.
    
    Bi-Temporal:
    - valid_at: Thời gian sự kiện XẢY RA (trích xuất từ text)
    - observed_at: Thời gian hệ thống GHI NHẬN
    
    Ví dụ: "Năm 2018 tôi làm ở Google"
    - valid_at: "2018" (khi sự kiện xảy ra)
    - observed_at: "2024-01-15" (khi bot nghe được)
    """
    id: Optional[str] = None
    content: str = Field(..., description="Nội dung fact")
    category: MemoryCategory = Field(default=MemoryCategory.FACT)
    
    # Bi-Temporal fields
    valid_at: Optional[str] = Field(
        default=None, 
        description="Thời gian sự kiện xảy ra (trích xuất từ text, có thể là '2018', '2018-2022', 'tuần trước')"
    )
    observed_at: datetime = Field(
        default_factory=datetime.now,
        description="Thời gian hệ thống ghi nhận"
    )
    
    # Confidence & Strength
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    memory_strength: MemoryStrength = Field(default_factory=MemoryStrength)
    
    # Zettelkasten linking (A-Mem)
    keywords: List[str] = Field(default_factory=list)
    context_tags: List[str] = Field(default_factory=list)
    related_to: List[str] = Field(default_factory=list)
    
    # Metadata
    source_message_ids: List[str] = Field(default_factory=list)
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict cho mem0"""
        return {
            "fcm_type": "crystal",
            "fcm_frequency": 2,
            "fcm_status": "active",
            "category": self.category.value,
            "valid_at": self.valid_at,
            "observed_at": self.observed_at.isoformat(),
            "confidence": self.confidence,
            "decay_score": self.memory_strength.current_strength,
            "access_count": self.memory_strength.access_count,
            "last_access_at": self.memory_strength.last_access_at.isoformat(),
            "keywords": self.keywords,
            "context_tags": self.context_tags,
            "related_to": self.related_to,
        }


class CrystalFact(AtomicFact):
    """Alias cho AtomicFact với Bi-Temporal support"""
    pass


# =============================================================================
# LIQUID MESSAGE
# =============================================================================

class LiquidMessage(BaseModel):
    """Raw message trong Liquid layer"""
    id: Optional[str] = None
    content: str
    role: Literal["user", "assistant", "system"] = "user"
    timestamp: datetime = Field(default_factory=datetime.now)
    message_index: int = 0
    is_attention_sink: bool = Field(
        default=False,
        description="Đánh dấu tin nhắn là Attention Sink (luôn giữ lại)"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Embedding vector cho Semantic Grouping"
    )
    
    def to_metadata(self) -> Dict[str, Any]:
        return {
            "fcm_type": "liquid",
            "fcm_frequency": 1,
            "fcm_status": "raw",
            "role": self.role,
            "message_index": self.message_index,
            "timestamp": self.timestamp.isoformat(),
            "is_attention_sink": self.is_attention_sink,
        }


# =============================================================================
# SOLID KNOWLEDGE
# =============================================================================

class SolidKnowledge(BaseModel):
    """Consolidated knowledge trong Solid layer"""
    id: Optional[str] = None
    content: str
    category: MemoryCategory = Field(default=MemoryCategory.FACT)
    
    # Version tracking (MAPLE)
    version: int = Field(default=1)
    supersedes: Optional[str] = Field(default=None, description="ID của fact cũ bị thay thế")
    superseded_by: Optional[str] = Field(default=None, description="ID của fact mới thay thế")
    change_type: Optional[ChangeType] = None
    change_reason: Optional[str] = None
    
    # Temporal
    validity_start: datetime = Field(default_factory=datetime.now)
    validity_end: Optional[datetime] = None
    
    # Strength
    memory_strength: MemoryStrength = Field(default_factory=MemoryStrength)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Status
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    
    def to_metadata(self) -> Dict[str, Any]:
        return {
            "fcm_type": "solid",
            "fcm_frequency": 3,
            "fcm_status": self.status.value,
            "profile_section": self.category.value,
            "version": self.version,
            "supersedes": self.supersedes,
            "change_type": self.change_type.value if self.change_type else None,
            "change_reason": self.change_reason,
            "validity_start": self.validity_start.isoformat(),
            "validity_end": self.validity_end.isoformat() if self.validity_end else None,
            "decay_score": self.memory_strength.current_strength,
            "access_count": self.memory_strength.access_count,
            "confidence": self.confidence,
        }


# =============================================================================
# CẢI TIẾN 4: DYNAMIC PERSONA
# =============================================================================

class UserPersona(BaseModel):
    """
    Dynamic User Persona - Cập nhật system prompt dựa trên interaction style.
    
    Trích xuất từ hội thoại:
    - Phong cách giao tiếp (formal/casual)
    - Độ dài phản hồi ưa thích
    - Chủ đề quan tâm
    - Giọng điệu (serious/playful)
    """
    user_id: str
    
    # Interaction Style
    communication_style: Literal["formal", "casual", "mixed"] = "mixed"
    preferred_response_length: Literal["brief", "detailed", "adaptive"] = "adaptive"
    humor_level: float = Field(default=0.5, ge=0.0, le=1.0, description="0=serious, 1=playful")
    
    # Preferences
    topics_of_interest: List[str] = Field(default_factory=list)
    avoided_topics: List[str] = Field(default_factory=list)
    
    # Language
    preferred_language: str = "vi"  # Vietnamese default
    use_emojis: bool = True
    
    # Relationship level (tăng theo thời gian)
    interaction_count: int = 0
    familiarity_level: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="Mức độ thân thiết: 0=stranger, 1=close friend"
    )
    
    # Inferred traits
    inferred_traits: List[str] = Field(default_factory=list)
    
    def to_system_prompt_injection(self) -> str:
        """
        Tạo đoạn text để inject vào system prompt.
        """
        lines = []
        
        # Style
        if self.communication_style == "casual":
            lines.append("- User thích giao tiếp thoải mái, có thể xưng hô thân mật")
        elif self.communication_style == "formal":
            lines.append("- User thích giao tiếp lịch sự, trang trọng")
        
        # Length
        if self.preferred_response_length == "brief":
            lines.append("- User thích câu trả lời ngắn gọn, đi vào trọng tâm")
        elif self.preferred_response_length == "detailed":
            lines.append("- User thích câu trả lời chi tiết, giải thích kỹ")
        
        # Humor
        if self.humor_level > 0.7:
            lines.append("- User thích đùa, có thể dùng humor trong câu trả lời")
        elif self.humor_level < 0.3:
            lines.append("- User thích nghiêm túc, tránh đùa giỡn")
        
        # Familiarity
        if self.familiarity_level > 0.7:
            lines.append("- Đây là user quen thuộc, có thể thoải mái hơn")
        
        # Topics
        if self.topics_of_interest:
            lines.append(f"- Chủ đề quan tâm: {', '.join(self.topics_of_interest[:5])}")
        
        # Traits
        if self.inferred_traits:
            lines.append(f"- Đặc điểm: {', '.join(self.inferred_traits[:5])}")
        
        return "\n".join(lines) if lines else ""
    
    def update_familiarity(self, increment: float = 0.05) -> None:
        """Tăng mức độ thân thiết sau mỗi interaction"""
        self.interaction_count += 1
        self.familiarity_level = min(1.0, self.familiarity_level + increment)


# =============================================================================
# RESULT SCHEMAS
# =============================================================================

class TopicShiftResult(BaseModel):
    """Kết quả phát hiện Topic Shift"""
    is_new_topic: bool = False
    confidence: float = 0.0
    old_topic: Optional[str] = None
    new_topic: Optional[str] = None
    reason: Optional[str] = None
    # Cải tiến: Sử dụng embedding similarity trước
    embedding_similarity: Optional[float] = None
    used_llm: bool = False  # True nếu đã gọi LLM


class CompressionResult(BaseModel):
    """Kết quả nén hội thoại"""
    compressed_narrative: str
    key_entities: List[str] = Field(default_factory=list)
    noise_count: int = 0
    original_length: int = 0
    compressed_length: int = 0
    compression_ratio: float = 0.0


class SearchResult(BaseModel):
    """Kết quả tìm kiếm với Weighted Ensemble"""
    query: str
    strategy: str
    
    # Results by layer
    solid_results: List[Dict[str, Any]] = Field(default_factory=list)
    crystal_results: List[Dict[str, Any]] = Field(default_factory=list)
    liquid_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Combined with weighted scores
    combined_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    best_source: Optional[str] = None
    total_results: int = 0
    
    # Weighted scores applied
    weights_used: Dict[str, float] = Field(default_factory=lambda: {
        "solid": 0.5,
        "crystal": 0.3,
        "liquid": 0.2
    })

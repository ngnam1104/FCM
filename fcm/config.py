"""
FCM Configuration
=================

Cấu hình cho Frequency-based Crystallizing Memory system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class FCMConfig:
    """Cấu hình cho FCM Agent"""
    
    # === Memory Layer Configuration ===
    buffer_size: int = 5  # Số tin nhắn Liquid trước khi trigger crystallize
    crystallize_threshold: int = 5  # Trigger crystallize sau N tin nhắn
    evolve_threshold: int = 20  # Trigger evolve sau M tin nhắn hoặc cuối session
    
    # === Memory Type Tags ===
    liquid_type: str = "liquid"
    crystal_type: str = "crystal"
    solid_type: str = "solid"
    
    # === Retrieval Configuration ===
    solid_search_limit: int = 5  # Số lượng Solid memories tối đa khi search
    crystal_search_limit: int = 5  # Số lượng Crystal memories tối đa
    liquid_search_limit: int = 3  # Số lượng Liquid memories tối đa
    hybrid_score_threshold: float = 0.7  # Ngưỡng score để quyết định có search tiếp không
    
    # === Enhanced Retrieval Configuration (Shared with V2) ===
    retrieval_weight_solid: float = 0.5
    retrieval_weight_crystal: float = 0.3
    retrieval_weight_liquid: float = 0.2
    parallel_search: bool = True
    enable_temporal_priority: bool = True
    
    # === LLM Configuration ===
    llm_provider: str = "groq"  # groq, openai, gemini
    llm_model: str = "llama-3.3-70b-versatile"  # 70B model cho extraction tốt hơn
    temperature: float = 0.1  # Temperature thấp cho task extraction
    max_tokens: int = 4000
    
    # === Embedder Configuration ===
    embedder_provider: str = "huggingface"
    embedder_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dims: int = 768
    
    # === Vector Store Configuration ===
    vector_store_provider: str = "qdrant"
    collection_name: str = "fcm_memory"
    vector_store_path: str = "./fcm_data"
    
    # === Session Configuration ===
    default_user_id: str = "default_user"
    history_db_path: str = "./fcm_history"
    
    # === Debug ===
    verbose: bool = True
    
    def get_retrieval_weights(self) -> Dict[str, float]:
        """Lấy trọng số retrieval đã normalize (shared with V2)"""
        total = self.retrieval_weight_solid + self.retrieval_weight_crystal + self.retrieval_weight_liquid
        return {
            "solid": self.retrieval_weight_solid / total,
            "crystal": self.retrieval_weight_crystal / total,
            "liquid": self.retrieval_weight_liquid / total,
        }
    
    def to_mem0_config(self) -> Dict[str, Any]:
        """Chuyển đổi FCMConfig thành config dict cho mem0"""
        return {
            "llm": {
                "provider": self.llm_provider,
                "config": {
                    "model": self.llm_model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
            },
            "embedder": {
                "provider": self.embedder_provider,
                "config": {
                    "model": self.embedder_model,
                    "embedding_dims": self.embedding_dims,
                }
            },
            "vector_store": {
                "provider": self.vector_store_provider,
                "config": {
                    "collection_name": self.collection_name,
                    "path": self.vector_store_path,
                    "embedding_model_dims": self.embedding_dims,
                }
            },
            "history_db_path": self.history_db_path,
            "version": "v1.1",
        }


def get_default_fcm_config() -> FCMConfig:
    """Trả về cấu hình FCM mặc định với Groq (miễn phí)"""
    return FCMConfig()


def get_openai_fcm_config() -> FCMConfig:
    """Cấu hình FCM với OpenAI"""
    return FCMConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        embedder_provider="openai",
        embedder_model="text-embedding-3-small",
        embedding_dims=1536,
    )


def get_gemini_fcm_config() -> FCMConfig:
    """Cấu hình FCM với Google Gemini"""
    return FCMConfig(
        llm_provider="gemini",
        llm_model="gemini-1.5-flash",
        embedder_provider="huggingface",
        embedder_model="sentence-transformers/all-mpnet-base-v2",
        embedding_dims=768,
    )

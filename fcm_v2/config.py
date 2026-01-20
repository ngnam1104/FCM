"""
FCM V2 Configuration
====================

Cấu hình mở rộng với các tham số mới cho:
1. Bi-Temporal queries
2. Attention Sinks
3. Active Forgetting (Ebbinghaus)
4. Dynamic Persona
5. Weighted Ensemble Retrieval
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class FCMConfigV2:
    """Cấu hình cho FCM Agent V2"""
    
    # === Memory Layer Configuration ===
    buffer_size: int = 5
    crystallize_threshold: int = 5
    evolve_threshold: int = 20
    
    # === Memory Type Tags ===
    liquid_type: str = "liquid"
    crystal_type: str = "crystal"
    solid_type: str = "solid"
    
    # === CẢI TIẾN 2: ATTENTION SINKS ===
    # Số tin nhắn đầu tiên luôn giữ lại (System prompt + Greeting)
    attention_sink_count: int = 3
    # Ngưỡng cosine similarity để gọi LLM check topic shift
    topic_shift_embedding_threshold: float = 0.6
    
    # === CẢI TIẾN 3: ACTIVE FORGETTING (Ebbinghaus) ===
    # Hằng số suy giảm τ (ngày)
    decay_constant_days: float = 7.0
    # Hệ số củng cố R khi memory được truy xuất
    reinforcement_factor: float = 0.1
    # Ngưỡng quên - dưới ngưỡng này sẽ move to cold storage
    forget_threshold: float = 0.2
    # Có bật Active Forgetting không
    enable_active_forgetting: bool = True
    # Path cho Cold Storage (file log)
    cold_storage_path: str = "./fcm_cold_storage"
    
    # === CẢI TIẾN 4: DYNAMIC PERSONA ===
    enable_dynamic_persona: bool = True
    # Mức tăng familiarity sau mỗi interaction
    familiarity_increment: float = 0.05
    
    # === CẢI TIẾN 5: WEIGHTED ENSEMBLE RETRIEVAL ===
    # Trọng số cho từng layer trong final score
    retrieval_weight_solid: float = 0.5
    retrieval_weight_crystal: float = 0.3
    retrieval_weight_liquid: float = 0.2
    # Có search song song không
    parallel_search: bool = True
    
    # === Retrieval Configuration ===
    solid_search_limit: int = 5
    crystal_search_limit: int = 5
    liquid_search_limit: int = 3
    hybrid_score_threshold: float = 0.7
    
    # === CẢI TIẾN 1: BI-TEMPORAL ===
    # Có ưu tiên theo valid_at khi query có context thời gian không
    enable_temporal_priority: bool = True
    
    # === LLM Configuration ===
    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"  # 8B model nhanh, ít rate limit
    temperature: float = 0.1  # Thấp hơn cho JSON output ổn định
    max_tokens: int = 4000
    
    # === Semantic Similarity ===
    semantic_similarity_threshold: float = 0.75  # Ngưỡng để skip LLM call
    
    # === Embedder Configuration ===
    embedder_provider: str = "huggingface"
    embedder_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dims: int = 768
    
    # === Vector Store Configuration ===
    vector_store_provider: str = "qdrant"
    collection_name: str = "fcm_memory_v2"
    vector_store_path: str = "./fcm_data_v2"
    
    # === Session Configuration ===
    default_user_id: str = "default_user"
    history_db_path: str = "./fcm_history_v2"
    
    # === Debug ===
    verbose: bool = True
    
    def get_retrieval_weights(self) -> Dict[str, float]:
        """Lấy trọng số retrieval đã normalize"""
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
            "version": "v2.0",
        }


def get_default_config_v2() -> FCMConfigV2:
    """Trả về cấu hình FCM V2 mặc định với Groq (miễn phí)"""
    return FCMConfigV2()


def get_openai_config_v2() -> FCMConfigV2:
    """Cấu hình FCM V2 với OpenAI"""
    return FCMConfigV2(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        embedder_provider="openai",
        embedder_model="text-embedding-3-small",
        embedding_dims=1536,
    )


def get_ollama_config_v2() -> FCMConfigV2:
    """Cấu hình FCM V2 với Ollama (local)"""
    return FCMConfigV2(
        llm_provider="ollama",
        llm_model="llama3.1:8b",
        embedder_provider="huggingface",
        embedder_model="sentence-transformers/all-mpnet-base-v2",
        embedding_dims=768,
    )

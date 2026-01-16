"""
Config sử dụng Groq LLM (miễn phí, không tốn GPU)
"""

def get_groq_config():
    """
    Cấu hình Groq LLM (hoàn toàn miễn phí)
    ✅ Combo an toàn: Groq LLM + HuggingFace embedder (768-dim)
    """
    return {
        "llm": {
            "provider": "groq",
            "config": {
                "model": "llama-3.1-8b-instant",  # Model nhẹ, tốc độ cao
                "temperature": 0.7,
                "max_tokens": 8000,
            }
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-mpnet-base-v2",  # 768 dim - PUBLIC
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0",
                "path": "./mem0_data",
                "embedding_model_dims": 768,  # 🔴 CRITICAL: Must be 768 for all-mpnet-base-v2
            }
        },
        "history_db_path": "./mem0_history",
        "version": "v1.0",
    }


def get_gemini_config():
    """
    Cấu hình Google Gemini (miễn phí, giới hạn 60 request/phút)
    """
    return {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-1.5-flash",
                "temperature": 0.7,
                "max_tokens": 8000,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "mem0",
                "path": "./mem0_data",
            }
        },
        "history_db_path": "./mem0_history",
        "version": "v1.0",
    }

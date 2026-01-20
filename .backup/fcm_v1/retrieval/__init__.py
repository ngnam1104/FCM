"""
FCM V1 Retrieval Module
=======================

Shared Enhanced Retriever với FCM V2 để so sánh công bằng kiến trúc nhớ.

Pipeline:
1. Preprocessing (Query Anonymization & Analysis)
2. Wide Retrieve (Top-20 từ mỗi layer)
3. Post-processing (Filter & Rerank)
4. Final Scoring & Deduplication
"""

from fcm_v2.retrieval.enhanced_retriever import EnhancedRetriever

__all__ = ["EnhancedRetriever"]

"""
FCM V2 Retrieval Module
=======================

Enhanced Retrieval Pipeline:
1. Preprocessing (Query Anonymization & Analysis)
2. Wide Retrieve (Top-20 từ mỗi layer)
3. Post-processing (Filter & Rerank)
4. Final Scoring & Deduplication

Weighted Ensemble Retrieval với công thức:
Score_final = (w_s * S_solid) + (w_c * S_crystal) + (w_l * S_liquid)

Mặc định: w_s=0.5, w_c=0.3, w_l=0.2
"""

from fcm.v2.retrieval.weighted_retriever import WeightedRetriever
from fcm.v2.retrieval.enhanced_retriever import EnhancedRetriever

__all__ = ["WeightedRetriever", "EnhancedRetriever"]

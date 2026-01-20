"""
FCM V2 Weighted Ensemble Retriever
===================================

Cải tiến 5: Weighted Ensemble Retrieval

Thay vì fallback tuần tự (Solid → Crystal → Liquid),
search song song trên cả 3 layers và áp dụng trọng số:

Score_final = (w_s * S_solid) + (w_c * S_crystal) + (w_l * S_liquid)

Mặc định: w_s=0.5, w_c=0.3, w_l=0.2 (ưu tiên Solid đã kiểm chứng)

Lợi ích:
- Không bỏ sót thông tin từ các layer khác
- Kết hợp được context gần đây (Liquid) với knowledge đã xác thực (Solid)
- Score cuối cùng phản ánh đúng độ tin cậy
"""

import logging
import re
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from fcm.v2.schemas.base import SearchResult
from fcm.v1.utils import calculate_keyword_boost

logger = logging.getLogger(__name__)


class WeightedRetriever:
    """
    Weighted Ensemble Retriever
    
    Search song song trên 3 layers và normalize + weight scores
    """
    
    def __init__(self, memory, config, user_id: str, verbose: bool = True):
        """
        Args:
            memory: mem0 Memory instance
            config: FCMConfigV2
            user_id: User ID
            verbose: Có log không
        """
        self.memory = memory
        self.config = config
        self.user_id = user_id
        self.verbose = verbose
        
        # Lấy weights từ config
        self.weights = config.get_retrieval_weights()
        
    def _log(self, message: str, level: str = "info"):
        """Helper logging"""
        if self.verbose:
            prefix = "[Retrieval]"
            if level == "info":
                logger.info(f"{prefix} {message}")
                print(f"{prefix} {message}")
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores về 0-1
        """
        if not results:
            return []
        
        scores = [r.get("score", 0) for r in results]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        
        # Avoid division by zero
        score_range = max_score - min_score
        if score_range == 0:
            score_range = 1.0
        
        for r in results:
            original_score = r.get("score", 0)
            normalized = (original_score - min_score) / score_range
            r["normalized_score"] = normalized
        
        return results
    
    def _calculate_keyword_boost(self, query: str, memory_content: str) -> float:
        """
        Tính keyword boost score cho memory dựa trên query.
        Sử dụng function từ fcm.utils để đảm bảo consistency giữa V1 và V2.
        
        Args:
            query: Câu hỏi/truy vấn
            memory_content: Nội dung memory
            
        Returns:
            Boost score (0.0 đến 0.4)
        """
        return calculate_keyword_boost(query, memory_content)
    
    def _search_layer(
        self,
        query: str,
        fcm_type: str,
        limit: int,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search trong một layer cụ thể với client-side filtering
        """
        try:
            all_results = self.memory.search(
                query,
                user_id=self.user_id,
                limit=limit * 4
            )
            
            layer_results = []
            for mem in all_results.get("results", []):
                metadata = mem.get("metadata", {})
                
                # Filter by type
                if metadata.get("fcm_type") != fcm_type:
                    continue
                
                # Filter archived
                if not include_archived:
                    status = metadata.get("fcm_status", "active")
                    if status == "archived":
                        continue
                    content = mem.get("memory", "")
                    if content.startswith("[ARCHIVED]"):
                        continue
                
                layer_results.append(mem)
            
            return layer_results[:limit]
            
        except Exception as e:
            self._log(f"Error searching {fcm_type}: {e}", "error")
            return []
    
    def search_parallel(
        self,
        query: str,
        solid_limit: Optional[int] = None,
        crystal_limit: Optional[int] = None,
        liquid_limit: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search song song trên cả 3 layers
        
        Returns:
            Dict với results từ mỗi layer
        """
        solid_limit = solid_limit or self.config.solid_search_limit
        crystal_limit = crystal_limit or self.config.crystal_search_limit
        liquid_limit = liquid_limit or self.config.liquid_search_limit
        
        results = {
            "solid": [],
            "crystal": [],
            "liquid": []
        }
        
        if self.config.parallel_search:
            # Search song song
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self._search_layer, query, "solid", solid_limit): "solid",
                    executor.submit(self._search_layer, query, "crystal", crystal_limit): "crystal",
                    executor.submit(self._search_layer, query, "liquid", liquid_limit): "liquid",
                }
                
                for future in as_completed(futures):
                    layer = futures[future]
                    try:
                        results[layer] = future.result()
                    except Exception as e:
                        self._log(f"Error in {layer} search: {e}", "error")
        else:
            # Search tuần tự (fallback)
            results["solid"] = self._search_layer(query, "solid", solid_limit)
            results["crystal"] = self._search_layer(query, "crystal", crystal_limit)
            results["liquid"] = self._search_layer(query, "liquid", liquid_limit)
        
        return results
    
    def apply_weighted_scores(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        query: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Cải tiến 5: Áp dụng Weighted Ensemble scores + Keyword Boost
        
        Công thức: Score_final = (w_s * S_solid) + (w_c * S_crystal) + (w_l * S_liquid) + keyword_boost
        
        Args:
            results: Dict với results từ mỗi layer
            query: Query string for keyword boosting (optional)
            
        Returns:
            Combined list với weighted_score
        """
        combined = []
        
        # Process mỗi layer
        for layer, layer_results in results.items():
            if not layer_results:
                continue
            
            # Normalize scores trong layer
            normalized = self._normalize_scores(layer_results.copy())
            
            # Apply weight
            weight = self.weights.get(layer, 0.2)
            
            for result in normalized:
                normalized_score = result.get("normalized_score", 0)
                weighted_score = weight * normalized_score
                
                # Add metadata
                result["source_layer"] = layer
                result["layer_weight"] = weight
                result["weighted_score"] = weighted_score
                
                combined.append(result)
        
        # Deduplicate by content
        seen_contents = set()
        unique_results = []
        
        for r in combined:
            content = r.get("memory", "")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_results.append(r)
            else:
                # Nếu trùng, cộng weighted_score
                for existing in unique_results:
                    if existing.get("memory") == content:
                        existing["weighted_score"] += r["weighted_score"]
                        existing["source_layer"] += f"+{r['source_layer']}"
                        break
        
        # Apply keyword boost if query provided
        if query:
            for r in unique_results:
                content = r.get("memory", "")
                keyword_boost = self._calculate_keyword_boost(query, content)
                if keyword_boost > 0:
                    r["keyword_boost"] = keyword_boost
                    r["original_weighted_score"] = r["weighted_score"]
                    r["weighted_score"] += keyword_boost
                    self._log(f"Keyword boost +{keyword_boost:.3f} for: {content[:50]}...")
        
        # Sort by weighted_score descending
        unique_results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        
        return unique_results
    
    def search(
        self,
        query: str,
        strategy: str = "weighted",
        limit: int = 10,
        temporal_context: Optional[str] = None
    ) -> SearchResult:
        """
        Main search method với multiple strategies
        
        Strategies:
        - "weighted": Weighted Ensemble (Cải tiến 5)
        - "hybrid": Fallback cũ (Solid → Crystal → Liquid)
        - "solid_first": Ưu tiên Solid
        - "all_layers": Search tất cả không weight
        
        Args:
            query: Câu truy vấn
            strategy: Chiến lược search
            limit: Số kết quả tối đa
            temporal_context: Context thời gian cho Bi-Temporal boost
            
        Returns:
            SearchResult
        """
        self._log(f"Query: '{query[:50]}...' | Strategy: {strategy}")
        
        # Search song song
        raw_results = self.search_parallel(query)
        
        if strategy == "weighted":
            # Cải tiến 5: Weighted Ensemble + Keyword Boost
            combined = self.apply_weighted_scores(raw_results, query)
            
            # Bi-Temporal boost nếu có temporal_context
            if temporal_context and self.config.enable_temporal_priority:
                combined = self._apply_temporal_boost(combined, temporal_context)
            
            best_source = self._determine_best_source(raw_results)
            
        elif strategy == "hybrid":
            # Fallback cũ
            combined = self._hybrid_fallback(raw_results)
            best_source = "hybrid"
            
        elif strategy == "solid_first":
            combined = raw_results["solid"] + raw_results["crystal"]
            best_source = "solid" if raw_results["solid"] else "crystal"
            
        else:  # all_layers
            combined = raw_results["solid"] + raw_results["crystal"] + raw_results["liquid"]
            best_source = "all"
        
        # Limit results
        combined = combined[:limit]
        
        self._log(f"Found {len(combined)} results (Best: {best_source})")
        
        return SearchResult(
            query=query,
            strategy=strategy,
            solid_results=raw_results["solid"],
            crystal_results=raw_results["crystal"],
            liquid_results=raw_results["liquid"],
            combined_results=combined,
            best_source=best_source,
            total_results=len(combined),
            weights_used=self.weights
        )
    
    def _apply_temporal_boost(
        self,
        results: List[Dict[str, Any]],
        temporal_context: str
    ) -> List[Dict[str, Any]]:
        """
        Apply Bi-Temporal boost cho results có valid_at khớp
        """
        for r in results:
            valid_at = r.get("metadata", {}).get("valid_at", "")
            
            if valid_at and temporal_context.lower() in str(valid_at).lower():
                # Boost 20%
                current_score = r.get("weighted_score", 0)
                r["weighted_score"] = current_score * 1.2
                r["temporal_match"] = True
            else:
                r["temporal_match"] = False
        
        # Re-sort
        results.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        
        return results
    
    def _determine_best_source(
        self,
        results: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Xác định source tốt nhất dựa trên số lượng và scores"""
        scores = {
            "solid": sum(r.get("score", 0) for r in results["solid"]),
            "crystal": sum(r.get("score", 0) for r in results["crystal"]),
            "liquid": sum(r.get("score", 0) for r in results["liquid"])
        }
        
        # Weight scores
        weighted = {
            k: v * self.weights.get(k, 0.2)
            for k, v in scores.items()
        }
        
        if not any(weighted.values()):
            return "none"
        
        return max(weighted.keys(), key=lambda k: weighted[k])
    
    def _hybrid_fallback(
        self,
        results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Fallback logic cũ: Solid → Crystal → Liquid"""
        combined = []
        
        # Solid first
        combined.extend(results["solid"])
        
        # Check threshold
        if combined and combined[0].get("score", 0) >= self.config.hybrid_score_threshold:
            return combined
        
        # Add Crystal
        combined.extend(results["crystal"])
        
        if combined and max(r.get("score", 0) for r in combined) >= self.config.hybrid_score_threshold:
            return combined
        
        # Add Liquid
        combined.extend(results["liquid"])
        
        # Deduplicate and sort
        seen = set()
        unique = []
        for r in combined:
            content = r.get("memory", "")
            if content not in seen:
                seen.add(content)
                unique.append(r)
        
        unique.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return unique
    
    def reinforce_accessed_memories(
        self,
        results: List[Dict[str, Any]],
        solid_layer
    ):
        """
        Củng cố các memories đã được truy xuất (Active Forgetting)
        
        Args:
            results: Kết quả search
            solid_layer: SolidLayer instance để update
        """
        for r in results:
            memory_id = r.get("id")
            fcm_type = r.get("metadata", {}).get("fcm_type")
            
            # Chỉ reinforce solid memories
            if memory_id and fcm_type == "solid":
                solid_layer.on_memory_accessed(memory_id)

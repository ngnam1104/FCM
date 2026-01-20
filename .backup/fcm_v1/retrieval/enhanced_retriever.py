"""
FCM V2 Enhanced Retriever - Advanced Retrieval Pipeline
=========================================================

Nâng cấp WeightedRetriever với pipeline:

1. Preprocessing (Query Anonymization)
   - Thay tên riêng thành placeholder: "Nam học gì?" ↁE"Người dùng học gì?"
   
2. Retrieve (Parallel + Wider Search)
   - Lấy Top-20 candidates thay vì Top-5 đềEtránh sót
   
3. Post-processing (Rerank & Filter)
   - Keyword Filter: Loại bềEfacts không liên quan
   - Semantic Rerank: Cross-Encoder hoặc similarity scoring
   - Content-Type Matching: Khớp loại query với loại fact
   
4. Final Scoring
   - Kết hợp semantic score + keyword score + relevance score
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from fcm.schemas.base import SearchResult
from fcm.utils import calculate_keyword_boost

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    """Phân tích query đềEhiểu intent"""
    original_query: str
    cleaned_query: str
    query_type: str  # "numeric", "entity", "temporal", "general"
    expected_content_patterns: List[str]
    extracted_entities: List[str]
    is_asking_name: bool
    is_asking_number: bool
    is_asking_location: bool
    is_asking_time: bool


class EnhancedRetriever:
    """
    Enhanced Retriever với Pipeline xử lý nâng cao
    
    Pipeline:
    1. Preprocess ↁE2. Wide Retrieve ↁE3. Filter & Rerank ↁE4. Final Score
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
        
        # Retrieval weights
        self.weights = config.get_retrieval_weights()
        
        # Query patterns for analysis
        self._init_query_patterns()
        
    def _log(self, message: str, level: str = "info"):
        """Helper logging"""
        if self.verbose:
            prefix = "[Enhanced Retrieval]"
            if level == "info":
                logger.info(f"{prefix} {message}")
                print(f"{prefix} {message}")
            elif level == "debug":
                logger.debug(f"{prefix} {message}")
                if self.verbose:
                    print(f"{prefix} [DEBUG] {message}")
    
    def _init_query_patterns(self):
        """Khoi tao cac patterns de phan tich query"""
        # Patterns hoi ve so, nam
        self.numeric_patterns = [
            r"sinh nam", r"nam sinh", r"bao nhieu tuoi", r"tuoi",
            r"nam nao", r"may tuoi", r"born in", r"year",
            r"bao nhieu", r"may", r"so"
        ]
        
        # Patterns hoi ve ten
        self.name_patterns = [
            r"ten la gi", r"ten gi", r"goi la gi", r"ai",
            r"what.*name", r"who"
        ]
        
        # Patterns hỏi vềEđịa điểm (Vietnamese + English)
        self.location_patterns = [
            r"ềEđâu", r"đâu", r"nơi nào", r"địa chềE,
            r"where", r"location", r"place", r"city", r"country", r"region"
        ]
        
        # Patterns hỏi vềEthời gian (Vietnamese + English)
        self.temporal_patterns = [
            r"khi nào", r"bao giềE, r"lúc nào", r"năm nào",
            r"when", r"time", r"year", r"date", r"month", r"day"
        ]
        
        # Common user names/references to anonymize
        self.user_references = [
            "nam", "minh", "lan", "hùng", "tôi", "bạn", "anh", "chềE,
            "em", "user", "người dùng"
        ]
    
    # =========================================================================
    # STEP 1: PREPROCESSING (Query Anonymization & Analysis)
    # =========================================================================
    
    def preprocess_query(self, query: str) -> QueryAnalysis:
        """
        Bước 1: Tiền xử lý query
        
        - Anonymize: "Nam sinh năm bao nhiêu?" ↁE"Người dùng sinh năm bao nhiêu?"
        - Phân tích loại query
        - Trích xuất expected content patterns
        """
        query_lower = query.lower()
        
        # Anonymize query - thay tên riêng bằng "người dùng"
        cleaned_query = query
        extracted_entities = []
        
        for name in self.user_references:
            pattern = rf'\b{name}\b'
            if re.search(pattern, query_lower):
                extracted_entities.append(name)
                # Thay thế nhưng giữ case sensitivity
                cleaned_query = re.sub(pattern, "người dùng", cleaned_query, flags=re.IGNORECASE)
        
        # Phân tích loại query
        is_asking_number = any(re.search(p, query_lower) for p in self.numeric_patterns)
        is_asking_name = any(re.search(p, query_lower) for p in self.name_patterns)
        is_asking_location = any(re.search(p, query_lower) for p in self.location_patterns)
        is_asking_time = any(re.search(p, query_lower) for p in self.temporal_patterns)
        
        # Xác định query type
        if is_asking_number:
            query_type = "numeric"
        elif is_asking_name:
            query_type = "entity"
        elif is_asking_time:
            query_type = "temporal"
        elif is_asking_location:
            query_type = "location"
        else:
            query_type = "general"
        
        # Expected content patterns dựa trên query
        expected_patterns = self._get_expected_patterns(query_lower, query_type)
        
        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned_query,
            query_type=query_type,
            expected_content_patterns=expected_patterns,
            extracted_entities=extracted_entities,
            is_asking_name=is_asking_name,
            is_asking_number=is_asking_number,
            is_asking_location=is_asking_location,
            is_asking_time=is_asking_time
        )
    
    def _get_expected_patterns(self, query_lower: str, query_type: str) -> List[str]:
        """Xác định patterns mà kết quả nên chứa"""
        patterns = []
        
        # Dựa trên query type
        if query_type == "numeric":
            patterns.extend([r'\d{4}', r'\d+'])  # Năm hoặc sềE        
        # Dựa trên keywords trong query
        keyword_to_pattern = {
            # Năm sinh
            ("sinh năm", "năm sinh", "tuổi"): [r'sinh.*\d{4}', r'\d{4}', r'năm.*\d{4}'],
            # Ngành học
            ("học ngành", "ngành gì", "chuyên ngành", "major"): [
                r'ngành', r'khmt', r'cntt', r'công nghềEthông tin', 
                r'khoa học máy tính', r'chuyên ngành', r'học'
            ],
            # Ngôn ngữ lập trình
            ("lập trình", "programming", "ngôn ngữ", "code"): [
                r'python', r'java', r'javascript', r'c\+\+', r'lập trình', r'code'
            ],
            # Thích/sềEthích
            ("thích", "sềEthích", "hobby", "yêu thích"): [
                r'thích', r'yêu thích', r'passion', r'hobby', r'sềEthích'
            ],
            # Nơi làm việc
            ("làm ềE, "công ty", "work", "job"): [
                r'làm.*ềE, r'công ty', r'work', r'thực tập', r'intern'
            ],
        }
        
        for keywords, related_patterns in keyword_to_pattern.items():
            if any(kw in query_lower for kw in keywords):
                patterns.extend(related_patterns)
        
        return patterns
    
    # =========================================================================
    # STEP 2: WIDE RETRIEVE (Top-20 từ mỗi layer)
    # =========================================================================
    
    def _search_layer_wide(
        self,
        query: str,
        fcm_type: str,
        limit: int = 20,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search trong một layer với limit rộng hơn
        """
        try:
            # Lấy nhiều hơn đềEfilter sau
            all_results = self.memory.search(
                query,
                user_id=self.user_id,
                limit=limit * 3  # Lấy 3x đềEcó đủ sau khi filter
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
                
                if len(layer_results) >= limit:
                    break
            
            return layer_results
            
        except Exception as e:
            self._log(f"Error searching {fcm_type}: {e}", "error")
            return []
    
    def retrieve_wide(
        self,
        query: str,
        analysis: QueryAnalysis
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Bước 2: Retrieve rộng từ tất cả layers (Top-20 mỗi layer)
        """
        # Sử dụng cleaned query đềEsearch
        search_query = analysis.cleaned_query
        
        results = {
            "solid": [],
            "crystal": [],
            "liquid": []
        }
        
        # Lấy limits rộng hơn (x2 so với default)
        solid_limit = self.config.solid_search_limit * 4  # 20
        crystal_limit = self.config.crystal_search_limit * 4  # 20
        liquid_limit = self.config.liquid_search_limit * 3  # 9
        
        if self.config.parallel_search:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self._search_layer_wide, search_query, "solid", solid_limit): "solid",
                    executor.submit(self._search_layer_wide, search_query, "crystal", crystal_limit): "crystal",
                    executor.submit(self._search_layer_wide, search_query, "liquid", liquid_limit): "liquid",
                }
                
                for future in as_completed(futures):
                    layer = futures[future]
                    try:
                        results[layer] = future.result()
                    except Exception as e:
                        self._log(f"Error in {layer} search: {e}", "error")
        else:
            results["solid"] = self._search_layer_wide(search_query, "solid", solid_limit)
            results["crystal"] = self._search_layer_wide(search_query, "crystal", crystal_limit)
            results["liquid"] = self._search_layer_wide(search_query, "liquid", liquid_limit)
        
        total = sum(len(v) for v in results.values())
        self._log(f"Wide retrieve: {total} candidates (S:{len(results['solid'])}, C:{len(results['crystal'])}, L:{len(results['liquid'])})")
        
        return results
    
    # =========================================================================
    # STEP 3: POST-PROCESSING (Filter & Rerank)
    # =========================================================================
    
    def filter_and_rerank(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        analysis: QueryAnalysis
    ) -> List[Dict[str, Any]]:
        """
        Bước 3: Filter và Rerank kết quả
        
        - Keyword Filter: Loại bềEfacts không match expected patterns
        - Content Relevance: Tính điểm relevance dựa trên query analysis
        - Semantic Rerank: Sắp xếp lại theo combined score
        """
        combined = []
        
        for layer, layer_results in results.items():
            weight = self.weights.get(layer, 0.2)
            
            for result in layer_results:
                content = result.get("memory", "")
                content_lower = content.lower()
                
                # Calculate multiple scores
                scores = self._calculate_relevance_scores(
                    content, content_lower, analysis
                )
                
                # Skip if completely irrelevant (use lenient threshold)
                # NOTE: Set to 0.0 to disable filter - let semantic similarity work
                if scores["total_relevance"] < 0.0:
                    continue
                
                # Add metadata
                result["source_layer"] = layer
                result["layer_weight"] = weight
                result["relevance_scores"] = scores
                
                # Combined score
                base_score = result.get("score", 0.5)
                final_score = (
                    base_score * 0.3 +  # Semantic similarity (30%)
                    scores["keyword_match"] * 0.25 +  # Keyword match (25%)
                    scores["pattern_match"] * 0.25 +  # Pattern match (25%)
                    weight * 0.2  # Layer weight (20%)
                )
                
                result["enhanced_score"] = final_score
                combined.append(result)
        
        # Sort by enhanced score
        combined.sort(key=lambda x: x.get("enhanced_score", 0), reverse=True)
        
        self._log(f"After filter: {len(combined)} candidates")
        
        return combined
    
    def _calculate_relevance_scores(
        self,
        content: str,
        content_lower: str,
        analysis: QueryAnalysis
    ) -> Dict[str, float]:
        """
        Tính các điểm relevance cho một memory
        """
        scores = {
            "keyword_match": 0.0,
            "pattern_match": 0.0,
            "type_match": 0.0,
            "total_relevance": 0.0
        }
        
        # 1. Keyword Match Score
        query_lower = analysis.original_query.lower()
        keyword_boost = calculate_keyword_boost(query_lower, content_lower)
        scores["keyword_match"] = min(keyword_boost * 2, 1.0)  # Scale up
        
        # 2. Pattern Match Score
        pattern_matches = 0
        for pattern in analysis.expected_content_patterns:
            if re.search(pattern, content_lower):
                pattern_matches += 1
        
        if analysis.expected_content_patterns:
            scores["pattern_match"] = min(pattern_matches / len(analysis.expected_content_patterns), 1.0)
        
        # 3. Type Match Score (Query type vs Content type)
        if analysis.is_asking_number:
            # Nếu hỏi sềE ưu tiên facts có sềE            if re.search(r'\d{4}|\d+', content_lower):
                scores["type_match"] = 0.8
        
        if analysis.query_type == "numeric":
            # Đặc biệt ưu tiên nếu có năm sinh
            if re.search(r'sinh.*\d{4}|năm.*\d{4}|\d{4}.*sinh', content_lower):
                scores["type_match"] = 1.0
        
        # 4. Special patterns for common queries
        special_boost = self._check_special_patterns(query_lower, content_lower)
        
        # Total relevance
        scores["total_relevance"] = (
            scores["keyword_match"] * 0.3 +
            scores["pattern_match"] * 0.4 +
            scores["type_match"] * 0.2 +
            special_boost * 0.1
        )
        
        return scores
    
    def _check_special_patterns(self, query_lower: str, content_lower: str) -> float:
        """Check các patterns đặc biệt và trả vềEboost score"""
        boost = 0.0
        
        # Query vềEnăm sinh ↁEContent có năm
        if any(p in query_lower for p in ["sinh năm", "năm sinh", "tuổi"]):
            years = re.findall(r'\b(19\d{2}|20[0-3]\d)\b', content_lower)
            if years:
                boost += 1.0
        
        # Query vềEngành học ↁEContent có ngành
        if any(p in query_lower for p in ["học ngành", "ngành gì", "chuyên ngành"]):
            if any(kw in content_lower for kw in ["ngành", "khmt", "cntt", "công nghềE, "khoa học"]):
                boost += 0.8
        
        # Query vềEngôn ngữ lập trình ↁEContent có tên ngôn ngữ
        if any(p in query_lower for p in ["lập trình", "ngôn ngữ", "programming"]):
            if any(lang in content_lower for lang in ["python", "java", "javascript", "c++", "code"]):
                boost += 0.8
        
        # Query vềEsềEthích ↁEContent có "thích"
        if any(p in query_lower for p in ["thích", "sềEthích", "hobby"]):
            if "thích" in content_lower:
                boost += 0.7
        
        return min(boost, 1.0)
    
    # =========================================================================
    # STEP 4: FINAL SCORING & DEDUPLICATION
    # =========================================================================
    
    def finalize_results(
        self,
        candidates: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Bước 4: Finalize kết quả
        
        - Deduplicate by content
        - Final ranking
        - Limit results
        """
        # Deduplicate by content similarity
        seen_contents = set()
        unique_results = []
        
        for result in candidates:
            content = result.get("memory", "")
            content_normalized = content.lower().strip()[:100]  # First 100 chars
            
            if content_normalized not in seen_contents:
                seen_contents.add(content_normalized)
                unique_results.append(result)
        
        # Take top results
        final_results = unique_results[:limit]
        
        self._log(f"Final results: {len(final_results)} (from {len(candidates)} candidates)")
        
        return final_results
    
    # =========================================================================
    # MAIN SEARCH METHOD
    # =========================================================================
    
    def search(
        self,
        query: str,
        strategy: str = "enhanced",
        limit: int = 10,
        temporal_context: Optional[str] = None
    ) -> SearchResult:
        """
        Main search method với Enhanced Pipeline
        
        Pipeline: Preprocess ↁEWide Retrieve ↁEFilter & Rerank ↁEFinalize
        
        Args:
            query: Câu truy vấn
            strategy: "enhanced" (default), "weighted", "hybrid"
            limit: SềEkết quả tối đa
            temporal_context: Context thời gian cho Bi-Temporal boost
        """
        self._log(f"Query: '{query}' | Strategy: {strategy}")
        
        if strategy == "enhanced":
            # Step 1: Preprocess
            analysis = self.preprocess_query(query)
            self._log(f"Query type: {analysis.query_type}, Patterns: {analysis.expected_content_patterns[:3]}")
            
            # Step 2: Wide Retrieve
            raw_results = self.retrieve_wide(query, analysis)
            
            # Step 3: Filter & Rerank
            filtered = self.filter_and_rerank(raw_results, analysis)
            
            # Step 4: Finalize
            combined = self.finalize_results(filtered, limit)
            
            # Apply temporal boost if specified
            if temporal_context and self.config.enable_temporal_priority:
                combined = self._apply_temporal_boost(combined, temporal_context)
            
            best_source = self._determine_best_source(raw_results)
            
        else:
            # Fallback to weighted strategy
            from fcm.retrieval.weighted_retriever import WeightedRetriever
            weighted = WeightedRetriever(self.memory, self.config, self.user_id, self.verbose)
            return weighted.search(query, strategy, limit, temporal_context)
        
        return SearchResult(
            query=query,
            strategy=strategy,
            solid_results=raw_results.get("solid", []),
            crystal_results=raw_results.get("crystal", []),
            liquid_results=raw_results.get("liquid", []),
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
        """Apply Bi-Temporal boost cho results có valid_at khớp"""
        for r in results:
            valid_at = r.get("metadata", {}).get("valid_at", "")
            
            if valid_at and temporal_context.lower() in str(valid_at).lower():
                current_score = r.get("enhanced_score", r.get("score", 0))
                r["enhanced_score"] = current_score * 1.2
                r["temporal_match"] = True
            else:
                r["temporal_match"] = False
        
        results.sort(key=lambda x: x.get("enhanced_score", 0), reverse=True)
        return results
    
    def _determine_best_source(
        self,
        results: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """Xác định source tốt nhất"""
        scores = {
            "solid": sum(r.get("score", 0) for r in results.get("solid", [])),
            "crystal": sum(r.get("score", 0) for r in results.get("crystal", [])),
            "liquid": sum(r.get("score", 0) for r in results.get("liquid", []))
        }
        
        weighted = {
            k: v * self.weights.get(k, 0.2)
            for k, v in scores.items()
        }
        
        if not any(weighted.values()):
            return "none"
        
        return max(weighted.keys(), key=lambda k: weighted[k])
    
    def reinforce_accessed_memories(
        self,
        results: List[Dict[str, Any]],
        solid_layer
    ):
        """Củng cềEcác memories đã được truy xuất"""
        for r in results:
            memory_id = r.get("id")
            fcm_type = r.get("metadata", {}).get("fcm_type")
            
            if memory_id and fcm_type == "solid":
                solid_layer.on_memory_accessed(memory_id)

    # =========================================================================
    # RAG REASONING - Suy luận từ retrieved context
    # =========================================================================
    
    def search_with_reasoning(
        self,
        query: str,
        limit: int = 10,
        max_context_length: int = 3000
    ) -> Dict[str, Any]:
        """
        Search + Reasoning: Retrieve context rồi dùng LLM đềEsuy luận câu trả lời.
        
        Đây là giải pháp cho LoCoMo benchmark - yêu cầu suy luận từ nhiều facts.
        
        Args:
            query: Câu hỏi
            limit: SềEmemories tối đa đềEretrieve
            max_context_length: ĐềEdài context tối đa (chars)
            
        Returns:
            {
                "answer": str,           # Câu trả lời đã suy luận
                "reasoning": str,        # Giải thích suy luận
                "retrieved_context": [...],  # Context đã retrieve
                "confidence": float      # ĐềEtin cậy
            }
        """
        self._log(f"[RAG Reasoning] Query: {query}")
        
        # Step 1: Retrieve với enhanced pipeline
        search_result = self.search(
            query=query,
            strategy="enhanced",
            limit=limit
        )
        
        # Step 2: Build context từ retrieved memories
        # search() trả vềESearchResult object với attribute combined_results
        if hasattr(search_result, 'combined_results'):
            retrieved = search_result.combined_results
        elif isinstance(search_result, dict):
            retrieved = search_result.get("combined", []) or search_result.get("combined_results", [])
        else:
            retrieved = []
        
        if not retrieved:
            return {
                "answer": "Không tìm thấy thông tin liên quan.",
                "reasoning": "Không có context đềEsuy luận.",
                "retrieved_context": [],
                "confidence": 0.0
            }
        
        # Build context string
        context_parts = []
        current_length = 0
        used_memories = []
        
        for mem in retrieved:
            content = mem.get("memory", "")
            if current_length + len(content) > max_context_length:
                break
            context_parts.append(f"- {content}")
            current_length += len(content)
            used_memories.append(mem)
        
        context_str = "\n".join(context_parts)
        
        self._log(f"[RAG Reasoning] Context: {len(used_memories)} memories, {current_length} chars")
        
        # Step 3: Reasoning với LLM
        reasoning_prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi một cách ngắn gọn và chính xác.

THÔNG TIN ĐÁEBIẾT:
{context_str}

CÂU HỎI: {query}

QUY TẮC:
1. ChềEtrả lời dựa trên thông tin đã cho, KHÔNG tự bịa thêm
2. Nếu không đủ thông tin, trả lời "Không có đủ thông tin"
3. Trả lời ngắn gọn, đúng trọng tâm câu hỏi
4. Nếu câu hỏi hỏi vềEthời gian/năm, trả lời chính xác con sềE5. Nếu câu hỏi hỏi vềEtên/địa điểm, trả lời chính xác tên

Trả vềEJSON:
{{
    "answer": "Câu trả lời ngắn gọn",
    "reasoning": "Giải thích ngắn gọn dựa trên thông tin nào",
    "confidence": 0.0-1.0
}}
"""
        
        try:
            response = self.memory.llm.generate_response(
                messages=[{"role": "user", "content": reasoning_prompt}]
            )
            
            # Parse JSON response
            import json
            import re
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback: treat entire response as answer
                result = {
                    "answer": response.strip(),
                    "reasoning": "Direct response from LLM",
                    "confidence": 0.5
                }
            
            result["retrieved_context"] = used_memories
            return result
            
        except Exception as e:
            self._log(f"[RAG Reasoning] Error: {e}", "error")
            return {
                "answer": f"Lỗi khi suy luận: {str(e)}",
                "reasoning": "Error occurred",
                "retrieved_context": used_memories,
                "confidence": 0.0
            }
    
    def answer_question(
        self,
        question: str,
        use_reasoning: bool = True
    ) -> str:
        """
        Convenience method: Trả lời câu hỏi với RAG reasoning.
        
        Args:
            question: Câu hỏi
            use_reasoning: Có dùng LLM reasoning không
            
        Returns:
            Câu trả lời dạng string
        """
        if use_reasoning:
            result = self.search_with_reasoning(question)
            return result.get("answer", "Không có câu trả lời")
        else:
            # ChềEretrieve, không reasoning
            search_result = self.search(question)
            # search() trả vềESearchResult object
            if hasattr(search_result, 'combined_results') and search_result.combined_results:
                return search_result.combined_results[0].get("memory", "")
            elif isinstance(search_result, dict):
                combined = search_result.get("combined", [])
                if combined:
                    return combined[0].get("memory", "")
            return "Không tìm thấy thông tin"


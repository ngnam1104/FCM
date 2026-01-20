"""
FCM Agent - Core Implementation (Refactored)
=============================================

Agent chính thực hiện kiến trúc Frequency-based Crystallizing Memory (FCM).
Đã tái cấu trúc để sử dụng các layer modules riêng biệt.

Pipeline:
1. Liquid Layer: Lưu raw messages (High-Frequency)
2. Crystal Layer: Extract atomic facts (Mid-Frequency, Crystallizer)
3. Solid Layer: Merge & evolve user profile (Low-Frequency, Evolver)
4. Hybrid Retrieval: Search Solid → Crystal → Liquid theo priority
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from mem0 import Memory
from fcm.v1.config import FCMConfig, get_default_fcm_config

# Import layer modules
from fcm.v1.liquid import LiquidLayer
from fcm.v1.crystal import CrystalLayer
from fcm.v1.solid import SolidLayer

# Shared Enhanced Retriever with V2
from fcm.v2.retrieval.enhanced_retriever import EnhancedRetriever

logger = logging.getLogger(__name__)


@dataclass
class ConversationStats:
    """Theo dõi statistics của conversation"""
    total_messages: int = 0
    liquid_count: int = 0
    crystal_count: int = 0
    solid_count: int = 0
    last_crystallize: int = 0
    last_evolve: int = 0


class FCMAgent:
    """
    Frequency-based Crystallizing Memory Agent (Refactored)
    
    Implements a 3-layer memory architecture:
    - Liquid (High-Frequency): Raw message storage
    - Crystal (Mid-Frequency): Extracted atomic facts
    - Solid (Low-Frequency): Consolidated user profile
    
    Now uses modular layer implementations for better maintainability.
    """
    
    def __init__(self, config: Optional[FCMConfig] = None, user_id: Optional[str] = None):
        """
        Khởi tạo FCM Agent
        
        Args:
            config: FCMConfig object, nếu None sẽ dùng default (Groq)
            user_id: User ID, nếu None sẽ dùng default từ config
        """
        self.config = config or get_default_fcm_config()
        self.user_id = user_id or self.config.default_user_id
        
        # Initialize mem0 Memory
        mem0_config = self.config.to_mem0_config()
        self.memory = Memory.from_config(mem0_config)
        
        # Conversation tracking
        self.stats = ConversationStats()
        
        # Initialize Layers (modular architecture)
        self.liquid_layer = LiquidLayer(
            self.memory, self.config, self.user_id, self.config.verbose
        )
        self.crystal_layer = CrystalLayer(
            self.memory, self.config, self.user_id,
            liquid_layer=self.liquid_layer,
            verbose=self.config.verbose
        )
        self.solid_layer = SolidLayer(
            self.memory, self.config, self.user_id,
            crystal_layer=self.crystal_layer,
            verbose=self.config.verbose
        )
        
        # Initialize Enhanced Retriever (shared with V2 for fair comparison)
        self.enhanced_retriever = EnhancedRetriever(
            self.memory, self.config, self.user_id, self.config.verbose
        )
        
        # Keep reference for backward compatibility
        self.conversation_buffer = self.liquid_layer.conversation_buffer
        
        self._log(f"FCM Agent initialized for user: {self.user_id}")
        self._log(f"Config: {self.config.llm_provider}/{self.config.llm_model}")
    
    def _log(self, message: str, level: str = "info"):
        """Helper logging với verbose check"""
        if self.config.verbose:
            if level == "info":
                logger.info(f"[FCM] {message}")
                print(f"[FCM] {message}")
            elif level == "debug":
                logger.debug(f"[FCM] {message}")
            elif level == "error":
                logger.error(f"[FCM] {message}")
                print(f"[FCM ERROR] {message}")
    
    # =========================================================================
    # LAYER 1: LIQUID (delegated to LiquidLayer)
    # =========================================================================
    
    def add_liquid_memory(self, content: str, role: str = "user",
                          extra_metadata: Optional[Dict[str, Any]] = None,
                          detect_topic_shift: bool = True) -> Dict[str, Any]:
        """
        Lưu tin nhắn vào Liquid Layer (raw storage)
        Delegates to LiquidLayer.
        """
        result = self.liquid_layer.add(
            content, role, extra_metadata, detect_topic_shift,
            on_topic_shift=lambda: self.crystallize(force=True)
        )
        
        # Sync stats
        self.stats.total_messages = self.liquid_layer.message_count
        self.stats.liquid_count = self.liquid_layer.message_count
        self.conversation_buffer = self.liquid_layer.conversation_buffer
        
        return result
    
    def get_liquid_memories(self, limit: int = 10, status: str = "raw") -> List[Dict[str, Any]]:
        """Lấy các Liquid memories. Delegates to LiquidLayer."""
        return self.liquid_layer.get(limit, status)
    
    # =========================================================================
    # LAYER 2: CRYSTAL (delegated to CrystalLayer)
    # =========================================================================
    
    def crystallize(self, force: bool = False) -> Dict[str, Any]:
        """
        Chạy Crystallizer để trích xuất Atomic Facts từ Liquid memories.
        Delegates to CrystalLayer.
        """
        liquid_memories = self.get_liquid_memories(limit=self.config.buffer_size, status="raw")
        messages_to_compress = self.liquid_layer.get_buffer()
        
        result = self.crystal_layer.crystallize(
            liquid_memories=liquid_memories,
            messages_to_compress=messages_to_compress,
            force=force,
            total_messages=self.stats.total_messages
        )
        
        # Sync stats
        if result.get("status") == "success":
            self.stats.crystal_count = self.crystal_layer.crystal_count
            self.stats.last_crystallize = self.stats.total_messages
            self.liquid_layer.clear_buffer()
            self.conversation_buffer = self.liquid_layer.conversation_buffer
        
        return result
    
    def get_crystal_memories(self, limit: int = 20,
                             category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lấy Crystal memories (atomic facts). Delegates to CrystalLayer."""
        return self.crystal_layer.get(limit, category)
    
    # =========================================================================
    # LAYER 3: SOLID (delegated to SolidLayer)
    # =========================================================================
    
    def evolve(self, force: bool = False) -> Dict[str, Any]:
        """
        Chạy Evolver để hợp nhất Crystal facts thành Solid knowledge.
        Delegates to SolidLayer.
        """
        result = self.solid_layer.evolve(
            force=force,
            total_messages=self.stats.total_messages
        )
        
        # Sync stats
        if result.get("status") == "success":
            self.stats.solid_count = self.solid_layer.solid_count
            self.stats.last_evolve = self.stats.total_messages
        
        return result
    
    def get_solid_memories(self, limit: int = 30,
                           section: Optional[str] = None,
                           status: str = "active") -> List[Dict[str, Any]]:
        """Lấy Solid memories. Delegates to SolidLayer."""
        return self.solid_layer.get(limit, section, status)
    
    def get_memory_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """[MAPLE/G-Memory] Truy vết lịch sử của một memory."""
        return self.solid_layer.get_history(memory_id)
    
    # =========================================================================
    # HYBRID RETRIEVAL - Uses Enhanced Retriever (shared with V2)
    # =========================================================================
    
    def search(self, query: str, strategy: str = "enhanced") -> Dict[str, Any]:
        """
        Tìm kiếm memories với Enhanced Retriever (shared với V2).
        
        Strategies:
        - "enhanced": Enhanced Pipeline (default, shared with V2)
        - "hybrid": Solid → Crystal → Liquid (legacy)
        - "solid_first": Chỉ search Solid, fallback Crystal
        - "all_layers": Search tất cả layers cùng lúc
        - "recent": Ưu tiên Liquid (context gần đây)
        """
        self._log(f"[Search] Query: '{query[:50]}...' | Strategy: {strategy}")
        
        # Use Enhanced Retriever for "enhanced" strategy
        if strategy == "enhanced":
            search_result = self.enhanced_retriever.search(
                query=query,
                strategy="enhanced",
                limit=10
            )
            return {
                "query": query,
                "strategy": strategy,
                "solid": search_result.solid_results,
                "crystal": search_result.crystal_results,
                "liquid": search_result.liquid_results,
                "combined": search_result.combined_results,
                "best_source": search_result.best_source
            }
        
        # Legacy strategies
        if strategy == "hybrid":
            results = self._search_hybrid(query)
        elif strategy == "solid_first":
            results = self._search_solid_first(query)
        elif strategy == "all_layers":
            results = self._search_all_layers(query)
        elif strategy == "recent":
            results = self._search_recent(query)
        else:
            results = self._search_hybrid(query)
        
        self._log(f"[Search] Found: {len(results.get('combined', []))} results")
        return results
    
    def _search_hybrid(self, query: str) -> Dict[str, Any]:
        """Chiến lược Hybrid: Solid → Crystal → Liquid với smart fallback"""
        results = {
            "query": query, "strategy": "hybrid",
            "solid": [], "crystal": [], "liquid": [],
            "combined": [], "best_source": None
        }
        
        # Step 1: Search Solid
        solid_results = self._search_layer(query, self.config.solid_type, self.config.solid_search_limit)
        results["solid"] = solid_results
        
        if solid_results:
            best_score = solid_results[0].get("score", 0)
            if best_score >= self.config.hybrid_score_threshold:
                results["combined"] = solid_results
                results["best_source"] = "solid"
                return results
        
        # Step 2: Search Crystal
        crystal_results = self._search_layer(query, self.config.crystal_type, self.config.crystal_search_limit)
        results["crystal"] = crystal_results
        
        combined = solid_results + crystal_results
        if combined:
            best_score = max(r.get("score", 0) for r in combined)
            if best_score >= self.config.hybrid_score_threshold:
                results["combined"] = self._deduplicate_and_rank(combined, query)
                results["best_source"] = "solid+crystal"
                return results
        
        # Step 3: Fallback to Liquid
        liquid_results = self._search_layer(query, self.config.liquid_type, self.config.liquid_search_limit)
        results["liquid"] = liquid_results
        
        all_results = solid_results + crystal_results + liquid_results
        results["combined"] = self._deduplicate_and_rank(all_results, query)
        results["best_source"] = "all_layers" if all_results else "none"
        
        return results
    
    def _search_solid_first(self, query: str) -> Dict[str, Any]:
        """Search Solid first, fallback to Crystal only"""
        results = {
            "query": query, "strategy": "solid_first",
            "solid": [], "crystal": [], "liquid": [],
            "combined": [], "best_source": None
        }
        
        solid_results = self._search_layer(query, self.config.solid_type, self.config.solid_search_limit)
        results["solid"] = solid_results
        
        if solid_results and solid_results[0].get("score", 0) >= self.config.hybrid_score_threshold:
            results["combined"] = solid_results
            results["best_source"] = "solid"
        else:
            crystal_results = self._search_layer(query, self.config.crystal_type, self.config.crystal_search_limit)
            results["crystal"] = crystal_results
            results["combined"] = solid_results + crystal_results
            results["best_source"] = "solid+crystal"
        
        return results
    
    def _search_all_layers(self, query: str) -> Dict[str, Any]:
        """Search all layers simultaneously"""
        results = {
            "query": query, "strategy": "all_layers",
            "solid": self._search_layer(query, self.config.solid_type, self.config.solid_search_limit),
            "crystal": self._search_layer(query, self.config.crystal_type, self.config.crystal_search_limit),
            "liquid": self._search_layer(query, self.config.liquid_type, self.config.liquid_search_limit),
            "combined": [], "best_source": "all_layers"
        }
        
        all_results = results["solid"] + results["crystal"] + results["liquid"]
        results["combined"] = self._deduplicate_and_rank(all_results, query)
        
        return results
    
    def _search_recent(self, query: str) -> Dict[str, Any]:
        """Ưu tiên Liquid (recent context)"""
        results = {
            "query": query, "strategy": "recent",
            "solid": [], "crystal": [], "liquid": [],
            "combined": [], "best_source": None
        }
        
        liquid_results = self._search_layer(query, self.config.liquid_type, self.config.liquid_search_limit * 2)
        results["liquid"] = liquid_results
        
        if liquid_results:
            results["combined"] = liquid_results
            results["best_source"] = "liquid"
        else:
            crystal_results = self._search_layer(query, self.config.crystal_type, self.config.crystal_search_limit)
            results["crystal"] = crystal_results
            results["combined"] = crystal_results
            results["best_source"] = "crystal"
        
        return results
    
    def _search_layer(self, query: str, fcm_type: str, limit: int,
                      include_archived: bool = False) -> List[Dict[str, Any]]:
        """Search trong một layer cụ thể với client-side filtering"""
        try:
            all_results = self.memory.search(
                query,
                user_id=self.user_id,
                limit=limit * 4
            )
            
            layer_results = []
            for mem in all_results.get("results", []):
                mem_metadata = mem.get("metadata", {})
                
                if mem_metadata.get("fcm_type") != fcm_type:
                    continue
                
                if not include_archived:
                    mem_status = mem_metadata.get("fcm_status", "active")
                    if mem_status == "archived":
                        continue
                    mem_content = mem.get("memory", "")
                    if mem_content.startswith("[ARCHIVED]"):
                        continue
                
                layer_results.append(mem)
            
            return layer_results[:limit]
            
        except Exception as e:
            self._log(f"[Search] Error searching {fcm_type} layer: {e}", "error")
            return []
    
    def _deduplicate_and_rank(self, results: List[Dict[str, Any]],
                              query: str = "") -> List[Dict[str, Any]]:
        """Loại bỏ duplicate và rank theo score với keyword boosting."""
        from fcm.v1.utils import calculate_keyword_boost
        
        seen_contents = set()
        unique_results = []
        
        for r in results:
            content = r.get("memory", "")
            if content not in seen_contents:
                seen_contents.add(content)
                
                if query:
                    original_score = r.get("score", 0)
                    keyword_boost = calculate_keyword_boost(query, content)
                    r["keyword_boost"] = keyword_boost
                    r["original_score"] = original_score
                    r["score"] = original_score + keyword_boost
                
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return unique_results
    
    # =========================================================================
    # HIGH-LEVEL API
    # =========================================================================
    
    def chat(self, user_message: str,
             auto_crystallize: bool = True,
             return_context: bool = False) -> Dict[str, Any]:
        """
        Process một tin nhắn từ user.
        
        Pipeline:
        1. Lưu vào Liquid layer (với Topic Shift detection)
        2. Kiểm tra trigger crystallize
        3. Retrieve context (nếu cần)
        """
        result = {
            "message": user_message,
            "liquid_saved": False,
            "crystallized": False,
            "crystallize_trigger": None,
            "topic_shifted": False,
            "context": None,
            "stats": None
        }
        
        # 1. Save to Liquid
        liquid_result = self.add_liquid_memory(user_message, role="user")
        result["liquid_saved"] = True
        result["topic_shifted"] = liquid_result.get("topic_shifted", False)
        
        # 2. Check crystallize trigger
        if auto_crystallize:
            should_crystallize = False
            trigger_reason = None
            
            if liquid_result.get("topic_shifted", False):
                should_crystallize = True
                trigger_reason = "topic_shift"
            
            messages_since = self.stats.total_messages - self.stats.last_crystallize
            if messages_since >= self.config.crystallize_threshold:
                should_crystallize = True
                trigger_reason = trigger_reason or "threshold"
            
            if should_crystallize:
                crystal_result = self.crystallize()
                result["crystallized"] = crystal_result.get("status") == "success"
                result["crystallize_trigger"] = trigger_reason
        
        # 3. Retrieve context if requested
        if return_context:
            context = self.search(user_message, strategy="hybrid")
            result["context"] = context
        
        # 4. Return stats
        result["stats"] = {
            "total_messages": self.stats.total_messages,
            "liquid_count": self.stats.liquid_count,
            "crystal_count": self.stats.crystal_count,
            "solid_count": self.stats.solid_count
        }
        
        return result
    
    def end_session(self, auto_evolve: bool = True) -> Dict[str, Any]:
        """Kết thúc session và chạy các cleanup processes."""
        result = {
            "crystallized": False,
            "evolved": False,
            "final_stats": None
        }
        
        # 1. Final crystallize (force)
        crystal_result = self.crystallize(force=True)
        result["crystallized"] = crystal_result.get("status") == "success"
        
        # 2. Evolve if requested
        if auto_evolve:
            evolve_result = self.evolve(force=True)
            result["evolved"] = evolve_result.get("status") == "success"
        
        # 3. Final stats
        result["final_stats"] = {
            "total_messages": self.stats.total_messages,
            "liquid_count": self.stats.liquid_count,
            "crystal_count": self.stats.crystal_count,
            "solid_count": self.stats.solid_count
        }
        
        self._log(f"[Session] Ended. Stats: {result['final_stats']}")
        return result
    
    def get_user_profile(self) -> Dict[str, Any]:
        """Lấy User Profile từ Solid layer"""
        return self.solid_layer.get_user_profile()
    
    def get_all_memories_by_layer(self) -> Dict[str, List[Dict[str, Any]]]:
        """Lấy tất cả memories, organized by layer"""
        return {
            "liquid": self.get_liquid_memories(limit=50, status="all"),
            "crystal": self.get_crystal_memories(limit=50),
            "solid": self.get_solid_memories(limit=50)
        }
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def reset(self) -> None:
        """Reset agent state (không xóa memories trong DB)"""
        self.stats = ConversationStats()
        self.liquid_layer.conversation_buffer = []
        self.liquid_layer.message_count = 0
        self.crystal_layer.crystal_count = 0
        self.crystal_layer.last_crystallize_at = 0
        self.solid_layer.solid_count = 0
        self.solid_layer.last_evolve_at = 0
        self.conversation_buffer = self.liquid_layer.conversation_buffer
        self._log("[FCM] Agent state reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy current statistics"""
        return {
            "total_messages": self.stats.total_messages,
            "liquid_count": self.stats.liquid_count,
            "crystal_count": self.stats.crystal_count,
            "solid_count": self.stats.solid_count,
            "last_crystallize": self.stats.last_crystallize,
            "last_evolve": self.stats.last_evolve,
            "buffer_size": len(self.liquid_layer.conversation_buffer)
        }
    
    def reset(self):
        """Reset agent state (không xóa memories trong DB)"""
        self.stats = ConversationStats()
        self.liquid_layer.conversation_buffer = []
        self.conversation_buffer = self.liquid_layer.conversation_buffer
        self._log("Agent state reset")
    
    def clear_all_memories(self):
        """Xóa tất cả memories của user trong DB"""
        try:
            # Clear from mem0
            self.memory.delete_all(user_id=self.user_id)
            # Reset agent state
            self.reset()
            self._log(f"Cleared all memories for user: {self.user_id}")
        except Exception as e:
            self._log(f"Error clearing memories: {e}", level="error")
    
    # =========================================================================
    # LEGACY COMPATIBILITY METHODS
    # =========================================================================
    
    def _detect_topic_shift(self, current_context, new_message):
        """Legacy method - delegates to LiquidLayer"""
        return self.liquid_layer._detect_topic_shift(current_context, new_message)
    
    def _segment_conversation(self, messages):
        """Legacy method - delegates to LiquidLayer"""
        return self.liquid_layer.segment_conversation(messages)
    
    def _compress_conversation(self, messages):
        """Legacy method - delegates to LiquidLayer"""
        result = self.liquid_layer.compress_conversation(messages)
        return {
            "compressed_narrative": result.compressed_narrative,
            "key_entities": result.key_entities,
            "noise_count": result.noise_count,
            "original_length": result.original_length,
            "compressed_length": result.compressed_length
        }
    
    def _is_noise_message(self, content):
        """Legacy method - delegates to LiquidLayer"""
        return self.liquid_layer.is_noise_message(content)
    
    def _archive_solid_memory(self, memory_id, valid_until, superseded_by):
        """Legacy method - delegates to SolidLayer"""
        return self.solid_layer._archive(memory_id, valid_until, superseded_by)

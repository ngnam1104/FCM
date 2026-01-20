"""
FCM V2 Agent - Main Implementation
===================================

Agent chính với tất cả 5 cải tiến:
1. Bi-Temporal Schema cho Crystal Layer
2. Attention Sinks & Semantic Grouping cho Liquid Layer
3. Active Forgetting (Ebbinghaus Curve) cho Solid Layer
4. Dynamic Persona
5. Weighted Ensemble Retrieval
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from mem0 import Memory

from .config import FCMConfigV2, get_default_config_v2
from .liquid.layer import LiquidLayer
from .crystal.layer import CrystalLayer
from .solid.layer import SolidLayer
from .retrieval.weighted_retriever import WeightedRetriever
from .retrieval.enhanced_retriever import EnhancedRetriever
from .schemas.base import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ConversationStatsV2:
    """Theo dõi statistics của conversation"""
    total_messages: int = 0
    liquid_count: int = 0
    crystal_count: int = 0
    solid_count: int = 0
    last_crystallize: int = 0
    last_evolve: int = 0
    topic_shifts_detected: int = 0
    llm_calls_saved: int = 0  # Bởi Semantic Grouping


class FCMAgentV2:
    """
    FCM Agent V2 - Frequency-based Crystallizing Memory với các cải tiến:
    
    1. Bi-Temporal: Crystal facts có valid_at (thời gian sự kiện xảy ra)
    2. Attention Sinks: Giữ K tin nhắn đầu tiên, semantic grouping trước LLM
    3. Active Forgetting: Decay memories theo Ebbinghaus curve
    4. Dynamic Persona: Track interaction style, update system prompt
    5. Weighted Retrieval: Score_final = 0.5*Solid + 0.3*Crystal + 0.2*Liquid
    """
    
    def __init__(
        self,
        config: Optional[FCMConfigV2] = None,
        user_id: Optional[str] = None
    ):
        """
        Khởi tạo FCM Agent V2
        
        Args:
            config: FCMConfigV2 object, nếu None sẽ dùng default
            user_id: User ID
        """
        self.config = config or get_default_config_v2()
        self.user_id = user_id or self.config.default_user_id
        
        # Initialize mem0 Memory
        mem0_config = self.config.to_mem0_config()
        self.memory = Memory.from_config(mem0_config)
        
        # Initialize layers
        self.liquid_layer = LiquidLayer(
            self.memory, self.config, self.user_id, self.config.verbose
        )
        
        self.crystal_layer = CrystalLayer(
            self.memory, self.config, self.user_id, self.config.verbose
        )
        
        self.solid_layer = SolidLayer(
            self.memory, self.config, self.user_id, self.config.verbose
        )
        
        # Initialize retrievers (Enhanced as default, Weighted as fallback)
        self.retriever = EnhancedRetriever(
            self.memory, self.config, self.user_id, self.config.verbose
        )
        self.weighted_retriever = WeightedRetriever(
            self.memory, self.config, self.user_id, self.config.verbose
        )
        
        # Statistics
        self.stats = ConversationStatsV2()
        
        # Conversation samples for persona extraction
        self._conversation_samples: List[str] = []
        
        self._log(f"FCM Agent V2 initialized for user: {self.user_id}")
        self._log(f"Config: {self.config.llm_provider}/{self.config.llm_model}")
        self._log(f"Features: AttentionSinks={self.config.attention_sink_count}, "
                 f"ActiveForgetting={self.config.enable_active_forgetting}, "
                 f"DynamicPersona={self.config.enable_dynamic_persona}")
    
    def _log(self, message: str, level: str = "info"):
        """Helper logging"""
        if self.config.verbose:
            prefix = "[FCM V2]"
            if level == "info":
                logger.info(f"{prefix} {message}")
                print(f"{prefix} {message}")
            elif level == "error":
                logger.error(f"{prefix} {message}")
                print(f"{prefix} ❌ {message}")
    
    # =========================================================================
    # HIGH-LEVEL API
    # =========================================================================
    
    def chat(
        self,
        user_message: str,
        auto_crystallize: bool = True,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Process một tin nhắn từ user.
        
        Pipeline:
        1. Add to Liquid (với Attention Sinks + Semantic Grouping)
        2. Check crystallize trigger
        3. Return context nếu requested
        
        Args:
            user_message: Tin nhắn từ user
            auto_crystallize: Tự động crystallize
            return_context: Có retrieve context không
            
        Returns:
            Dict với kết quả processing
        """
        result = {
            "message": user_message,
            "liquid_saved": False,
            "is_attention_sink": False,
            "crystallized": False,
            "crystallize_trigger": None,
            "topic_shifted": False,
            "used_llm_for_topic_shift": None,
            "context": None,
            "stats": None
        }
        
        # 1. Add to Liquid (với cải tiến 2)
        liquid_result = self.liquid_layer.add_message(
            user_message,
            role="user",
            detect_topic_shift=True
        )
        
        result["liquid_saved"] = True
        result["is_attention_sink"] = liquid_result.get("is_attention_sink", False)
        result["topic_shifted"] = liquid_result.get("topic_shifted", False)
        
        # Track LLM calls saved
        shift_info = liquid_result.get("shift_info")
        if shift_info:
            result["used_llm_for_topic_shift"] = shift_info.get("used_llm", False)
            if not shift_info.get("used_llm", True):
                self.stats.llm_calls_saved += 1
        
        # Update stats
        self.stats.total_messages += 1
        self.stats.liquid_count += 1
        
        if result["topic_shifted"]:
            self.stats.topic_shifts_detected += 1
        
        # Store for persona extraction
        self._conversation_samples.append(f"user: {user_message}")
        
        # 2. Check crystallize trigger
        if auto_crystallize:
            should_crystallize = False
            trigger_reason = None
            
            # Trigger 1: Topic Shift
            if liquid_result.get("topic_shifted", False):
                should_crystallize = True
                trigger_reason = "topic_shift"
            
            # Trigger 2: Threshold
            buffer_size = len(self.liquid_layer.conversation_buffer)
            if buffer_size >= self.config.crystallize_threshold:
                should_crystallize = True
                trigger_reason = trigger_reason or "threshold"
            
            if should_crystallize:
                crystal_result = self.crystallize()
                result["crystallized"] = crystal_result.get("status") == "success"
                result["crystallize_trigger"] = trigger_reason
        
        # 3. Retrieve context
        # 3. Retrieve context (ĐOẠN CẦN SỬA)
        if return_context:
            # Hàm search giờ trả về Dict
            context = self.search(user_message, strategy="weighted")
            
            # Gán thẳng dict vào kết quả
            result["context"] = context
            
            # Lấy list kết quả an toàn từ dict
            combined_results = context.get("combined") or context.get("combined_results", [])
            
            # Cải tiến 3: Reinforce accessed memories
            if combined_results:
                self.retriever.reinforce_accessed_memories(
                    combined_results,
                    self.solid_layer
                )
        
        # 4. Stats
        result["stats"] = self.get_stats()
        
        return result
    def crystallize(self, force: bool = False, session_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run Crystallization process với Bi-Temporal extraction
        
        Args:
            force: Bỏ qua threshold check
            session_date: Session date for temporal computation (Option 2)
        """
        # Check threshold
        buffer_size = len(self.liquid_layer.conversation_buffer)
        if not force and buffer_size < self.config.crystallize_threshold:
            return {"status": "skipped", "reason": "threshold_not_met"}
        
        # Get messages to crystallize
        messages = self.liquid_layer.get_buffer_for_crystallize()
        
        # If buffer empty but force=True, try getting from attention sinks
        if not messages and force:
            # Get all messages including attention sinks
            messages = []
            for sink in self.liquid_layer.attention_sinks:
                # Handle both dict and LiquidMessage objects
                if hasattr(sink, 'content'):
                    content = sink.content
                else:
                    content = sink.get("content", "") if isinstance(sink, dict) else str(sink)
                messages.append({
                    "role": "user",
                    "content": content
                })
            for msg in self.liquid_layer.conversation_buffer:
                # Handle both dict and LiquidMessage objects
                if hasattr(msg, 'content'):
                    content = msg.content
                    role = msg.role if hasattr(msg, 'role') else "user"
                else:
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    role = msg.get("role", "user") if isinstance(msg, dict) else "user"
                messages.append({
                    "role": role,
                    "content": content
                })
        
        if not messages:
            return {"status": "skipped", "reason": "empty_buffer"}
        
        # Run crystallization với Bi-Temporal (pass session_date for Option 2)
        result = self.crystal_layer.crystallize(
            messages, 
            session_date=session_date
        )
        
        # Update stats
        if result.get("status") == "success":
            self.stats.crystal_count += result.get("facts_extracted", 0)
            self.stats.last_crystallize = self.stats.total_messages
            
            # Clear liquid buffer (giữ attention sinks)
            self.liquid_layer.clear_buffer()
        
        return result
    
    def evolve(self, force: bool = False) -> Dict[str, Any]:
        """
        Run Evolution process với:
        - MAPLE versioning
        - Dynamic Persona extraction
        - Active Forgetting (prune weak memories)
        
        Args:
            force: Bỏ qua threshold check
        """
        # Check threshold
        messages_since = self.stats.total_messages - self.stats.last_evolve
        if not force and messages_since < self.config.evolve_threshold:
            return {"status": "skipped", "reason": "threshold_not_met"}
        
        # Get crystal facts
        crystal_facts = self.crystal_layer.get_memories(limit=20)
        
        if not crystal_facts:
            return {"status": "skipped", "reason": "no_crystal_facts"}
        
        # Run evolution với Dynamic Persona
        result = self.solid_layer.evolve(
            crystal_facts=crystal_facts,
            conversation_samples=self._conversation_samples[-20:],  # Last 20 samples
            total_messages=self.stats.total_messages
        )
        
        # Update stats
        if result.get("status") == "success":
            self.stats.solid_count += result.get("processed", 0)
            self.stats.last_evolve = self.stats.total_messages
            
            # Clear conversation samples
            self._conversation_samples = []
        
        return result
    
    def search(
        self,
        query: str,
        strategy: str = "enhanced",
        limit: int = 10,
        temporal_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search wrapper trả về Dictionary tương thích với demoUI.py
        """
        # 1. Thực hiện search (giữ nguyên logic cũ)
        if strategy == "weighted":
            result_obj = self.weighted_retriever.search(
                query=query,
                strategy=strategy,
                limit=limit,
                temporal_context=temporal_context
            )
        else:
            result_obj = self.retriever.search(
                query=query,
                strategy=strategy,
                limit=limit,
                temporal_context=temporal_context
            )
            
        # 2. Chuyển đổi Object -> Dict
        # Kiểm tra nếu object có hàm model_dump (Pydantic V2) hoặc dict (Pydantic V1)
        if hasattr(result_obj, "model_dump"):
            result_dict = result_obj.model_dump()
        elif hasattr(result_obj, "dict"):
            result_dict = result_obj.dict()
        else:
            result_dict = result_obj if isinstance(result_obj, dict) else {}

        # 3. FIX TƯƠNG THÍCH: Map 'combined_results' (V2) thành 'combined' (V1 demoUI)
        # UI dùng .get('combined'), còn V2 trả về 'combined_results'
        if "combined_results" in result_dict:
            result_dict["combined"] = result_dict["combined_results"]
        
        return result_dict
    
    def end_session(self, auto_evolve: bool = True) -> Dict[str, Any]:
        """
        Kết thúc session
        
        Args:
            auto_evolve: Tự động chạy evolve
        """
        result = {
            "crystallized": False,
            "evolved": False,
            "pruned_memories": 0,
            "final_stats": None
        }
        
        # Final crystallize
        crystal_result = self.crystallize(force=True)
        result["crystallized"] = crystal_result.get("status") == "success"
        
        # Evolve
        if auto_evolve:
            evolve_result = self.evolve(force=True)
            result["evolved"] = evolve_result.get("status") == "success"
            result["pruned_memories"] = evolve_result.get("pruned", 0)
        
        # Final stats
        result["final_stats"] = self.get_stats()
        
        self._log(f"Session ended. Stats: {result['final_stats']}")
        
        return result
    
    # =========================================================================
    # GETTERS
    # =========================================================================
    
    def get_user_profile(self) -> Dict[str, List[str]]:
        """Lấy User Profile từ Solid layer"""
        return self.solid_layer.get_user_profile()
    
    def get_user_persona(self) -> Optional[Dict[str, Any]]:
        """Lấy Dynamic User Persona (Cải tiến 4)"""
        if self.solid_layer.user_persona:
            return self.solid_layer.user_persona.model_dump()
        return None
    
    def get_persona_prompt_injection(self) -> str:
        """
        Lấy đoạn text để inject vào system prompt
        Dùng cho Dynamic Persona (Cải tiến 4)
        """
        return self.solid_layer.get_persona_prompt_injection()
    
    def get_full_context(self) -> List[Dict[str, Any]]:
        """
        Lấy full context bao gồm Attention Sinks + Buffer
        (Cải tiến 2: Attention Sinks)
        """
        return self.liquid_layer.get_full_context()
    
    def get_memories_by_layer(self) -> Dict[str, List[Dict[str, Any]]]:
        """Lấy tất cả memories organized by layer"""
        return {
            "liquid": self.liquid_layer.get_messages(limit=50),
            "crystal": self.crystal_layer.get_memories(limit=50),
            "solid": self.solid_layer.get_memories(limit=50)
        }
        
    def get_crystal_memories(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Wrapper lấy crystal memories cho UI"""
        return self.crystal_layer.get_memories(limit=limit)

    def get_liquid_memories(self, limit: int = 20, status: str = "all") -> List[Dict[str, Any]]:
        """Wrapper lấy liquid messages cho UI (bỏ qua tham số status của V1)"""
        return self.liquid_layer.get_messages(limit=limit)

    def get_all_memories_by_layer(self) -> Dict[str, List[Dict[str, Any]]]:
        """Alias cho hàm get_memories_by_layer để khớp tên gọi trong UI"""
        return self.get_memories_by_layer()
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy statistics"""
        return {
            "total_messages": self.stats.total_messages,
            "liquid_count": self.stats.liquid_count,
            "crystal_count": self.stats.crystal_count,
            "solid_count": self.stats.solid_count,
            "last_crystallize": self.stats.last_crystallize,
            "last_evolve": self.stats.last_evolve,
            "topic_shifts_detected": self.stats.topic_shifts_detected,
            "llm_calls_saved": self.stats.llm_calls_saved,
            "attention_sinks_count": len(self.liquid_layer.attention_sinks),
            "buffer_size": len(self.liquid_layer.conversation_buffer)
        }
    
    def reset(self):
        """Reset agent state (không xóa memories trong DB)"""
        self.stats = ConversationStatsV2()
        self.liquid_layer.attention_sinks = []
        self.liquid_layer.conversation_buffer = []
        self._conversation_samples = []
        self._log("Agent state reset")
    
    def clear_all_memories(self):
        """Xóa tất cả memories của user trong DB"""
        try:
            # Clear from all layers
            self.liquid_layer.memory.delete_all(user_id=self.user_id)
            self.crystal_layer.memory.delete_all(user_id=self.user_id)
            self.solid_layer.memory.delete_all(user_id=self.user_id)
            
            # Reset agent state
            self.reset()
            self._log(f"Cleared all memories for user: {self.user_id}")
        except Exception as e:
            self._log(f"Error clearing memories: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_fcm_agent_v2(
    llm_provider: str = "groq",
    llm_model: str = "llama-3.1-8b-instant",
    user_id: str = "default_user",
    verbose: bool = True
) -> FCMAgentV2:
    """
    Convenience function để tạo FCMAgentV2
    
    Args:
        llm_provider: "groq", "openai", "ollama"
        llm_model: Model name
        user_id: User ID
        verbose: Có log không
    """
    config = FCMConfigV2(
        llm_provider=llm_provider,
        llm_model=llm_model,
        verbose=verbose
    )
    
    return FCMAgentV2(config=config, user_id=user_id)

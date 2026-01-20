"""
FCM V1 Crystal Layer
======================

Mid-Frequency Memory Layer - The Crystallizer
Trích xuất Atomic Facts từ Liquid memories.

Dựa trên:
- SeCom: Segmentation & Denoising
- A-Mem: Atomic Notes (Zettelkasten style)
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from mem0 import Memory
from fcm.v1.config import FCMConfig
from fcm.v1.schemas import CrystalFact, CompressionResult
from fcm.v1.utils import extract_json_from_text
from fcm.v1.prompts import CRYSTALLIZER_SYSTEM_PROMPT, CRYSTALLIZER_USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class CrystalLayer:
    """
    Crystal Layer - Mid-Frequency Atomic Fact Extraction
    
    SeCom Concepts Applied:
    - Segmentation: Phân đoạn hội thoại thành semantic units
    - Denoising: Loại bỏ noise (greetings, fillers, acknowledgments)
    - Compression: Nén thông tin thành dạng ngắn gọn
    
    A-Mem Concepts Applied:
    - Atomic Notes: Output là JSON objects độc lập
    - Self-contained: Mỗi fact có thể hiểu được mà không cần context
    - Categorized: Phân loại theo type (personal_info, preference, fact...)
    """
    
    def __init__(self, memory: Memory, config: FCMConfig, user_id: str, 
                 liquid_layer: Optional[Any] = None, verbose: bool = True):
        """
        Khởi tạo Crystal Layer
        
        Args:
            memory: mem0 Memory instance
            config: FCMConfig
            user_id: User ID
            liquid_layer: Reference to LiquidLayer (for compression)
            verbose: Enable logging
        """
        self.memory = memory
        self.config = config
        self.user_id = user_id
        self.liquid_layer = liquid_layer
        self.verbose = verbose
        
        self.crystal_count = 0
        self.last_crystallize_at = 0
    
    def _log(self, message: str, level: str = "info"):
        """Helper logging"""
        if self.verbose:
            if level == "info":
                logger.info(f"[Crystal] {message}")
                print(f"[Crystal] {message}")
            elif level == "debug":
                logger.debug(f"[Crystal] {message}")
            elif level == "error":
                logger.error(f"[Crystal] {message}")
                print(f"[Crystal ERROR] {message}")
            elif level == "warning":
                logger.warning(f"[Crystal] {message}")
                print(f"[Crystal WARNING] {message}")
    
    def crystallize(self, liquid_memories: List[Dict[str, Any]],
                    messages_to_compress: Optional[List[Dict[str, Any]]] = None,
                    force: bool = False,
                    total_messages: int = 0) -> Dict[str, Any]:
        """
        Chạy Crystallizer để trích xuất Atomic Facts từ Liquid memories.
        
        [SeCom] Pipeline:
        1. Segmentation: Phân đoạn conversation thành semantic units
        2. Compression: Nén hội thoại, loại bỏ noise
        3. Atomic Fact Extraction: Trích xuất facts từ compressed narrative
        
        Args:
            liquid_memories: List Liquid memories cần crystallize
            messages_to_compress: Messages từ conversation buffer
            force: Bỏ qua threshold check
            total_messages: Total message count (for threshold check)
            
        Returns:
            Dict với kết quả crystallization
        """
        # Check threshold
        messages_since_last = total_messages - self.last_crystallize_at
        if not force and messages_since_last < self.config.crystallize_threshold:
            self._log(f"Skip: {messages_since_last}/{self.config.crystallize_threshold} messages")
            return {"status": "skipped", "reason": "threshold_not_met"}
        
        if not liquid_memories:
            self._log("No raw liquid memories to crystallize")
            return {"status": "skipped", "reason": "no_raw_memories"}
        
        # === Step 1: Segmentation ===
        self._log("[SeCom] Step 1: Semantic Segmentation...")
        segments = []
        if self.liquid_layer:
            segments = self.liquid_layer.segment_conversation(liquid_memories)
        else:
            segments = [liquid_memories]
        self._log(f"[SeCom] Segmented into {len(segments)} semantic units")
        
        # === Step 2: Compression ===
        self._log("[SeCom] Step 2: Conversation Compression...")
        
        if messages_to_compress:
            messages = messages_to_compress[-self.config.buffer_size:]
        else:
            messages = liquid_memories
        
        compression_result = self._compress(messages)
        compressed_narrative = compression_result.compressed_narrative
        
        if not compressed_narrative.strip():
            self._log("Compression produced empty result, skipping")
            return {"status": "skipped", "reason": "empty_compression"}
        
        self._log(f"Crystallizing compressed narrative ({compression_result.compressed_length} chars)...")
        
        # === Step 3: Atomic Fact Extraction ===
        try:
            response = self.memory.llm.generate_response(
                messages=[
                    {"role": "system", "content": CRYSTALLIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": CRYSTALLIZER_USER_PROMPT_TEMPLATE.format(
                        chat_log=compressed_narrative
                    )}
                ]
            )
            
            facts_data = extract_json_from_text(response)
            
            # Normalize to list
            facts = []
            if facts_data:
                if isinstance(facts_data, dict):
                    facts = facts_data.get("facts", [])
                elif isinstance(facts_data, list):
                    facts = facts_data
            
            if not facts:
                self._log(f"Warning: No facts extracted. Raw: {str(response)[:100]}...", "warning")
            else:
                self._log(f"Extracted {len(facts)} atomic facts")
            
            # Save Crystal facts
            crystal_results = []
            for fact in facts:
                result = self._save_fact(fact, liquid_memories)
                if result:
                    crystal_results.append(result)
            
            # Update state
            self.last_crystallize_at = total_messages
            
            return {
                "status": "success",
                "facts_extracted": len(facts),
                "compression_info": {
                    "original_chars": compression_result.original_length,
                    "compressed_chars": compression_result.compressed_length,
                    "noise_removed": compression_result.noise_count,
                },
                "segments_count": len(segments),
                "crystals": crystal_results
            }
            
        except Exception as e:
            self._log(f"Error during crystallization: {e}", "error")
            return {"status": "error", "error": str(e)}
    
    def _compress(self, messages: List[Dict[str, Any]]) -> CompressionResult:
        """Compress conversation using LiquidLayer or fallback"""
        if self.liquid_layer:
            return self.liquid_layer.compress_conversation(messages)
        
        # Fallback: just join messages
        chat_log = "\n".join([
            f"[{msg.get('role', 'user').upper()}]: {msg.get('memory', msg.get('content', ''))}"
            for msg in messages
        ])
        return CompressionResult(
            compressed_narrative=chat_log,
            key_entities=[],
            noise_count=0,
            original_length=len(chat_log),
            compressed_length=len(chat_log)
        )
    
    def _save_fact(self, fact: Any, source_memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Save a single Crystal fact"""
        if isinstance(fact, dict):
            fact_content = fact.get("content", str(fact))
            fact_category = fact.get("category", "fact")
            fact_confidence = fact.get("confidence", 0.8)
            fact_keywords = fact.get("keywords", [])
            fact_context_tags = fact.get("context_tags", [])
            fact_related_to = fact.get("related_to", [])
        else:
            fact_content = str(fact)
            fact_category = "fact"
            fact_confidence = 0.8
            fact_keywords = []
            fact_context_tags = []
            fact_related_to = []
        
        if not fact_content:
            return None
        
        crystal_metadata = {
            "fcm_type": self.config.crystal_type,
            "fcm_frequency": 2,
            "fcm_status": "active",
            "category": fact_category,
            "confidence": fact_confidence,
            "keywords": fact_keywords,
            "context_tags": fact_context_tags,
            "related_to": fact_related_to,
            "source_messages": [m.get("id") for m in source_memories],
            "crystallized_at": datetime.now().isoformat(),
        }
        
        add_result = self.memory.add(
            fact_content,
            user_id=self.user_id,
            metadata=crystal_metadata,
            infer=False
        )
        
        self.crystal_count += 1
        self._log(f"✓ Formed: '{fact_content[:60]}...'")
        
        return {
            "content": fact_content,
            "category": fact_category,
            "result": add_result
        }
    
    def get(self, limit: int = 20, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lấy Crystal memories (atomic facts)
        
        Args:
            limit: Số lượng tối đa
            category: Filter theo category (personal_info, preference, fact, etc.)
            
        Returns:
            List các crystal memories
        """
        all_memories = self.memory.get_all(
            user_id=self.user_id,
            limit=limit * 2
        )
        
        crystal_memories = []
        for mem in all_memories.get("results", []):
            mem_metadata = mem.get("metadata", {})
            if mem_metadata.get("fcm_type") == self.config.crystal_type:
                if category is None or mem_metadata.get("category") == category:
                    crystal_memories.append(mem)
        
        return crystal_memories[:limit]
    
    def get_count(self) -> int:
        """Get total crystal count"""
        return self.crystal_count

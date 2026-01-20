"""
FCM V2 Liquid Layer Implementation
==================================

Cải tiến 2: Attention Sinks & Semantic Grouping

Attention Sinks:
- Luôn giữ lại K tin nhắn đầu tiên của phiên (System prompt + Greeting)
- Tin nhắn này không bị trôi đi khi buffer đầy

Semantic Grouping:
- Trước khi gọi LLM check topic shift, tính cosine similarity
- Chỉ gọi LLM nếu similarity < threshold (tiết kiệm chi phí)
"""

import logging
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from fcm.v2.schemas.base import LiquidMessage, TopicShiftResult
from fcm.v2.prompts import TOPIC_SHIFT_DETECTION_PROMPT

logger = logging.getLogger(__name__)


class LiquidLayer:
    """
    Liquid Layer - High-frequency memory storage
    
    Cải tiến:
    1. Attention Sinks: Giữ K tin nhắn đầu tiên
    2. Semantic Grouping: Dùng embedding similarity trước LLM
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
        
        # Attention Sinks: Lưu K tin nhắn đầu tiên
        self.attention_sinks: List[LiquidMessage] = []
        
        # Conversation buffer (không bao gồm attention sinks)
        self.conversation_buffer: List[LiquidMessage] = []
        
        # Embedding cache cho Semantic Grouping
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Statistics
        self.total_messages = 0
        self.last_crystallize = 0
        
    def _log(self, message: str, level: str = "info"):
        """Helper logging"""
        if self.verbose:
            prefix = "[Liquid]"
            if level == "info":
                logger.info(f"{prefix} {message}")
                print(f"{prefix} {message}")
            elif level == "warning":
                logger.warning(f"{prefix} {message}")
                print(f"{prefix} ⚠ {message}")
            elif level == "error":
                logger.error(f"{prefix} {message}")
                print(f"{prefix} ❌ {message}")
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Lấy embedding cho text (với caching)
        """
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            # Sử dụng embedder từ mem0
            if hasattr(self.memory, 'embedding_model'):
                embedding = self.memory.embedding_model.embed(text)
            else:
                # Fallback: Dùng mem0 internal
                embedding = self.memory.vector_store.embedding_model.embed(text)
            
            # Cache
            self._embedding_cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            self._log(f"Embedding error: {e}", "error")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Tính cosine similarity giữa 2 vectors"""
        try:
            a = np.array(vec1)
            b = np.array(vec2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        except Exception:
            return 0.5  # Default nếu lỗi
    
    def _get_buffer_average_embedding(self) -> Optional[List[float]]:
        """
        Tính trung bình embedding của buffer hiện tại
        """
        if not self.conversation_buffer:
            return None
        
        embeddings = []
        for msg in self.conversation_buffer[-5:]:  # Lấy 5 tin nhắn gần nhất
            if msg.embedding:
                embeddings.append(msg.embedding)
            else:
                emb = self._get_embedding(msg.content)
                if emb:
                    embeddings.append(emb)
        
        if not embeddings:
            return None
        
        # Tính trung bình
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        return avg_embedding
    
    def _detect_topic_shift_with_embedding(
        self, 
        new_message: str
    ) -> Tuple[bool, float]:
        """
        Cải tiến 2: Kiểm tra topic shift bằng embedding similarity trước
        
        Returns:
            (should_call_llm, embedding_similarity)
        """
        # Lấy embedding của tin nhắn mới
        new_embedding = self._get_embedding(new_message)
        if not new_embedding:
            return True, 0.5  # Fallback: gọi LLM
        
        # Lấy average embedding của buffer
        buffer_avg = self._get_buffer_average_embedding()
        if not buffer_avg:
            return False, 1.0  # Buffer rỗng, coi như cùng topic
        
        # Tính cosine similarity
        similarity = self._cosine_similarity(new_embedding, buffer_avg)
        
        self._log(f"[Semantic] Embedding similarity: {similarity:.3f} (threshold: {self.config.topic_shift_embedding_threshold})")
        
        # Nếu similarity cao → cùng topic → không cần gọi LLM
        should_call_llm = similarity < self.config.topic_shift_embedding_threshold
        
        return should_call_llm, similarity
    
    def detect_topic_shift(
        self, 
        new_message: str,
        force_llm: bool = False
    ) -> TopicShiftResult:
        """
        Phát hiện Topic Shift với tối ưu chi phí:
        1. Tính embedding similarity trước
        2. Chỉ gọi LLM nếu similarity thấp
        
        Args:
            new_message: Tin nhắn mới
            force_llm: Bỏ qua embedding check, gọi LLM trực tiếp
            
        Returns:
            TopicShiftResult
        """
        # Check có đủ context không
        if len(self.conversation_buffer) < 2:
            return TopicShiftResult(
                is_new_topic=False,
                confidence=1.0,
                reason="Not enough context",
                embedding_similarity=None,
                used_llm=False
            )
        
        # Step 1: Check embedding similarity
        should_call_llm = True
        embedding_similarity = None
        
        if not force_llm:
            should_call_llm, embedding_similarity = self._detect_topic_shift_with_embedding(new_message)
            
            if not should_call_llm:
                # Similarity cao → cùng topic → skip LLM
                self._log(f"[Semantic] High similarity ({embedding_similarity:.3f}) - Skipping LLM call")
                return TopicShiftResult(
                    is_new_topic=False,
                    confidence=embedding_similarity,
                    reason=f"High embedding similarity: {embedding_similarity:.3f}",
                    embedding_similarity=embedding_similarity,
                    used_llm=False
                )
        
        # Step 2: Gọi LLM check topic shift
        sim_str = f"{embedding_similarity:.3f}" if embedding_similarity is not None else "N/A"
        self._log(f"[Semantic] Low similarity ({sim_str}) - Calling LLM")
        
        try:
            # Lấy context từ buffer
            context_text = "\n".join([
                f"- {m.content}" 
                for m in self.conversation_buffer[-3:]
            ])
            
            prompt = TOPIC_SHIFT_DETECTION_PROMPT.format(
                current_context=context_text,
                new_message=new_message
            )
            
            response = self.memory.llm.generate_response(
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            from fcm.v1.utils import extract_json_from_text
            result = extract_json_from_text(response)
            
            if result and isinstance(result, dict):
                decision = result.get("decision", "SAME_TOPIC")
                confidence = float(result.get("confidence", 0.5))
                reason = result.get("reason", "")
                
                return TopicShiftResult(
                    is_new_topic=(decision == "NEW_TOPIC"),
                    confidence=confidence,
                    old_topic=result.get("old_topic"),
                    new_topic=result.get("new_topic"),
                    reason=reason,
                    embedding_similarity=embedding_similarity,
                    used_llm=True
                )
            
        except Exception as e:
            self._log(f"Topic detection error: {e}", "error")
        
        # Fallback
        return TopicShiftResult(
            is_new_topic=False,
            confidence=0.5,
            reason="LLM error fallback",
            embedding_similarity=embedding_similarity,
            used_llm=True
        )
    
    def add_message(
        self,
        content: str,
        role: str = "user",
        detect_topic_shift: bool = True,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Thêm tin nhắn vào Liquid Layer
        
        Cải tiến 1: Attention Sinks
        - K tin nhắn đầu tiên được đánh dấu là attention_sink
        - Các tin nhắn này luôn được giữ lại
        
        Returns:
            Dict với kết quả và thông tin topic shift
        """
        self.total_messages += 1
        
        # Kiểm tra có phải attention sink không
        is_attention_sink = self.total_messages <= self.config.attention_sink_count
        
        # Tạo LiquidMessage
        message = LiquidMessage(
            content=content,
            role=role,
            timestamp=datetime.now(),
            message_index=self.total_messages,
            is_attention_sink=is_attention_sink
        )
        
        # Lấy embedding cho message
        embedding = self._get_embedding(content)
        if embedding:
            message.embedding = embedding
        
        # Topic shift detection (chỉ cho non-attention-sink messages)
        topic_shifted = False
        shift_result = None
        
        if detect_topic_shift and not is_attention_sink and len(self.conversation_buffer) >= 2:
            shift_result = self.detect_topic_shift(content)
            
            if shift_result.is_new_topic and shift_result.confidence >= 0.7:
                topic_shifted = True
                self._log(f"⚡ Topic Shift detected! (conf={shift_result.confidence:.2f}, used_llm={shift_result.used_llm})")
        
        # Chuẩn bị metadata
        metadata = message.to_metadata()
        metadata["topic_shifted"] = topic_shifted
        if extra_metadata:
            metadata.update(extra_metadata)
        
        # Lưu vào mem0
        result = self.memory.add(
            content,
            user_id=self.user_id,
            metadata=metadata,
            infer=False
        )
        
        # Update local state
        if is_attention_sink:
            self.attention_sinks.append(message)
            self._log(f"💎 Attention Sink #{len(self.attention_sinks)}: '{content[:50]}...'")
        else:
            self.conversation_buffer.append(message)
            self._log(f"Saved: '{content[:50]}...' (Buffer: {len(self.conversation_buffer)})")
        
        return {
            "layer": "liquid",
            "content": content,
            "message_index": self.total_messages,
            "is_attention_sink": is_attention_sink,
            "topic_shifted": topic_shifted,
            "shift_info": shift_result.model_dump() if shift_result else None,
            "result": result
        }
    
    def get_messages(
        self,
        limit: int = 10,
        include_attention_sinks: bool = True,
        status: str = "raw"
    ) -> List[Dict[str, Any]]:
        """
        Lấy tin nhắn từ Liquid layer
        
        Args:
            limit: Số lượng tối đa
            include_attention_sinks: Có bao gồm attention sinks không
            status: "raw", "processed", "all"
        """
        all_memories = self.memory.get_all(
            user_id=self.user_id,
            limit=limit * 2
        )
        
        liquid_memories = []
        for mem in all_memories.get("results", []):
            mem_metadata = mem.get("metadata", {})
            if mem_metadata.get("fcm_type") == self.config.liquid_type:
                if status == "all" or mem_metadata.get("fcm_status") == status:
                    # Filter attention sinks nếu cần
                    if not include_attention_sinks and mem_metadata.get("is_attention_sink"):
                        continue
                    liquid_memories.append(mem)
        
        return liquid_memories[:limit]
    
    def get_buffer_for_crystallize(self) -> List[Dict[str, Any]]:
        """
        Lấy buffer để crystallize (không bao gồm attention sinks)
        """
        return [
            {"content": m.content, "role": m.role, "timestamp": m.timestamp.isoformat()}
            for m in self.conversation_buffer
        ]
    
    def get_full_context(self) -> List[Dict[str, Any]]:
        """
        Lấy full context bao gồm attention sinks + buffer
        (để đưa vào LLM context)
        """
        all_messages = []
        
        # Attention sinks đầu tiên
        for m in self.attention_sinks:
            all_messages.append({
                "content": m.content,
                "role": m.role,
                "is_attention_sink": True
            })
        
        # Buffer
        for m in self.conversation_buffer:
            all_messages.append({
                "content": m.content,
                "role": m.role,
                "is_attention_sink": False
            })
        
        return all_messages
    
    def clear_buffer(self):
        """Clear buffer sau khi crystallize (giữ attention sinks)"""
        self.conversation_buffer = []
        self.last_crystallize = self.total_messages
        self._log("Buffer cleared (Attention Sinks preserved)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy statistics"""
        return {
            "total_messages": self.total_messages,
            "attention_sinks_count": len(self.attention_sinks),
            "buffer_size": len(self.conversation_buffer),
            "last_crystallize": self.last_crystallize,
            "embedding_cache_size": len(self._embedding_cache)
        }

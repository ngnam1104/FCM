"""
FCM V1 Liquid Layer
====================

High-Frequency Memory Layer với Topic Shift Detection (SeCom).
Dựa trên: Nested Learning / InfLLM, SeCom Semantic Segmentation.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from mem0 import Memory
from fcm.v1.config import FCMConfig
from fcm.v1.schemas import LiquidMessage, TopicShiftResult, CompressionResult
from fcm.v1.utils import extract_json_from_text
from fcm.v1.prompts import TOPIC_SHIFT_DETECTION_PROMPT, CONVERSATION_COMPRESSION_PROMPT

logger = logging.getLogger(__name__)


class LiquidLayer:
    """
    Liquid Layer - High-Frequency Memory Storage
    
    Xử lý tin nhắn thô sơ cấp (Raw Input Processing):
    - Lưu trữ ngay lập tức mọi tin nhắn
    - Không xử lý, giữ nguyên context
    - Frequency: t=1 (mỗi tin nhắn)
    
    SeCom Integration:
    - Topic Shift Detection để trigger Crystallize sớm
    - Semantic Segmentation cho phân đoạn hội thoại
    - Conversation Compression để loại bỏ noise
    """
    
    def __init__(self, memory: Memory, config: FCMConfig, user_id: str, verbose: bool = True):
        """
        Khởi tạo Liquid Layer
        
        Args:
            memory: mem0 Memory instance
            config: FCMConfig
            user_id: User ID
            verbose: Enable logging
        """
        self.memory = memory
        self.config = config
        self.user_id = user_id
        self.verbose = verbose
        
        # Buffer cho conversation hiện tại
        self.conversation_buffer: List[Dict[str, Any]] = []
        self.message_count = 0
    
    def _log(self, message: str, level: str = "info"):
        """Helper logging với verbose check"""
        if self.verbose:
            if level == "info":
                logger.info(f"[Liquid] {message}")
                print(f"[Liquid] {message}")
            elif level == "debug":
                logger.debug(f"[Liquid] {message}")
            elif level == "error":
                logger.error(f"[Liquid] {message}")
                print(f"[Liquid ERROR] {message}")
    
    def add(self, content: str, role: str = "user",
            extra_metadata: Optional[Dict[str, Any]] = None,
            detect_topic_shift: bool = True,
            on_topic_shift: Optional[callable] = None) -> Dict[str, Any]:
        """
        Lưu tin nhắn vào Liquid Layer (raw storage)
        
        [Nested Learning / InfLLM]: High-frequency layer xử lý raw input
        [SeCom]: Tích hợp Topic Shift Detection để trigger Crystallize sớm
        
        Args:
            content: Nội dung tin nhắn
            role: "user" hoặc "assistant"
            extra_metadata: Metadata bổ sung
            detect_topic_shift: Có detect topic shift không (default True)
            on_topic_shift: Callback khi phát hiện topic shift (gọi crystallize)
            
        Returns:
            Dict với thông tin memory đã lưu và topic_shifted flag
        """
        topic_shifted = False
        shift_result = None
        
        # [SeCom] Detect Topic Shift TRƯỚC khi lưu
        if detect_topic_shift and len(self.conversation_buffer) >= 2:
            current_context = "\n".join([
                f"{m['role']}: {m['content']}" 
                for m in self.conversation_buffer[-5:]
            ])
            
            shift_result = self._detect_topic_shift(current_context, content)
            
            if shift_result.get("is_new_topic", False):
                confidence = shift_result.get("confidence", 0)
                if confidence >= 0.7:
                    topic_shifted = True
                    reason = shift_result.get("reason", "No reason provided")
                    self._log(f"[SeCom] ⚡ Topic Shift detected! Reason: '{reason}' (conf={confidence})")
                    
                    # Trigger Crystallize via callback
                    if on_topic_shift:
                        self._log(f"[SeCom] 💡 Triggering crystallize due to Topic Shift")
                        on_topic_shift()
        
        # Build metadata
        metadata = {
            "fcm_type": self.config.liquid_type,
            "fcm_frequency": 1,
            "fcm_status": "raw",
            "role": role,
            "message_index": self.message_count,
            "timestamp": datetime.now().isoformat(),
            "topic_shifted": topic_shifted,
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        # Lưu vào mem0
        result = self.memory.add(
            content,
            user_id=self.user_id,
            metadata=metadata,
            infer=False
        )
        
        # Update state
        self.message_count += 1
        self.conversation_buffer.append({
            "content": content,
            "role": role,
            "index": self.message_count,
            "timestamp": metadata["timestamp"]
        })
        
        self._log(f"Saved: '{content[:50]}...' (Total: {self.message_count})")
        
        return {
            "layer": "liquid",
            "content": content,
            "message_index": self.message_count,
            "result": result,
            "topic_shifted": topic_shifted,
            "shift_info": shift_result if topic_shifted else None
        }
    
    def get(self, limit: int = 10, status: str = "raw") -> List[Dict[str, Any]]:
        """
        Lấy các Liquid memories
        
        Args:
            limit: Số lượng tối đa
            status: "raw" (chưa xử lý) hoặc "processed" (đã crystallize) hoặc "all"
            
        Returns:
            List các liquid memories
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
                    liquid_memories.append(mem)
        
        return liquid_memories[:limit]
    
    def clear_buffer(self):
        """Clear conversation buffer (sau khi crystallize)"""
        self.conversation_buffer = []
    
    def get_buffer(self) -> List[Dict[str, Any]]:
        """Get current conversation buffer"""
        return self.conversation_buffer.copy()
    
    def get_buffer_size(self) -> int:
        """Get buffer size"""
        return len(self.conversation_buffer)
    
    # =========================================================================
    # SeCom: Topic Shift Detection
    # =========================================================================
    
    def _detect_topic_shift(self, current_context: Union[List[str], str], 
                            new_message: str) -> Dict[str, Any]:
        """
        [SeCom] Phát hiện Topic Shift sử dụng LLM với cơ chế Fallback mạnh mẽ.
        """
        try:
            if isinstance(current_context, list):
                context_text = "\n".join(current_context[-3:])
            else:
                context_text = str(current_context)[-500:]
            
            prompt = TOPIC_SHIFT_DETECTION_PROMPT.format(
                current_context=context_text,
                new_message=new_message
            )

            response = self.memory.llm.generate_response(
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            decision = "SAME_TOPIC"
            reason = ""
            confidence = 0.5

            json_result = extract_json_from_text(response)
            
            if json_result and isinstance(json_result, dict):
                decision = json_result.get("decision", "SAME_TOPIC")
                reason = json_result.get("reason", "")
                confidence = float(json_result.get("confidence", 0.9))
            else:
                response_upper = str(response).upper()
                if "NEW_TOPIC" in response_upper:
                    decision = "NEW_TOPIC"
                    reason = "Detected via text fallback"
                    confidence = 0.7
                elif "SAME_TOPIC" in response_upper:
                    decision = "SAME_TOPIC"
            
            return {
                "is_new_topic": decision == "NEW_TOPIC",
                "confidence": confidence,
                "reason": reason
            }

        except Exception as e:
            self._log(f"[SeCom] Topic detection error: {e}", "error")
            return {"is_new_topic": False, "confidence": 0.0, "reason": "Error fallback"}
    
    def segment_conversation(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        [SeCom] Semantic Segmentation: Phân đoạn hội thoại dựa trên Topic Shift
        
        Returns:
            List of segments, mỗi segment là list of messages cùng topic
        """
        if not messages or len(messages) <= 2:
            return [messages] if messages else []
        
        segments = []
        current_segment = [messages[0]]
        
        for i, msg in enumerate(messages[1:], start=1):
            current_context = "\n".join([
                f"- {m.get('memory', m.get('content', ''))}"
                for m in current_segment[-3:]
            ])
            
            new_message = msg.get('memory', msg.get('content', ''))
            detection = self._detect_topic_shift(current_context, new_message)
            
            if detection["is_new_topic"] and detection["confidence"] >= 0.7:
                if current_segment:
                    segments.append(current_segment)
                    self._log(f"[SeCom] Topic shift detected: '{detection['reason']}'")
                current_segment = [msg]
            else:
                current_segment.append(msg)
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    # =========================================================================
    # SeCom: Conversation Compression
    # =========================================================================
    
    def compress_conversation(self, messages: List[Dict[str, Any]]) -> CompressionResult:
        """
        [SeCom/COMEDY] Compressive Memory: Nén hội thoại thành văn xuôi
        
        Returns:
            CompressionResult với compressed_narrative và metadata
        """
        chat_log = "\n".join([
            f"[{msg.get('role', 'user').upper()}]: {msg.get('memory', msg.get('content', ''))}"
            for msg in messages
        ])
        
        # Nếu ít messages (<= 5), không compress
        if len(messages) <= 5:
            self._log(f"[SeCom] Skip compression for {len(messages)} messages (<=5)")
            return CompressionResult(
                compressed_narrative=chat_log,
                key_entities=[],
                noise_count=0,
                original_length=len(chat_log),
                compressed_length=len(chat_log)
            )
        
        try:
            response = self.memory.llm.generate_response(
                messages=[{
                    "role": "user",
                    "content": CONVERSATION_COMPRESSION_PROMPT.format(chat_log=chat_log)
                }],
                response_format={"type": "json_object"}
            )
            
            result = extract_json_from_text(response)
            
            compressed = result.get("compressed_narrative", "") if result else ""
            entities = result.get("key_entities", []) if result else []
            noise_count = result.get("noise_count", 0) if result else 0
            
            self._log(f"[SeCom] Compressed: {len(chat_log)} → {len(compressed)} chars")
            
            return CompressionResult(
                compressed_narrative=compressed,
                key_entities=entities,
                noise_count=noise_count,
                original_length=len(chat_log),
                compressed_length=len(compressed)
            )
            
        except Exception as e:
            self._log(f"[SeCom] Compression error: {e}", "error")
            return CompressionResult(
                compressed_narrative=chat_log,
                key_entities=[],
                noise_count=0,
                original_length=len(chat_log),
                compressed_length=len(chat_log)
            )
    
    @staticmethod
    def is_noise_message(content: str) -> bool:
        """
        [SeCom] Quick Noise Check (Regex-based fallback)
        """
        import re
        
        content_lower = content.strip().lower()
        
        noise_patterns = [
            r'^(xin\s+)?chào(\s+bạn)?[!.]?$',
            r'^hi[!.]?$',
            r'^hello[!.]?$',
            r'^ok[!.]?$',
            r'^ừ[m]?[!.]?$',
            r'^à[!.]?$',
            r'^ờ[!.]?$',
            r'^(cảm\s+ơn|thanks?|thank\s+you)[!.]?$',
            r'^bye[!.]?$',
            r'^tạm\s+biệt[!.]?$',
            r'^được[!.]?$',
            r'^vâng[!.]?$',
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, content_lower):
                return True
        
        if len(content_lower) < 5:
            return True
        
        return False

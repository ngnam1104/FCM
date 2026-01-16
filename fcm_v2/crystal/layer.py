"""
FCM V2 Crystal Layer Implementation
====================================

Cải tiến 1: Bi-Temporal Schema

Bi-Temporal:
- valid_at: Thời gian sự kiện XẢY RA (trích xuất từ text)
- observed_at: Thời gian hệ thống GHI NHẬN

Ví dụ: "Năm 2018 tôi làm ở Google"
- valid_at: "2018" (khi sự kiện xảy ra)
- observed_at: "2024-01-15" (khi bot nghe được)

Lợi ích:
- Trả lời câu hỏi "Năm 2018 bạn làm gì?" chính xác hơn
- Phân biệt được thông tin hiện tại vs quá khứ
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fcm_v2.schemas.base import (
    CrystalFact, 
    MemoryStrength, 
    MemoryCategory,
    CompressionResult
)
from fcm_v2.prompts import (
    CRYSTALLIZER_SYSTEM_PROMPT_V2,
    CRYSTALLIZER_USER_PROMPT_TEMPLATE_V2,
    CONVERSATION_COMPRESSION_PROMPT
)

logger = logging.getLogger(__name__)


class CrystalLayer:
    """
    Crystal Layer - Mid-frequency memory storage với Bi-Temporal support
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
        
        # Statistics
        self.crystal_count = 0
        
    def _log(self, message: str, level: str = "info"):
        """Helper logging"""
        if self.verbose:
            prefix = "[Crystal]"
            if level == "info":
                logger.info(f"{prefix} {message}")
                print(f"{prefix} {message}")
            elif level == "warning":
                logger.warning(f"{prefix} {message}")
                print(f"{prefix} ⚠ {message}")
            elif level == "error":
                logger.error(f"{prefix} {message}")
                print(f"{prefix} ❌ {message}")
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON từ LLM response"""
        from fcm.utils import extract_json_from_text
        return extract_json_from_text(response)
    
    def compress_conversation(
        self, 
        messages: List[Dict[str, Any]]
    ) -> CompressionResult:
        """
        Nén hội thoại thành narrative (SeCom/COMEDY)
        
        Args:
            messages: List tin nhắn cần compress
            
        Returns:
            CompressionResult
        """
        # Tạo chat log
        chat_log = "\n".join([
            f"[{msg.get('role', 'user').upper()}]: {msg.get('content', '')}"
            for msg in messages
        ])
        
        original_length = len(chat_log)
        
        # Nếu ít messages (<= 5), không compress - giữ nguyên để không mất chi tiết
        if len(messages) <= 5:
            self._log(f"Skip compression for {len(messages)} messages (<=5)")
            return CompressionResult(
                compressed_narrative=chat_log,
                key_entities=[],
                noise_count=0,
                original_length=original_length,
                compressed_length=original_length,
                compression_ratio=1.0
            )
        
        try:
            response = self.memory.llm.generate_response(
                messages=[{
                    "role": "user",
                    "content": CONVERSATION_COMPRESSION_PROMPT.format(chat_log=chat_log)
                }]
            )
            
            result = self._parse_json_response(response)
            
            if result:
                compressed = result.get("compressed_narrative", "")
                entities = result.get("key_entities", [])
                noise_count = result.get("noise_count", 0)
                
                return CompressionResult(
                    compressed_narrative=compressed,
                    key_entities=entities,
                    noise_count=noise_count,
                    original_length=original_length,
                    compressed_length=len(compressed),
                    compression_ratio=len(compressed) / max(1, original_length)
                )
                
        except Exception as e:
            self._log(f"Compression error: {e}", "error")
        
        # Fallback
        return CompressionResult(
            compressed_narrative=chat_log,
            key_entities=[],
            noise_count=0,
            original_length=original_length,
            compressed_length=original_length,
            compression_ratio=1.0
        )
    
    def extract_facts_with_bitemporal(
        self,
        compressed_narrative: str,
        session_date: Optional[str] = None
    ) -> List[CrystalFact]:
        """
        Trích xuất Atomic Facts với Bi-Temporal information
        
        Cải tiến Option 2: 
        - Nếu có session_date, prompt sẽ tính toán computed_date
        - LLM sẽ chuyển "yesterday" → "7 May 2023" dựa trên session_date
        
        Args:
            compressed_narrative: Đoạn văn đã compress
            session_date: Session date string for temporal computation
            
        Returns:
            List[CrystalFact]
        """
        try:
            # Format session_date for prompt
            session_date_str = session_date if session_date else "Không xác định"
            
            response = self.memory.llm.generate_response(
                messages=[
                    {"role": "system", "content": CRYSTALLIZER_SYSTEM_PROMPT_V2},
                    {"role": "user", "content": CRYSTALLIZER_USER_PROMPT_TEMPLATE_V2.format(
                        chat_log=compressed_narrative,
                        session_date=session_date_str
                    )}
                ]
            )
            
            result = self._parse_json_response(response)
            
            if not result:
                self._log("Failed to parse crystallizer response", "warning")
                return []
            
            # Extract facts
            facts_data = result.get("facts", [])
            if isinstance(result, list):
                facts_data = result
            
            facts = []
            current_time = datetime.now()
            
            for fact_data in facts_data:
                if not isinstance(fact_data, dict):
                    continue
                
                content = fact_data.get("content", "")
                if not content:
                    continue
                
                # Map category
                category_str = fact_data.get("category", "fact")
                try:
                    category = MemoryCategory(category_str)
                except ValueError:
                    # Handle new category: temporal_event
                    if category_str == "temporal_event":
                        category = MemoryCategory.EXPERIENCE  # Map to experience
                    else:
                        category = MemoryCategory.FACT
                
                # Bi-Temporal: Extract valid_at và computed_date (Option 2)
                valid_at = fact_data.get("valid_at")  # Thời gian gốc (e.g., "yesterday")
                computed_date = fact_data.get("computed_date")  # Ngày đã tính (e.g., "7 May 2023")
                
                # Tạo CrystalFact với Bi-Temporal
                fact = CrystalFact(
                    content=content,
                    category=category,
                    valid_at=valid_at,  # ← Bi-Temporal field (original)
                    observed_at=current_time,  # ← Bi-Temporal field
                    confidence=float(fact_data.get("confidence", 0.9)),
                    memory_strength=MemoryStrength(
                        initial_strength=1.0,
                        current_strength=1.0,
                        created_at=current_time
                    ),
                    keywords=fact_data.get("keywords", []),
                    context_tags=fact_data.get("context_tags", []),
                    related_to=fact_data.get("related_to", [])
                )
                
                # Store computed_date as extra attribute for Option 2
                if computed_date:
                    fact.computed_date = computed_date
                
                facts.append(fact)
                
                # Log với Bi-Temporal info (include computed_date if present)
                temporal_info = ""
                if valid_at:
                    temporal_info = f" [valid_at={valid_at}]"
                if computed_date:
                    temporal_info += f" [computed={computed_date}]"
                self._log(f"✓ Extracted: '{content[:60]}...'{temporal_info}")
            
            return facts
            
        except Exception as e:
            self._log(f"Extraction error: {e}", "error")
            return []
    
    def crystallize(
        self,
        messages: List[Dict[str, Any]],
        source_message_ids: Optional[List[str]] = None,
        session_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main Crystallize Pipeline:
        1. Compress conversation
        2. Extract Bi-Temporal facts (with computed dates if session_date provided)
        3. Save to memory
        
        Args:
            messages: Tin nhắn cần crystallize
            source_message_ids: IDs của source messages
            session_date: Session date string for temporal computation (Option 2)
            
        Returns:
            Dict với kết quả
        """
        if not messages:
            return {"status": "skipped", "reason": "no_messages"}
        
        self._log(f"Crystallizing {len(messages)} messages...")
        if session_date:
            self._log(f"Session date for temporal computation: {session_date}")
        
        # Step 1: Compress
        compression_result = self.compress_conversation(messages)
        
        if not compression_result.compressed_narrative.strip():
            return {"status": "skipped", "reason": "empty_compression"}
        
        self._log(f"Compressed: {compression_result.original_length} → {compression_result.compressed_length} chars")
        
        # Step 2: Extract facts với Bi-Temporal (pass session_date)
        facts = self.extract_facts_with_bitemporal(
            compression_result.compressed_narrative,
            session_date=session_date
        )
        
        if not facts:
            return {"status": "skipped", "reason": "no_facts_extracted"}
        
        # Step 3: Save to memory
        crystal_results = []
        
        for fact in facts:
            # Chuẩn bị metadata với Bi-Temporal
            metadata = fact.to_metadata()
            metadata["source_message_ids"] = source_message_ids or []
            metadata["compression_ratio"] = compression_result.compression_ratio
            if session_date:
                metadata["session_date"] = session_date
            
            # Option 2: Include computed_date in metadata if available
            if hasattr(fact, 'computed_date') and fact.computed_date:
                metadata["computed_date"] = fact.computed_date
            
            # Add to mem0
            result = self.memory.add(
                fact.content,
                user_id=self.user_id,
                metadata=metadata,
                infer=False
            )
            
            self.crystal_count += 1
            crystal_results.append({
                "content": fact.content,
                "category": fact.category.value,
                "valid_at": fact.valid_at,  # ← Bi-Temporal original
                "computed_date": getattr(fact, 'computed_date', None),  # ← Option 2
                "result": result
            })
        
        return {
            "status": "success",
            "facts_extracted": len(facts),
            "compression_info": {
                "original_chars": compression_result.original_length,
                "compressed_chars": compression_result.compressed_length,
                "noise_removed": compression_result.noise_count,
                "compression_ratio": compression_result.compression_ratio
            },
            "crystals": crystal_results
        }
    
    def get_memories(
        self,
        limit: int = 20,
        category: Optional[str] = None,
        valid_at_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy Crystal memories với optional Bi-Temporal filter
        
        Args:
            limit: Số lượng tối đa
            category: Filter theo category
            valid_at_filter: Filter theo valid_at (Bi-Temporal)
        """
        all_memories = self.memory.get_all(
            user_id=self.user_id,
            limit=limit * 2
        )
        
        crystal_memories = []
        for mem in all_memories.get("results", []):
            mem_metadata = mem.get("metadata", {})
            
            # Filter by type
            if mem_metadata.get("fcm_type") != self.config.crystal_type:
                continue
            
            # Filter by category
            if category and mem_metadata.get("category") != category:
                continue
            
            # Filter by valid_at (Bi-Temporal)
            if valid_at_filter:
                mem_valid_at = mem_metadata.get("valid_at", "")
                # Simple substring match for now
                if valid_at_filter not in str(mem_valid_at):
                    continue
            
            crystal_memories.append(mem)
        
        return crystal_memories[:limit]
    
    def search_with_temporal_context(
        self,
        query: str,
        temporal_context: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search với Bi-Temporal context
        
        Nếu query có context thời gian, ưu tiên facts có valid_at khớp
        
        Args:
            query: Câu truy vấn
            temporal_context: Context thời gian (e.g., "2018", "tuần trước")
            limit: Số lượng tối đa
        """
        # Search base
        all_results = self.memory.search(
            query,
            user_id=self.user_id,
            limit=limit * 3
        )
        
        crystal_results = []
        for mem in all_results.get("results", []):
            mem_metadata = mem.get("metadata", {})
            
            if mem_metadata.get("fcm_type") != self.config.crystal_type:
                continue
            
            # Add temporal score boost nếu có temporal_context
            score = mem.get("score", 0.5)
            
            if temporal_context and self.config.enable_temporal_priority:
                valid_at = mem_metadata.get("valid_at", "")
                
                if valid_at and temporal_context.lower() in str(valid_at).lower():
                    # Boost score 20% nếu temporal match
                    score *= 1.2
                    mem["temporal_match"] = True
                else:
                    mem["temporal_match"] = False
            
            mem["adjusted_score"] = min(1.0, score)
            crystal_results.append(mem)
        
        # Sort by adjusted score
        crystal_results.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)
        
        return crystal_results[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy statistics"""
        return {
            "crystal_count": self.crystal_count
        }

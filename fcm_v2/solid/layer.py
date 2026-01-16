"""
FCM V2 Solid Layer Implementation
==================================

Cải tiến 3: Active Forgetting (Ebbinghaus Curve)
Cải tiến 4: Dynamic Persona

Active Forgetting:
- Công thức: S(t) = S_0 * e^(-Δt/τ) + R * N_access
- Nếu S < threshold, move to cold storage
- Khi memory được retrieve, reset S = 1.0

Dynamic Persona:
- Trích xuất interaction_style từ conversation
- Cập nhật system prompt dựa trên persona
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from fcm_v2.schemas.base import (
    SolidKnowledge,
    MemoryStrength,
    UserPersona,
    MemoryCategory,
    MemoryStatus,
    ChangeType
)
from fcm_v2.prompts import EVOLVER_SYSTEM_PROMPT_V2, EVOLVER_USER_PROMPT_TEMPLATE_V2

logger = logging.getLogger(__name__)


class SolidLayer:
    """
    Solid Layer - Low-frequency consolidated knowledge
    
    Cải tiến:
    - Active Forgetting: Decay memories theo thời gian
    - Dynamic Persona: Track và update user interaction style
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
        
        # Dynamic Persona (Cải tiến 4)
        self.user_persona: Optional[UserPersona] = self._load_persona()
        
        # Cold Storage path cho Active Forgetting
        self.cold_storage_path = config.cold_storage_path
        os.makedirs(self.cold_storage_path, exist_ok=True)
        
        # Statistics
        self.solid_count = 0
        self.last_evolve = 0
        
    def _log(self, message: str, level: str = "info"):
        """Helper logging"""
        if self.verbose:
            prefix = "[Solid]"
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
    
    def _load_persona(self) -> Optional[UserPersona]:
        """Load persona từ file nếu có"""
        persona_file = os.path.join(
            self.config.cold_storage_path, 
            f"persona_{self.user_id}.json"
        )
        
        if os.path.exists(persona_file):
            try:
                with open(persona_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return UserPersona(**data)
            except Exception as e:
                self._log(f"Failed to load persona: {e}", "warning")
        
        return UserPersona(user_id=self.user_id)
    
    def _save_persona(self):
        """Save persona ra file"""
        if not self.user_persona:
            return
        
        persona_file = os.path.join(
            self.config.cold_storage_path,
            f"persona_{self.user_id}.json"
        )
        
        try:
            with open(persona_file, "w", encoding="utf-8") as f:
                json.dump(self.user_persona.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log(f"Failed to save persona: {e}", "error")
    
    # =========================================================================
    # CẢI TIẾN 3: ACTIVE FORGETTING
    # =========================================================================
    
    def calculate_memory_strength(
        self,
        memory: Dict[str, Any],
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Tính sức mạnh ký ức theo công thức Ebbinghaus:
        
        S(t) = S_0 * e^(-Δt/τ) + R * N_access
        
        Args:
            memory: Memory dict từ mem0
            current_time: Thời điểm hiện tại
            
        Returns:
            Strength score 0.0 - 1.0
        """
        import math
        
        if current_time is None:
            current_time = datetime.now()
        
        metadata = memory.get("metadata", {})
        
        # Lấy các tham số
        initial_strength = 1.0
        decay_constant = self.config.decay_constant_days
        reinforcement_factor = self.config.reinforcement_factor
        access_count = metadata.get("access_count", 0)
        
        # Tính Δt (số ngày từ lần access cuối)
        last_access_str = metadata.get("last_access_at")
        if last_access_str:
            try:
                last_access = datetime.fromisoformat(last_access_str)
            except:
                last_access = datetime.now()
        else:
            # Fallback: dùng evolved_at hoặc now
            evolved_str = metadata.get("evolved_at", metadata.get("last_updated"))
            if evolved_str:
                try:
                    last_access = datetime.fromisoformat(evolved_str)
                except:
                    last_access = datetime.now()
            else:
                last_access = datetime.now()
        
        delta_days = (current_time - last_access).total_seconds() / 86400
        
        # Công thức Ebbinghaus cải biên
        decay_term = initial_strength * math.exp(-delta_days / decay_constant)
        reinforcement_term = reinforcement_factor * access_count
        
        # Clamp về [0, 1]
        strength = min(1.0, max(0.0, decay_term + reinforcement_term))
        
        return strength
    
    def prune_memories(self) -> Dict[str, Any]:
        """
        Cải tiến 3: Prune weak memories (Active Forgetting)
        
        - Duyệt qua tất cả solid memories
        - Tính strength cho mỗi memory
        - Nếu strength < threshold → move to cold storage
        
        Returns:
            Dict với số lượng đã prune
        """
        if not self.config.enable_active_forgetting:
            return {"status": "disabled", "pruned": 0}
        
        self._log("[Active Forgetting] Starting memory pruning...")
        
        # Lấy tất cả solid memories
        all_memories = self.memory.get_all(
            user_id=self.user_id,
            limit=100
        )
        
        pruned_count = 0
        cold_storage_file = os.path.join(
            self.cold_storage_path,
            f"cold_storage_{self.user_id}_{datetime.now().strftime('%Y%m')}.jsonl"
        )
        
        current_time = datetime.now()
        
        for mem in all_memories.get("results", []):
            metadata = mem.get("metadata", {})
            
            # Chỉ xử lý solid memories active
            if metadata.get("fcm_type") != self.config.solid_type:
                continue
            if metadata.get("fcm_status") == "archived":
                continue
            
            # Tính strength
            strength = self.calculate_memory_strength(mem, current_time)
            
            if strength < self.config.forget_threshold:
                # Move to cold storage
                self._move_to_cold_storage(mem, cold_storage_file, strength)
                pruned_count += 1
                self._log(f"🗑 Pruned: '{mem.get('memory', '')[:50]}...' (strength={strength:.3f})")
        
        self._log(f"[Active Forgetting] Pruned {pruned_count} weak memories")
        
        return {
            "status": "success",
            "pruned": pruned_count,
            "threshold": self.config.forget_threshold
        }
    
    def _move_to_cold_storage(
        self,
        memory: Dict[str, Any],
        cold_storage_file: str,
        strength: float
    ):
        """
        Move memory to cold storage (file log)
        
        Args:
            memory: Memory dict
            cold_storage_file: Path to cold storage file
            strength: Current strength score
        """
        try:
            # Chuẩn bị cold storage entry
            cold_entry = {
                "original_id": memory.get("id"),
                "content": memory.get("memory", ""),
                "metadata": memory.get("metadata", {}),
                "strength_at_prune": strength,
                "pruned_at": datetime.now().isoformat(),
                "reason": "active_forgetting"
            }
            
            # Append to file
            with open(cold_storage_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(cold_entry, ensure_ascii=False) + "\n")
            
            # Delete from vector store
            memory_id = memory.get("id")
            if memory_id:
                self.memory.delete(memory_id)
                
        except Exception as e:
            self._log(f"Failed to move to cold storage: {e}", "error")
    
    def on_memory_accessed(self, memory_id: str) -> bool:
        """
        Gọi khi một memory được retrieve (truy xuất)
        → Reset strength về 1.0 (củng cố ký ức)
        
        Args:
            memory_id: ID của memory được access
            
        Returns:
            True nếu update thành công
        """
        try:
            # Lấy memory hiện tại
            mem = self.memory.get(memory_id)
            if not mem:
                return False
            
            metadata = mem.get("metadata", {})
            
            # Update access count và timestamp
            access_count = metadata.get("access_count", 0) + 1
            
            # Trong mem0, không có update trực tiếp metadata
            # Workaround: Xóa và thêm lại với metadata mới
            content = mem.get("memory", "")
            new_metadata = metadata.copy()
            new_metadata["access_count"] = access_count
            new_metadata["last_access_at"] = datetime.now().isoformat()
            new_metadata["decay_score"] = 1.0  # Reset strength
            
            # Delete old
            self.memory.delete(memory_id)
            
            # Add new
            self.memory.add(
                content,
                user_id=self.user_id,
                metadata=new_metadata,
                infer=False
            )
            
            self._log(f"🔄 Memory reinforced (access #{access_count}): '{content[:40]}...'")
            return True
            
        except Exception as e:
            self._log(f"Failed to reinforce memory: {e}", "error")
            return False
    
    # =========================================================================
    # CẢI TIẾN 4: DYNAMIC PERSONA
    # =========================================================================
    
    def update_persona_from_evolution(
        self,
        interaction_style: Dict[str, Any]
    ):
        """
        Cập nhật User Persona từ kết quả evolution
        
        Args:
            interaction_style: Dict với communication_style, humor_level, etc.
        """
        if not self.config.enable_dynamic_persona:
            return
        
        if not self.user_persona:
            self.user_persona = UserPersona(user_id=self.user_id)
        
        # Update các field
        if "communication_style" in interaction_style:
            self.user_persona.communication_style = interaction_style["communication_style"]
        
        if "preferred_response_length" in interaction_style:
            self.user_persona.preferred_response_length = interaction_style["preferred_response_length"]
        
        if "humor_level" in interaction_style:
            self.user_persona.humor_level = float(interaction_style["humor_level"])
        
        if "inferred_traits" in interaction_style:
            # Merge traits
            new_traits = interaction_style["inferred_traits"]
            existing = set(self.user_persona.inferred_traits)
            existing.update(new_traits)
            self.user_persona.inferred_traits = list(existing)[:10]  # Giữ 10 traits
        
        if "topics_of_interest" in interaction_style:
            new_topics = interaction_style["topics_of_interest"]
            existing = set(self.user_persona.topics_of_interest)
            existing.update(new_topics)
            self.user_persona.topics_of_interest = list(existing)[:10]
        
        # Update familiarity
        self.user_persona.update_familiarity(self.config.familiarity_increment)
        
        # Save
        self._save_persona()
        
        self._log(f"🎭 Persona updated: style={self.user_persona.communication_style}, familiarity={self.user_persona.familiarity_level:.2f}")
    
    def get_persona_prompt_injection(self) -> str:
        """
        Lấy đoạn text để inject vào system prompt
        """
        if not self.user_persona:
            return ""
        
        return self.user_persona.to_system_prompt_injection()
    
    # =========================================================================
    # EVOLVE
    # =========================================================================
    
    def evolve(
        self,
        crystal_facts: List[Dict[str, Any]],
        conversation_samples: Optional[List[str]] = None,
        total_messages: int = 0
    ) -> Dict[str, Any]:
        """
        Evolve: Hợp nhất Crystal → Solid với:
        1. MAPLE versioning
        2. Interaction Style extraction (Dynamic Persona)
        3. Active Forgetting trigger
        
        Args:
            crystal_facts: List Crystal facts mới
            conversation_samples: Mẫu hội thoại để phân tích style
            total_messages: Tổng số messages (để check threshold)
            
        Returns:
            Dict với kết quả evolution
        """
        if not crystal_facts:
            return {"status": "skipped", "reason": "no_crystal_facts"}
        
        self._log(f"Evolving {len(crystal_facts)} crystal facts...")
        
        # Lấy solid memories hiện tại
        solid_memories = self.get_memories(limit=50, status="active")
        
        # Chuẩn bị prompt
        crystal_text = "\n".join([
            f"- [{m.get('metadata', {}).get('category', 'general')}] {m.get('memory', '')}"
            for m in crystal_facts
        ])
        
        solid_text = "\n".join([
            f"- [ID:{m.get('id')}] {m.get('memory', '')}"
            for m in solid_memories
        ]) if solid_memories else "(Chưa có thông tin)"
        
        conv_samples = "\n".join(conversation_samples[:10]) if conversation_samples else "(Không có mẫu)"
        
        current_timestamp = datetime.now().isoformat()
        
        try:
            response = self.memory.llm.generate_response(
                messages=[
                    {"role": "system", "content": EVOLVER_SYSTEM_PROMPT_V2},
                    {"role": "user", "content": EVOLVER_USER_PROMPT_TEMPLATE_V2.format(
                        solid_facts=solid_text,
                        crystal_facts=crystal_text,
                        conversation_samples=conv_samples
                    )}
                ]
            )
            
            result = self._parse_json_response(response)
            
            if not result:
                self._log("Failed to parse evolver response", "warning")
                return {"status": "failed_parse"}
            
            # Process updates
            updates = result.get("updates", [])
            processed_count = 0
            archived_count = 0
            
            for item in updates:
                if not isinstance(item, dict):
                    continue
                
                action = item.get("action", "IGNORE").upper()
                content = item.get("content")
                
                if not content or action == "IGNORE":
                    continue
                
                # Metadata với Memory Strength tracking
                new_metadata = {
                    "fcm_type": "solid",
                    "fcm_frequency": 3,
                    "fcm_status": "active",
                    "category": item.get("category", "general"),
                    "confidence": item.get("confidence", 1.0),
                    "reason": item.get("reason", "Evolution"),
                    "change_type": item.get("change_type", "SUPPLEMENT"),
                    "last_updated": current_timestamp,
                    "supersedes": None,
                    # Active Forgetting fields
                    "access_count": 0,
                    "last_access_at": current_timestamp,
                    "decay_score": 1.0
                }
                
                # Handle UPDATE
                if action == "UPDATE":
                    old_id = item.get("supersedes")
                    if old_id:
                        old_id = str(old_id).replace("ID:", "").strip()
                        if self._archive_memory(old_id, current_timestamp, content):
                            new_metadata["supersedes"] = old_id
                            archived_count += 1
                
                # Save new
                self.memory.add(
                    content,
                    user_id=self.user_id,
                    metadata=new_metadata,
                    infer=False
                )
                
                icon = "🔄" if action == "UPDATE" else "➕"
                self._log(f"{icon} {action}: {content[:60]}...")
                processed_count += 1
            
            # Cải tiến 4: Update Dynamic Persona
            interaction_style = result.get("interaction_style", {})
            if interaction_style:
                self.update_persona_from_evolution(interaction_style)
            
            # Cải tiến 3: Run Active Forgetting
            prune_result = self.prune_memories()
            
            self.solid_count += processed_count
            self.last_evolve = total_messages
            
            return {
                "status": "success",
                "processed": processed_count,
                "archived": archived_count,
                "pruned": prune_result.get("pruned", 0),
                "persona_updated": bool(interaction_style)
            }
            
        except Exception as e:
            self._log(f"Evolution error: {e}", "error")
            return {"status": "error", "error": str(e)}
    
    def _archive_memory(
        self,
        memory_id: str,
        valid_until: str,
        superseded_by: str
    ) -> bool:
        """Archive memory (MAPLE versioning)"""
        try:
            old_mem = self.memory.get(memory_id)
            if not old_mem:
                return False
            
            content = old_mem.get("memory", "")
            old_meta = old_mem.get("metadata", {})
            
            # Delete old
            self.memory.delete(memory_id)
            
            # Add archived version
            new_meta = old_meta.copy()
            new_meta.update({
                "fcm_status": "archived",
                "original_memory_id": memory_id,
                "valid_until": valid_until,
                "superseded_by": superseded_by,
                "archived_at": datetime.now().isoformat()
            })
            
            self.memory.add(
                f"[ARCHIVED] {content}",
                user_id=self.user_id,
                metadata=new_meta,
                infer=False
            )
            
            return True
            
        except Exception as e:
            self._log(f"Archive error: {e}", "error")
            return False
    
    def get_memories(
        self,
        limit: int = 30,
        section: Optional[str] = None,
        status: str = "active"
    ) -> List[Dict[str, Any]]:
        """Lấy Solid memories"""
        all_memories = self.memory.get_all(
            user_id=self.user_id,
            limit=limit * 3
        )
        
        solid_memories = []
        for mem in all_memories.get("results", []):
            metadata = mem.get("metadata", {})
            
            if metadata.get("fcm_type") != self.config.solid_type:
                continue
            
            mem_status = metadata.get("fcm_status", "active")
            if status != "all" and mem_status != status:
                continue
            
            if section and metadata.get("profile_section") != section:
                continue
            
            solid_memories.append(mem)
        
        return solid_memories[:limit]
    
    def get_user_profile(self) -> Dict[str, List[str]]:
        """Lấy User Profile organized by sections"""
        solid_memories = self.get_memories(limit=100, status="active")
        
        profile = {
            "personal_info": [],
            "preferences": [],
            "relationships": [],
            "plans": [],
            "experiences": [],
            "other": []
        }
        
        for mem in solid_memories:
            section = mem.get("metadata", {}).get("profile_section", "other")
            content = mem.get("memory", "")
            
            if section in profile:
                profile[section].append(content)
            else:
                profile["other"].append(content)
        
        return profile
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy statistics"""
        return {
            "solid_count": self.solid_count,
            "last_evolve": self.last_evolve,
            "persona": self.user_persona.model_dump() if self.user_persona else None
        }

"""
FCM V1 Solid Layer
===================

Low-Frequency Memory Layer - The Evolver
Hợp nhất Crystal facts thành consolidated knowledge.

Dựa trên:
- MAPLE Archiver: Session-level consolidation
- G-Memory: Version tracking với linked list of knowledge
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from mem0 import Memory
from fcm.config import FCMConfig
from fcm.schemas import SolidKnowledge
from fcm.utils import extract_json_from_text
from fcm.prompts import EVOLVER_SYSTEM_PROMPT, EVOLVER_USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class SolidLayer:
    """
    Solid Layer - Low-Frequency Knowledge Consolidation
    
    MAPLE Archiver Concepts Applied:
    - Session Summarization: Tóm tắt phiên làm việc
    - Conflict Detection: Phát hiện mâu thuẫn giữa thông tin mới và cũ
    - Merge Decision: Quyết định có nên update bộ nhớ dài hạn không
    - Profile Evolution: Cập nhật User Profile theo thời gian
    
    G-Memory Versioning:
    - Tạo linked list of knowledge: new_fact -> supersedes -> old_fact
    - Archive old facts thay vì xóa: fcm_status = "archived"
    - Truy vết lịch sử: previous_version, valid_until
    """
    
    def __init__(self, memory: Memory, config: FCMConfig, user_id: str,
                 crystal_layer: Optional[Any] = None, verbose: bool = True):
        """
        Khởi tạo Solid Layer
        
        Args:
            memory: mem0 Memory instance
            config: FCMConfig
            user_id: User ID
            crystal_layer: Reference to CrystalLayer (for getting crystals)
            verbose: Enable logging
        """
        self.memory = memory
        self.config = config
        self.user_id = user_id
        self.crystal_layer = crystal_layer
        self.verbose = verbose
        
        self.solid_count = 0
        self.last_evolve_at = 0
    
    def _log(self, message: str, level: str = "info"):
        """Helper logging"""
        if self.verbose:
            if level == "info":
                logger.info(f"[Solid] {message}")
                print(f"[Solid] {message}")
            elif level == "debug":
                logger.debug(f"[Solid] {message}")
            elif level == "error":
                logger.error(f"[Solid] {message}")
                print(f"[Solid ERROR] {message}")
            elif level == "warning":
                logger.warning(f"[Solid] {message}")
                print(f"[Solid WARNING] {message}")
    
    def evolve(self, crystal_memories: Optional[List[Dict[str, Any]]] = None,
               force: bool = False,
               total_messages: int = 0) -> Dict[str, Any]:
        """
        Chạy Evolver để hợp nhất Crystal facts thành Solid knowledge.
        
        [MAPLE Archiver] Session-level consolidation với VERSION TRACKING:
        - Summarize: Tóm tắt session thành key facts
        - Detect: Phát hiện conflicts với existing knowledge
        - Archive: Đánh dấu facts cũ là "archived" thay vì xóa
        - Link: Tạo liên kết giữa versions
        
        Args:
            crystal_memories: Crystal memories to evolve (if None, fetch from layer)
            force: Bỏ qua threshold check
            total_messages: Total message count (for threshold check)
            
        Returns:
            Dict với kết quả evolution và version tracking info
        """
        # Check threshold
        messages_since_last = total_messages - self.last_evolve_at
        if not force and messages_since_last < self.config.evolve_threshold:
            self._log(f"Skip evolve: {messages_since_last}/{self.config.evolve_threshold} messages")
            return {"status": "skipped", "reason": "threshold_not_met"}
        
        # Get crystal memories
        if crystal_memories is None:
            if self.crystal_layer:
                crystal_memories = self.crystal_layer.get(limit=20)
            else:
                crystal_memories = self._get_crystals(limit=20)
        
        if not crystal_memories:
            self._log("No crystal memories to evolve")
            return {"status": "skipped", "reason": "no_crystal_memories"}
        
        # Get current solid memories (only active)
        solid_memories = self.get(limit=50, status="active")
        
        # Prepare prompt context
        crystal_text = "\n".join([
            f"- [{m.get('metadata', {}).get('category', 'general')}] {m.get('memory', '')}"
            for m in crystal_memories
        ])
        
        solid_text = "\n".join([
            f"- [ID:{m.get('id')}] {m.get('memory', '')}"
            for m in solid_memories
        ]) if solid_memories else "(Chưa có thông tin)"
        
        current_timestamp = datetime.now().isoformat()
        
        self._log(f"[MAPLE] Evolving {len(crystal_memories)} crystals with {len(solid_memories)} solids...")
        
        try:
            # Call LLM
            response = self.memory.llm.generate_response(
                messages=[
                    {"role": "system", "content": EVOLVER_SYSTEM_PROMPT},
                    {"role": "user", "content": EVOLVER_USER_PROMPT_TEMPLATE.format(
                        solid_facts=solid_text,
                        crystal_facts=crystal_text,
                        current_timestamp=current_timestamp
                    )}
                ]
            )
            
            # Parse JSON
            result = extract_json_from_text(response)
            
            # Normalize to list
            updates_list = []
            if result:
                if isinstance(result, dict):
                    updates_list = result.get("updates") or result.get("reflection") or []
                elif isinstance(result, list):
                    updates_list = result
            
            if not updates_list:
                self._log(f"Warning: No updates parsed. Raw: {str(response)[:100]}...", "warning")
                return {"status": "failed_parse"}
            
            # Process updates
            processed_count = 0
            archived_count = 0
            
            for item in updates_list:
                if not isinstance(item, dict):
                    continue
                
                action = item.get("action", "IGNORE").upper()
                content = item.get("content")
                if not content or action == "IGNORE":
                    continue
                
                # Build metadata
                new_metadata = {
                    "fcm_type": self.config.solid_type,
                    "fcm_frequency": 3,
                    "fcm_status": "active",
                    "category": item.get("category", "general"),
                    "confidence": item.get("confidence", 1.0),
                    "reason": item.get("reason", "Evolution"),
                    "last_updated": current_timestamp,
                    "supersedes": None
                }
                
                # Handle UPDATE: Archive old -> Add new
                if action == "UPDATE":
                    old_id_raw = item.get("supersedes")
                    if old_id_raw:
                        old_id = str(old_id_raw).replace("ID:", "").replace("[", "").replace("]", "").strip()
                        
                        if self._archive(old_id, current_timestamp, "evolved_version"):
                            new_metadata["supersedes"] = old_id
                            archived_count += 1
                        else:
                            self._log(f"Warning: Could not archive {old_id}, treating as ADD", "warning")
                
                # Save new solid memory
                self.memory.add(
                    content,
                    user_id=self.user_id,
                    metadata=new_metadata,
                    infer=False
                )
                
                # Log
                icon = "🔄" if action == "UPDATE" else "➕"
                self._log(f"{icon} {action}: {content[:60]}...")
                processed_count += 1
            
            # Update state
            self.last_evolve_at = total_messages
            self.solid_count += processed_count
            
            return {
                "status": "success",
                "processed": processed_count,
                "archived": archived_count
            }
            
        except Exception as e:
            self._log(f"Evolution error: {e}", "error")
            return {"status": "error", "error": str(e)}
    
    def _archive(self, memory_id: str, valid_until: str, superseded_by: str) -> bool:
        """
        [MAPLE] Archive memory: Get old -> Delete old -> Add new with archived status.
        
        Args:
            memory_id: ID của memory cần archive
            valid_until: Timestamp khi memory hết hiệu lực
            superseded_by: Nội dung của fact mới thay thế
            
        Returns:
            True nếu archive thành công, False nếu lỗi
        """
        try:
            # Get old memory
            old_mem = self.memory.get(memory_id)
            if not old_mem:
                self._log(f"[MAPLE] Memory {memory_id} not found for archiving", "warning")
                return False
            
            content = old_mem.get("memory", "")
            old_meta = old_mem.get("metadata", {})
            
            # Delete old memory
            self.memory.delete(memory_id)
            self._log(f"[MAPLE] Deleted old memory {memory_id}")
            
            # Create archived metadata
            new_meta = old_meta.copy()
            new_meta.update({
                "fcm_status": "archived",
                "original_memory_id": memory_id,
                "valid_until": valid_until,
                "superseded_by": superseded_by,
                "archived_at": datetime.now().isoformat()
            })
            
            # Save with [ARCHIVED] prefix
            self.memory.add(
                f"[ARCHIVED] {content}",
                user_id=self.user_id,
                metadata=new_meta,
                infer=False
            )
            
            self._log(f"[MAPLE] ✓ Archived: '{content[:50]}...'")
            return True
            
        except Exception as e:
            self._log(f"[MAPLE] Archive error: {e}", "error")
            return False
    
    def _get_crystals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get crystal memories directly from memory store"""
        all_memories = self.memory.get_all(
            user_id=self.user_id,
            limit=limit * 2
        )
        
        crystal_memories = []
        for mem in all_memories.get("results", []):
            mem_metadata = mem.get("metadata", {})
            if mem_metadata.get("fcm_type") == self.config.crystal_type:
                crystal_memories.append(mem)
        
        return crystal_memories[:limit]
    
    def get(self, limit: int = 30, section: Optional[str] = None,
            status: str = "active") -> List[Dict[str, Any]]:
        """
        Lấy Solid memories (consolidated knowledge)
        
        [MAPLE] Hỗ trợ filter theo status để phân biệt active vs archived
        
        Args:
            limit: Số lượng tối đa
            section: Filter theo profile section
            status: "active" (mặc định), "archived", hoặc "all"
            
        Returns:
            List các solid memories
        """
        all_memories = self.memory.get_all(
            user_id=self.user_id,
            limit=limit * 3
        )
        
        solid_memories = []
        for mem in all_memories.get("results", []):
            mem_metadata = mem.get("metadata", {})
            
            # Filter by fcm_type
            if mem_metadata.get("fcm_type") != self.config.solid_type:
                continue
            
            # Filter by status
            mem_status = mem_metadata.get("fcm_status", "active")
            if status != "all" and mem_status != status:
                continue
            
            # Filter by section
            if section is not None and mem_metadata.get("profile_section") != section:
                continue
            
            solid_memories.append(mem)
        
        return solid_memories[:limit]
    
    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        """
        [MAPLE/G-Memory] Truy vết lịch sử của một memory
        
        Trả về linked list: current -> previous_version -> ...
        
        Args:
            memory_id: ID của memory cần truy vết
            
        Returns:
            List các versions theo thứ tự từ mới đến cũ
        """
        history = []
        current_id = memory_id
        visited = set()
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            
            all_memories = self.memory.get_all(user_id=self.user_id, limit=100)
            
            found = None
            for mem in all_memories.get("results", []):
                if mem.get("id") == current_id:
                    found = mem
                    break
            
            if not found:
                break
            
            history.append(found)
            current_id = found.get("metadata", {}).get("previous_version")
        
        return history
    
    def get_user_profile(self) -> Dict[str, List[str]]:
        """
        Lấy User Profile từ Solid layer
        
        Returns:
            Dict với user profile organized by sections
        """
        solid_memories = self.get(limit=100)
        
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
            if section in profile:
                profile[section].append(mem.get("memory", ""))
            else:
                profile["other"].append(mem.get("memory", ""))
        
        return profile
    
    def get_count(self) -> int:
        """Get total solid count"""
        return self.solid_count

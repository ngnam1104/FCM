"""
FCM Utilities
=============

Các utility functions hỗ trợ cho FCM system.
"""

import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime


import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str):
    """
    Trích xuất chuỗi JSON từ văn bản LLM trả về, xử lý cả Markdown code blocks.
    """
    if not text or not text.strip():
        return None
        
    try:
        # 1. Xóa Markdown code blocks (```json ... ```)
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # 2. Sửa double curly braces {{ }} thành single { }
        text = text.replace('{{', '{').replace('}}', '}')

        # 2a. Fast path: thử parse trực tiếp toàn bộ chuỗi sau khi làm sạch
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # 3. Tìm đoạn nằm giữa { ... } hoặc [ ... ] đầu tiên và cuối cùng
        match_obj = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if not match_obj:
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            match_obj_str = text[first_brace:last_brace+1] if (first_brace != -1 and last_brace != -1 and last_brace > first_brace) else None
        else:
            match_obj_str = match_obj.group()
            
        match_arr = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
        if not match_arr:
            first_bracket = text.find('[')
            last_bracket = text.rfind(']')
            match_arr_str = text[first_bracket:last_bracket+1] if (first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket) else None
        else:
            match_arr_str = match_arr.group()
        
        json_str = match_obj_str or match_arr_str or text.strip()

        # 4. Thử parse thô trước khi escape newline
        try:
            return json.loads(json_str)
        except Exception:
            pass

        # 5. Escape control characters rồi thử lại
        json_esc = json_str.replace('\r\n', '\\n').replace('\r', '\\n').replace('\n', '\\n')
        json_esc = json_esc.replace('\t', '\\t')
        try:
            return json.loads(json_esc)
        except Exception:
            pass
        
        # 6. Sửa lỗi phổ biến: single quotes, trailing commas
        json_fix = re.sub(r"'(\w+)':", r'"\1":', json_esc)
        json_fix = re.sub(r":\s*'([^']*)'", r': "\1"', json_fix)
        json_fix = re.sub(r',\s*([}\]])', r'\1', json_fix)
        try:
            return json.loads(json_fix)
        except Exception:
            print(f"JSON Decode Error: Cannot parse | Content: {json_str[:100]}...")
            return None

    except Exception as e:
        print(f"Extract JSON Error: {e}")
        return None


def format_memories_for_prompt(memories: List[Dict[str, Any]], 
                               include_metadata: bool = True) -> str:
    """
    Format list of memories thành string cho LLM prompt
    
    Args:
        memories: List các memory objects
        include_metadata: Có include metadata không
        
    Returns:
        Formatted string
    """
    lines = []
    for i, mem in enumerate(memories, 1):
        content = mem.get("memory", mem.get("content", ""))
        
        if include_metadata:
            meta = mem.get("metadata", {})
            fcm_type = meta.get("fcm_type", "unknown")
            category = meta.get("category", "")
            
            if category:
                lines.append(f"{i}. [{fcm_type}/{category}] {content}")
            else:
                lines.append(f"{i}. [{fcm_type}] {content}")
        else:
            lines.append(f"{i}. {content}")
    
    return "\n".join(lines)


def calculate_memory_score(memory: Dict[str, Any], query: str) -> float:
    """
    Tính điểm tổng hợp cho memory dựa trên nhiều factors
    
    Args:
        memory: Memory object
        query: Query string
        
    Returns:
        Score từ 0.0 đến 1.0
    """
    base_score = memory.get("score", 0.5)
    
    # Adjust by layer priority
    fcm_type = memory.get("metadata", {}).get("fcm_type", "liquid")
    layer_weights = {
        "solid": 1.2,
        "crystal": 1.0,
        "liquid": 0.8
    }
    
    layer_weight = layer_weights.get(fcm_type, 1.0)
    
    # Adjust by confidence (if available)
    confidence = memory.get("metadata", {}).get("confidence", 0.8)
    
    # Final score
    final_score = base_score * layer_weight * confidence
    
    return min(1.0, max(0.0, final_score))


def deduplicate_facts(facts: List[str], threshold: float = 0.85) -> List[str]:
    """
    Loại bỏ facts trùng lặp hoặc quá giống nhau
    
    Simple implementation using character overlap.
    For production, use embedding similarity.
    
    Args:
        facts: List of fact strings
        threshold: Similarity threshold
        
    Returns:
        Deduplicated list
    """
    if not facts:
        return []
    
    unique_facts = [facts[0]]
    
    for fact in facts[1:]:
        is_duplicate = False
        fact_lower = fact.lower()
        
        for existing in unique_facts:
            existing_lower = existing.lower()
            
            # Simple overlap check
            shorter = min(len(fact_lower), len(existing_lower))
            if shorter == 0:
                continue
                
            # Check if one contains the other
            if fact_lower in existing_lower or existing_lower in fact_lower:
                is_duplicate = True
                break
            
            # Character overlap ratio
            overlap = len(set(fact_lower.split()) & set(existing_lower.split()))
            total = len(set(fact_lower.split()) | set(existing_lower.split()))
            
            if total > 0 and overlap / total > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_facts.append(fact)
    
    return unique_facts


def merge_user_profiles(old_profile: Dict[str, List[str]], 
                       new_profile: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge hai user profiles, ưu tiên thông tin mới
    
    Args:
        old_profile: Profile cũ
        new_profile: Profile mới
        
    Returns:
        Merged profile
    """
    merged = {}
    
    all_sections = set(old_profile.keys()) | set(new_profile.keys())
    
    for section in all_sections:
        old_facts = old_profile.get(section, [])
        new_facts = new_profile.get(section, [])
        
        # New facts take priority, then old facts
        combined = new_facts + old_facts
        
        # Deduplicate
        merged[section] = deduplicate_facts(combined)
    
    return merged


def timestamp_now() -> str:
    """Get current timestamp in ISO format"""
    return datetime.now().isoformat()


def calculate_keyword_boost(query: str, memory_content: str) -> float:
    """
    Tính keyword boost score cho memory dựa trên query.
    
    Ý tưởng: Embedding similarity tốt cho semantic matching nhưng yếu với
    keyword matching (số, năm, tên riêng). Function này boost score khi
    memory chứa keywords quan trọng từ query.
    
    Args:
        query: Câu hỏi/truy vấn
        memory_content: Nội dung memory
        
    Returns:
        Boost score (0.0 đến 0.4)
    """
    boost = 0.0
    query_lower = query.lower()
    content_lower = memory_content.lower()
    
    # 1. Detect số năm sinh (4 chữ số từ 1900-2030)
    year_keywords = ["sinh năm", "năm sinh", "sinh vào năm", "tuổi", "born in", "year of birth"]
    if any(kw in query_lower for kw in year_keywords):
        # Tìm năm trong memory
        years_in_memory = re.findall(r'\b(19\d{2}|20[0-3]\d)\b', content_lower)
        if years_in_memory:
            boost += 0.20  # Tăng từ 0.15 lên 0.20
            
    # 2. Detect số (số điện thoại, tuổi, số lượng...)
    numbers_in_query = re.findall(r'\b\d+\b', query)
    if numbers_in_query:
        for num in numbers_in_query:
            if num in content_lower:
                boost += 0.1
                break
                
    # 3. Keyword matching cho các từ quan trọng
    # Loại bỏ stopwords và lấy keywords quan trọng
    vietnamese_stopwords = {
        'là', 'của', 'và', 'có', 'được', 'cho', 'với', 'trong', 'này', 'đó',
        'những', 'các', 'một', 'để', 'khi', 'từ', 'theo', 'như', 'về', 'bạn',
        'tôi', 'gì', 'nào', 'sao', 'bao', 'nhiêu', 'thế', 'nào', 'không', 'có',
        'người', 'dùng', 'user', 'the', 'a', 'an', 'is', 'are', 'what', 'how'
    }
    
    # Trích xuất keywords từ query
    query_words = set(re.findall(r'[a-zA-ZÀ-ỹ]+', query_lower))
    query_keywords = query_words - vietnamese_stopwords
    
    # Đếm keyword matches
    keyword_matches = 0
    for kw in query_keywords:
        if len(kw) >= 3 and kw in content_lower:  # Chỉ count keywords >= 3 chars
            keyword_matches += 1
    
    if query_keywords:
        match_ratio = keyword_matches / len(query_keywords)
        boost += match_ratio * 0.1  # Max 0.1 for keyword matching
        
    # 4. Detect patterns đặc biệt (tăng cường)
    special_patterns = [
        # (query_patterns, memory_patterns, boost_amount)
        (["thích", "yêu thích", "sở thích", "hobby"], ["thích", "yêu thích", "passion", "hobby"], 0.08),
        (["làm việc", "nghề", "job", "work", "công việc"], ["làm", "job", "work", "developer", "engineer"], 0.08),
        (["tên", "name", "gọi là"], ["tên", "name"], 0.08),
        (["ở đâu", "sống", "live", "địa chỉ"], ["sống", "ở", "live", "địa chỉ"], 0.08),
        (["học ngành", "ngành học", "chuyên ngành", "major", "ngành gì"], ["ngành", "chuyên ngành", "major", "khmt", "cntt", "công nghệ thông tin"], 0.15),
        (["học trường", "trường", "school", "university", "học ở đâu"], ["học", "trường", "university", "school", "bách khoa", "đại học"], 0.10),
        (["ngôn ngữ", "language", "lập trình", "programming", "code"], ["python", "java", "javascript", "c++", "lập trình", "code"], 0.10),
    ]
    
    for query_patterns, memory_patterns, pattern_boost in special_patterns:
        if any(p in query_lower for p in query_patterns):
            if any(p in content_lower for p in memory_patterns):
                boost += pattern_boost
                break
    
    return min(boost, 0.4)  # Cap at 0.4


def clean_noise_from_message(message: str) -> Optional[str]:
    """
    Loại bỏ các patterns noise phổ biến
    
    Args:
        message: Message string
        
    Returns:
        Cleaned message hoặc None nếu là pure noise
    """
    noise_patterns = [
        r'^(xin\s+)?chào(\s+bạn)?[!.]?$',
        r'^cảm\s+ơn(\s+bạn)?[!.]?$',
        r'^ok[!.]?$',
        r'^ừ[m]?[!.]?$',
        r'^à[!.]?$',
        r'^tạm\s+biệt[!.]?$',
        r'^bye[!.]?$',
        r'^hi[!.]?$',
        r'^hello[!.]?$',
    ]
    
    message_lower = message.strip().lower()
    
    for pattern in noise_patterns:
        if re.match(pattern, message_lower):
            return None
    
    return message


class SessionTracker:
    """Track conversation sessions for better memory management"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def start_session(self, user_id: str) -> str:
        """Start a new session"""
        session_id = f"{user_id}_{timestamp_now()}"
        self.sessions[session_id] = {
            "user_id": user_id,
            "started_at": timestamp_now(),
            "message_count": 0,
            "topics": []
        }
        return session_id
    
    def add_message(self, session_id: str, message: str) -> None:
        """Add a message to session"""
        if session_id in self.sessions:
            self.sessions[session_id]["message_count"] += 1
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a session and return summary"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        session["ended_at"] = timestamp_now()
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info"""
        return self.sessions.get(session_id)

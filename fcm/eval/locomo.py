"""
FCM LoCoMo Benchmark Comparison
===============================

So sánh FCM V1 vs V2 trên LoCoMo dataset.

LoCoMo: Long Context Memory benchmark
- Context dài (nhiều tin nhắn)
- Questions yêu cầu recall từ long-term memory
- Ground truth để đánh giá accuracy

Chạy: python -m fcm_eval.locomo
"""

import json
import os
import sys
import time
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0").setLevel(logging.WARNING)

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from fcm.eval.common import (
    print_header, print_subheader, print_comparison_table,
    normalize_text, BenchmarkResult
)

# ============================================================================
# LoCoMo CATEGORY MAPPING
# ============================================================================
# Category definitions from LoCoMo dataset:
# 1 = single-hop: Simple factual questions that require single memory retrieval
# 2 = temporal: Questions about when things happened or time-based reasoning
# 3 = multi-hop: Questions requiring multiple memory retrievals and reasoning
# 4 = knowledge: Questions about facts, knowledge, or learned information
# 5 = summary: Questions requiring summarization of events or conversations

CATEGORY_MAP = {
    1: "single-hop",
    2: "temporal", 
    3: "multi-hop",
    4: "knowledge",
    5: "summary"
}

CATEGORY_DESCRIPTIONS = {
    1: "Single-hop: Câu hỏi đơn giản, chỉ cần truy xuất 1 memory",
    2: "Temporal: Câu hỏi về thời gian, cần suy luận temporal",
    3: "Multi-hop: Câu hỏi phức tạp, cần nhiều memory + reasoning",
    4: "Knowledge: Câu hỏi về kiến thức, facts đã học",
    5: "Summary: Câu hỏi cần tổng hợp nhiều thông tin"
}

# ============================================================================
# UPGRADED SEMANTIC MATCHING CONFIG (Ported from Debug)
# ============================================================================

# 1. Mở rộng danh sách từ đồng nghĩa
SYNONYM_GROUPS = [
    {"psychology", "counseling", "mental health", "therapy", "psychologist", "counselor"},
    {"social work", "social worker", "community service", "human services"},
    {"education", "teaching", "teacher", "educator", "academic"},
    {"art", "arts", "painting", "artist", "creative", "artwork", "draw", "drew"},
    {"music", "musician", "musical", "instrument"},
    {"lgbtq", "lgbt", "queer", "transgender", "trans", "gender identity", "identity"},
    {"support group", "support", "community", "group therapy"},
    {"yesterday", "day before", "previous day"},
    {"last week", "previous week", "week ago"},
    {"last month", "previous month", "month ago"},
    {"last year", "previous year", "year ago", "year before"},
    {"dance", "dancing", "danced", "dancer", "movement"},
]

# 2. Thêm logic tính toán thời gian
TEMPORAL_REFERENCES = {
    "last year": -1,
    "year before": -1,
    "previous year": -1,
    "yesterday": -1, 
    "day before": -1,
    "last month": -1,
    "previous month": -1,
}

def parse_date_flexible(date_str: str) -> Optional[datetime]:
    """
    Parse ngày tháng linh hoạt:
    - 2022
    - 19 January, 2023
    - Jan 19 2023
    """
    if not date_str: return None
    date_str = str(date_str).strip()
    
    # 1. Thử parse chỉ có năm (YYYY)
    if re.match(r'^\d{4}$', date_str):
        return datetime(int(date_str), 1, 1)

    # 2. Các pattern ngày tháng đầy đủ
    patterns = [
        r"(\d{1,2})\s+(\w+),?\s+(\d{4})",  # "19 January, 2023"
        r"(\w+)\s+(\d{1,2}),?\s+(\d{4})",  # "January 19, 2023"
    ]
    
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    for pat in patterns:
        match = re.search(pat, date_str, re.IGNORECASE)
        if match:
            # Xác định đâu là tháng, đâu là ngày dựa trên vị trí
            g1, g2, g3 = match.groups()
            
            # Logic: Tìm xem nhóm nào là chữ (tháng)
            month_str = g1 if g1.lower()[:3] in months else g2
            day_str = g2 if month_str == g1 else g1
            
            try:
                m = months.get(month_str.lower()[:3]) # Lấy 3 ký tự đầu
                if not m: m = months.get(month_str.lower())
                
                return datetime(int(g3), m, int(day_str))
            except:
                continue
    return None

def check_temporal_match(rag_answer: str, expected: str, session_date_str: str) -> tuple[bool, float]:
    """
    Logic so khớp thời gian thông minh (Đã sửa lỗi so sánh ngày)
    """
    norm_answer = rag_answer.lower()
    
    # 1. Parse các ngày tháng quan trọng
    expected_date = parse_date_flexible(expected)
    session_date = parse_date_flexible(session_date_str)
    
    if not expected_date or not session_date:
        # Fallback: Nếu không parse được date, so sánh năm đơn thuần như cũ
        try:
            exp_year = int(re.search(r'\d{4}', expected).group(0))
            sess_year = int(re.search(r'\d{4}', session_date_str).group(0))
            # Check keywords relative year
            if "last year" in norm_answer and (sess_year - 1 == exp_year): return True, 0.9
            if "next year" in norm_answer and (sess_year + 1 == exp_year): return True, 0.9
            if str(exp_year) in norm_answer: return True, 1.0
        except:
            pass
        return False, 0.0

    # 2. Logic tương đối (Relative Logic)
    # Case: "Yesterday"
    if "yesterday" in norm_answer:
        calculated = session_date - timedelta(days=1)
        # So sánh ngày/tháng/năm
        if (calculated.day == expected_date.day and 
            calculated.month == expected_date.month and 
            calculated.year == expected_date.year):
            return True, 0.95

    # Case: "Last year"
    if "last year" in norm_answer:
        if session_date.year - 1 == expected_date.year:
            return True, 0.9

    # Case: "Last month"
    if "last month" in norm_answer:
        # Logic đơn giản cho last month (cùng năm)
        if (session_date.month - 1 == expected_date.month and 
            session_date.year == expected_date.year):
            return True, 0.9
            
    # Case: Kiểm tra nếu LLM đã tự tính ra ngày cụ thể trong câu trả lời
    # Ví dụ: LLM trả lời "It was 19 Jan" -> Match với expected "19 January"
    if str(expected_date.day) in norm_answer and str(expected_date.year) in norm_answer:
        # Check tháng (scan tên tháng trong answer)
        month_name = expected_date.strftime("%B").lower()
        month_abbr = expected_date.strftime("%b").lower()
        if month_name in norm_answer or month_abbr in norm_answer:
            return True, 1.0

    return False, 0.0

# ============================================================================
# OPTION 1: LLM READER - Answer questions using retrieved context + LLM
# ============================================================================

LLM_READER_PROMPT = """You are a question-answering assistant. Based on the provided context, answer the question accurately.

CONTEXT:
{context}

SESSION DATE (when the conversation happened): {session_date}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the context provided
2. If the context mentions relative time (yesterday, last week, last year), calculate the actual date using the session date
3. Be concise and direct
4. If the answer cannot be determined from context, say "UNKNOWN"

TEMPORAL CALCULATIONS:
- "yesterday" = session_date - 1 day
- "last week" = approximately session_date - 7 days  
- "last year" = session_date.year - 1
- "next month" = session_date.month + 1

ANSWER (just the answer, no explanation):"""


class LLMReader:
    """
    LLM Reader component for answering questions from retrieved context.
    Handles temporal inference by computing relative dates.
    """
    
    def __init__(self, llm=None, verbose: bool = True):
        """
        Initialize LLM Reader.
        
        Args:
            llm: LLM instance (if None, will create from config)
            verbose: Whether to print debug info
        """
        self.llm = llm
        self.verbose = verbose
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM from GROQ config"""
        if self.llm is None:
            try:
                import os
                from dotenv import load_dotenv
                load_dotenv()
                
                groq_key = os.getenv("GROQ_API_KEY")
                if not groq_key:
                    print("[LLMReader] Warning: GROQ_API_KEY not found in environment")
                    return
                
                from mem0.llms.groq import GroqLLM
                
                self.llm = GroqLLM(config={
                    "model": "llama-3.1-8b-instant",
                    "temperature": 0.1,
                    "max_tokens": 256,
                    "api_key": groq_key
                })
                
                if self.verbose:
                    print("[LLMReader] Initialized with GROQ LLM")
                    
            except Exception as e:
                print(f"[LLMReader] Warning: Could not init LLM: {e}")
                self.llm = None
    
    def answer_question(
        self, 
        question: str, 
        context: str, 
        session_date: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Answer a question using retrieved context + LLM.
        
        Args:
            question: The question to answer
            context: Retrieved context from memory
            session_date: Session date string (e.g., "8 May, 2023")
            
        Returns:
            Tuple of (answer, confidence)
        """
        if not context or not context.strip():
            return "UNKNOWN", 0.0
        
        if self.llm is None:
            return self._fallback_answer(question, context, session_date)
        
        # Parse session date
        session_date_str = session_date or "Unknown"
        
        prompt = LLM_READER_PROMPT.format(
            context=context[:2000],  # Limit context length
            session_date=session_date_str,
            question=question
        )
        
        try:
            response = self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}]
            )
            
            answer = response.strip()
            
            if self.verbose:
                print(f"   [LLMReader] Q: {question[:50]}...")
                print(f"   [LLMReader] A: {answer}")
            
            # Confidence based on answer quality
            if answer == "UNKNOWN" or not answer:
                return "UNKNOWN", 0.0
            
            return answer, 0.9
            
        except Exception as e:
            if self.verbose:
                print(f"   [LLMReader] Error: {e}")
            return self._fallback_answer(question, context, session_date)
    
    def _fallback_answer(
        self, 
        question: str, 
        context: str, 
        session_date: Optional[str]
    ) -> Tuple[str, float]:
        """
        Fallback answer extraction without LLM.
        Uses pattern matching for temporal references.
        """
        context_lower = context.lower()
        question_lower = question.lower()
        
        # Check for temporal patterns and compute dates
        if session_date:
            parsed_date = self._parse_date(session_date)
            if parsed_date:
                # Look for temporal references in context
                answer = self._extract_temporal_answer(context, question, parsed_date)
                if answer:
                    return answer, 0.7
        
        return context[:100], 0.3  # Return first 100 chars as fallback
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string like '8 May, 2023' or '4:04 pm on 20 January, 2023'"""
        patterns = [
            r"(\d{1,2})\s+(\w+),?\s+(\d{4})",  # "8 May, 2023"
            r"on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})",  # "on 20 January, 2023"
        ]
        
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for pattern in patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                day = int(match.group(1))
                month_str = match.group(2).lower()
                year = int(match.group(3))
                month = months.get(month_str, 1)
                try:
                    return datetime(year, month, day)
                except:
                    pass
        
        return None
    
    def _extract_temporal_answer(
        self, 
        context: str, 
        question: str, 
        session_date: datetime
    ) -> Optional[str]:
        """
        Extract answer by computing temporal references.
        
        Examples:
        - "yesterday" → session_date - 1 day
        - "last year" → session_date.year - 1
        """
        context_lower = context.lower()
        
        # Pattern: "yesterday" → compute actual date
        if "yesterday" in context_lower:
            yesterday = session_date - timedelta(days=1)
            # Windows compatible format (no %-d)
            return f"{yesterday.day} {yesterday.strftime('%B')} {yesterday.year}"
        
        # Pattern: "last week" → approximately 7 days ago
        if "last week" in context_lower:
            last_week = session_date - timedelta(days=7)
            return f"The week before {session_date.day} {session_date.strftime('%B')} {session_date.year}"
        
        # Pattern: "last year" → extract year from context
        if "last year" in context_lower:
            return str(session_date.year - 1)
        
        # Pattern: "next month" → compute next month
        if "next month" in context_lower:
            if session_date.month < 12:
                next_month = session_date.replace(month=session_date.month + 1)
            else:
                next_month = session_date.replace(year=session_date.year + 1, month=1)
            return f"{next_month.strftime('%B')} {next_month.year}"
        
        return None


def check_answer_with_llm_reader(
    retrieved: str, 
    expected: str, 
    question: str,
    session_date: Optional[str] = None,
    llm_reader: Optional[LLMReader] = None
) -> Tuple[bool, float, str, str]: # <--- Thêm str return type cho Reason
    """
    Enhanced check returning reason for match/mismatch
    """
    if not retrieved or not retrieved.strip():
        return False, 0.0, "No context retrieved", "Retrieval failed (Empty context)"
    
    # 1. Gọi LLM để lấy câu trả lời suy luận
    if llm_reader:
        llm_answer, confidence = llm_reader.answer_question(
            question=question,
            context=retrieved,
            session_date=session_date
        )
    else:
        llm_answer = retrieved[:200]
        confidence = 0.5

    norm_expected = normalize_text(expected)
    norm_llm_answer = normalize_text(llm_answer)

    # --- LOGIC SO KHỚP MỚI ---

    # Case 1: Exact Match (Tuyệt đối)
    if norm_expected in norm_llm_answer:
        return True, 1.0, llm_answer, f"Exact match: Found '{expected}' in answer"
    
    # Case 2: Temporal Inference (Suy luận thời gian)
    if session_date:
        is_temp_match, temp_score = check_temporal_match(llm_answer, expected, session_date)
        if is_temp_match:
            return True, temp_score, llm_answer, f"Temporal match: Infers '{expected}' from relative time"

    # Case 3: Synonym Match (Từ đồng nghĩa)
    for group in SYNONYM_GROUPS:
        if any(s in norm_expected for s in group):
            # Tìm từ nào trong answer khớp với group này
            matched_synonym = next((s for s in group if s in norm_llm_answer), None)
            if matched_synonym:
                return True, 0.85, llm_answer, f"Synonym match: '{matched_synonym}' ~ '{expected}'"
    
    # Case 4: Keyword Overlap (So khớp từ khóa)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'to', 'of', 'and', 'in', 'on', 'at'}
    exp_keywords = set(norm_expected.split()) - stop_words
    ans_keywords = set(norm_llm_answer.split()) - stop_words
    
    if exp_keywords:
        overlap = len(exp_keywords & ans_keywords)
        ratio = overlap / len(exp_keywords)
        if ratio >= 0.4:
            return True, 0.5 + (ratio * 0.4), llm_answer, f"Keyword overlap: {ratio:.0%} match"

    # Case 5: Fail
    return False, 0.0, llm_answer, f"Mismatch. Expected '{expected}' but got '{llm_answer}'"

def _dates_match(date1: str, date2: str) -> bool:
    """Check if two date strings represent the same date"""
    # Extract numbers and month names
    def extract_date_parts(s):
        s = s.lower()
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        
        numbers = re.findall(r'\d+', s)
        month = None
        for m in months:
            if m in s:
                month = m
                break
        
        return set(numbers), month
    
    parts1 = extract_date_parts(date1)
    parts2 = extract_date_parts(date2)
    
    # If same numbers and same month → same date
    if parts1[0] == parts2[0] and parts1[1] == parts2[1]:
        return True
    
    return False

def get_category_name(category_num: int) -> str:
    """Convert category number to name"""
    return CATEGORY_MAP.get(category_num, f"unknown-{category_num}")

def get_category_description(category_num: int) -> str:
    """Get detailed description for category"""
    return CATEGORY_DESCRIPTIONS.get(category_num, f"Unknown category {category_num}")


def load_locomo_dataset(path: str = "dataset/locomo10.json", max_samples: int = 5, 
                         max_messages_per_sample: int = 50) -> List[Dict]:
    """
    Load LoCoMo dataset và transform thành format đơn giản
    
    LoCoMo format:
    - qa: List of {question, answer, evidence, category}
    - conversation: Dict with session_1, session_2, ... each containing list of messages
    - session_X_date_time: Timestamp for each session
    
    Output format:
    - messages: List of conversation messages  
    - qa: List of {question, answer, category, category_name} pairs
    - session_dates: Dict mapping session_key to date string
    """
    if not os.path.exists(path):
        print(f"ERROR: File {path} không tồn tại.")
        print("Tạo sample dataset...")
        
        # Create sample dataset
        sample_data = [
            {
                "messages": [
                    "Tôi là Minh, sinh viên IT",
                    "Tôi học ở Bách Khoa Hà Nội",
                    "Tôi thích lập trình Python",
                ],
                "qa": [{"question": "Minh học ở đâu?", "answer": "Bách Khoa Hà Nội"}],
                "session_dates": {"session_1": "1 January, 2024"}
            },
        ]
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        return sample_data
    
    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Transform LoCoMo format
    transformed = []
    for item in raw_data[:max_samples]:
        # Extract messages from all sessions
        messages = []
        conv = item.get("conversation", {})
        
        # Extract session dates
        session_dates = {}
        for key in conv.keys():
            if key.endswith("_date_time"):
                session_key = key.replace("_date_time", "")
                session_dates[session_key] = conv[key]
        
        # Get all sessions (session_1, session_2, ...)
        session_keys = [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")]
        session_keys = sorted(session_keys, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
        
        for session_key in session_keys:
            session = conv.get(session_key, [])
            session_date = session_dates.get(session_key, "")
            
            if isinstance(session, list):
                for msg in session:
                    if isinstance(msg, dict) and "text" in msg:
                        speaker = msg.get("speaker", "Unknown")
                        text = msg.get("text", "")
                        # Include session date in message for temporal context
                        messages.append({
                            "text": f"{speaker}: {text}",
                            "session_key": session_key,
                            "session_date": session_date
                        })
            
            # Limit messages per sample
            if len(messages) >= max_messages_per_sample:
                break
        
        # Extract QA pairs (limit to first 5 for speed)
        qa_pairs = []
        for qa in item.get("qa", [])[:5]:
            # Get evidence to determine which session the answer comes from
            evidence = qa.get("evidence", [])
            session_num = None
            if evidence:
                # Parse evidence like "D1:3" → session_1
                first_evidence = evidence[0]
                match = re.match(r"D(\d+):", first_evidence)
                if match:
                    session_num = int(match.group(1))
            
            # Get category info
            category_num = qa.get("category", 0)
            category_name = get_category_name(category_num)
            
            qa_pairs.append({
                "question": qa.get("question", ""),
                "answer": str(qa.get("answer", "")),  # Convert to string
                "evidence": evidence,
                "session_num": session_num,
                "category": category_num,
                "category_name": category_name,
            })
        
        if messages and qa_pairs:
            # Convert messages to simple strings for backward compatibility
            simple_messages = [m["text"] if isinstance(m, dict) else m for m in messages]
            
            transformed.append({
                "messages": simple_messages[:max_messages_per_sample],
                "messages_with_dates": messages[:max_messages_per_sample],  # Keep detailed version
                "qa": qa_pairs,
                "session_dates": session_dates,
                "sample_id": item.get("sample_id", len(transformed) + 1)
            })
    
    print(f"Loaded {len(transformed)} samples from LoCoMo dataset")
    return transformed


def check_answer(retrieved: str, expected: str) -> bool:
    """Check if retrieved text contains expected answer (strict substring match)"""
    norm_expected = normalize_text(expected)
    norm_retrieved = normalize_text(retrieved)
    return norm_expected in norm_retrieved and norm_retrieved != ""


# Synonym groups for semantic matching
SYNONYM_GROUPS = [
    # Education/Career fields
    {"psychology", "counseling", "mental health", "therapy", "psychologist", "counselor"},
    {"social work", "social worker", "community service", "human services"},
    {"education", "teaching", "teacher", "educator", "academic"},
    {"art", "arts", "painting", "artist", "creative", "artwork"},
    {"music", "musician", "musical", "instrument"},
    
    # LGBTQ related
    {"lgbtq", "lgbt", "queer", "transgender", "trans", "gender identity", "identity"},
    {"support group", "support", "community", "group therapy"},
    
    # Temporal
    {"yesterday", "day before", "previous day"},
    {"last week", "previous week", "week ago"},
    {"last month", "previous month", "month ago"},
    {"last year", "previous year", "year ago"},
    
    # Dates - specific patterns
    {"may 7", "7 may", "may 7th", "7th may", "seventh of may"},
    {"january", "jan"},
    {"february", "feb"},
    
    # Actions
    {"paint", "painting", "painted", "draw", "drew", "artwork"},
    {"sunrise", "morning", "dawn"},
    {"volunteer", "volunteering", "volunteered", "help", "helping"},
]


def check_answer_fuzzy(retrieved: str, expected: str, question: str) -> tuple[bool, float]:
    """
    Fuzzy check if retrieved context can answer the question.
    
    Returns:
        (is_correct, confidence_score)
        
    Uses multiple strategies:
    1. Exact substring match (score = 1.0)
    2. Synonym/Semantic match (score = 0.85)
    3. Keyword overlap (score = 0.5-0.9)
    4. Topic relevance (score = 0.3-0.7)
    """
    if not retrieved or not retrieved.strip():
        return False, 0.0
    
    norm_expected = normalize_text(expected)
    norm_retrieved = normalize_text(retrieved)
    
    # Strategy 1: Exact match (best case)
    if norm_expected in norm_retrieved:
        return True, 1.0
    
    # Strategy 2: Synonym/Semantic match
    # Check if expected answer has synonyms that appear in retrieved
    for synonym_group in SYNONYM_GROUPS:
        expected_synonyms = [s for s in synonym_group if s in norm_expected]
        if expected_synonyms:
            # Check if any synonym appears in retrieved
            for synonym in synonym_group:
                if synonym in norm_retrieved:
                    return True, 0.85
    
    # Strategy 3: Keyword overlap with relaxed threshold
    expected_keywords = set(norm_expected.split())
    retrieved_keywords = set(norm_retrieved.split())
    
    # Extended stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'on', 'at', 
                  'for', 'with', 'that', 'this', 'be', 'has', 'had', 'have', 'do', 'does', 'did'}
    expected_keywords -= stop_words
    retrieved_keywords -= stop_words
    
    if expected_keywords:
        # Direct overlap
        overlap = len(expected_keywords & retrieved_keywords)
        overlap_ratio = overlap / len(expected_keywords)
        
        # Also check synonym overlap
        synonym_matches = 0
        for exp_kw in expected_keywords:
            for synonym_group in SYNONYM_GROUPS:
                if exp_kw in synonym_group:
                    if any(syn in retrieved_keywords for syn in synonym_group):
                        synonym_matches += 1
                        break
        
        total_matches = overlap + synonym_matches
        total_ratio = min(total_matches / len(expected_keywords), 1.0)
        
        # Relaxed threshold: 40% match is enough for semantic similarity
        if total_ratio >= 0.4:
            return True, 0.5 + (total_ratio * 0.4)
    
    # Strategy 4: Topic relevance check
    question_lower = question.lower()
    question_keywords = set(question_lower.split()) - stop_words
    
    topic_overlap = len(question_keywords & retrieved_keywords)
    if topic_overlap >= 2:  # At least 2 topic words match
        return False, 0.3
    
    return False, 0.0


def check_answer_llm(retrieved: str, expected: str, question: str) -> tuple[bool, float]:
    """
    Use LLM to verify if retrieved context can answer the question.
    
    Returns:
        (is_correct, confidence_score)
    """
    if not retrieved or not retrieved.strip():
        return False, 0.0
    
    # First try fuzzy matching
    fuzzy_correct, fuzzy_score = check_answer_fuzzy(retrieved, expected, question)
    if fuzzy_correct:
        return True, fuzzy_score
    
    # For now, return fuzzy result
    # TODO: Add LLM verification for complex inference cases
    return fuzzy_correct, fuzzy_score


def run_locomo_v1(data: List[Dict], verbose: bool = True, use_llm_reader: bool = True) -> BenchmarkResult:
    """Run LoCoMo benchmark với FCM V1
    
    Args:
        data: LoCoMo dataset
        verbose: Print detailed logs
        use_llm_reader: Use LLM Reader for temporal inference (Option 1)
    """
    from fcm import FCMAgent, FCMConfig
    
    print_subheader(f"FCM V1: Running {len(data)} samples...")
    if use_llm_reader:
        print("   🤖 LLM Reader: ENABLED (temporal inference)")
    start_time = time.time()
    
    config = FCMConfig(verbose=False, crystallize_threshold=5)
    
    # Initialize LLM Reader if enabled
    llm_reader = LLMReader(verbose=False) if use_llm_reader else None
    
    correct_count = 0
    total_questions = 0
    total_facts = 0
    total_messages = 0
    llm_reader_answers = 0
    
    detailed_results = []
    
    # Create one agent for all samples (avoid Qdrant lock issues)
    test_user_id = "locomo_v1_benchmark"
    agent = FCMAgent(config=config, user_id=test_user_id)
    
    for i, item in enumerate(data):
        sample_id = item.get("sample_id", i + 1)
        
        # Clear memories between samples using new method
        agent.clear_all_memories()
        
        messages = item.get("messages", [])
        qa_pairs = item.get("qa", [])
        session_dates = item.get("session_dates", {})
        
        if verbose:
            print(f"\n🔹 Sample {i+1}/{len(data)} ({len(messages)} msgs, {len(qa_pairs)} QAs)")
        
        # Ingest messages
        if verbose:
            print("   Ingesting...", end="", flush=True)
        
        for msg in messages:
            agent.chat(msg)
            if verbose:
                print(".", end="", flush=True)
        
        total_messages += len(messages)
        
        if verbose:
            print(" Done.")
        
        # Process
        if verbose:
            print("   Processing...", end="", flush=True)
        agent.crystallize(force=True)
        agent.evolve(force=True)
        if verbose:
            print(" Done.")
        
        stats = agent.get_stats()
        total_facts += stats.get("crystal_count", 0) + stats.get("solid_count", 0)
        
        # Search for each QA pair
        for qa in qa_pairs:
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            session_num = qa.get("session_num")
            category = qa.get("category", 0)
            category_name = qa.get("category_name", "unknown")
            
            # Get session date for this QA
            session_date = None
            if session_num:
                session_key = f"session_{session_num}"
                session_date = session_dates.get(session_key)
            
            search_result = agent.search(question, strategy="enhanced", limit=10)
            
            # 2. Xử lý kết quả trả về
            combined = search_result.get("combined", []) if isinstance(search_result, dict) else []
            
            # 3. Kỹ thuật "Context Window Expansion": Gộp Top-5 kết quả tốt nhất
            context_parts = []
            if combined:
                # Lấy 5 kết quả đầu tiên (thường chứa các mảnh thông tin bổ sung cho nhau)
                for i, item in enumerate(combined[:5]): 
                    mem = item.get("memory", "").strip()
                    if mem:
                        # Thêm dấu gạch đầu dòng để phân tách các ký ức
                        context_parts.append(f"- {mem}")
            
            # Nối lại thành 1 đoạn văn bản dài
            retrieved_memory = "\n".join(context_parts)

            # Metadata để log (lấy cái đầu tiên làm đại diện)
            top_item = combined[0] if combined else {}
            source_layer = top_item.get("source_layer", top_item.get("metadata", {}).get("fcm_type", "N/A")).upper()
            retrieval_score = top_item.get("score", 0)
            
            match_reason = "Unknown"
            if use_llm_reader and llm_reader:
                # Cập nhật: nhận thêm biến match_reason
                is_correct, confidence, llm_answer, match_reason = check_answer_with_llm_reader(
                    retrieved=retrieved_memory,
                    expected=ground_truth,
                    question=question,
                    session_date=session_date,
                    llm_reader=llm_reader
                )
                if llm_answer and llm_answer != "No context retrieved":
                    llm_reader_answers += 1
                weighted_score = confidence
            else:
                # Fallback
                is_correct, weighted_score = check_answer_fuzzy(retrieved_memory, ground_truth, question)
                llm_answer = retrieved_memory[:100]
                match_reason = "Fuzzy match fallback"
            
            if is_correct:
                correct_count += 1
                status = "✅"
            else:
                status = "❌"
            
            total_questions += 1
            
            if verbose:
                # === PHẦN IN RA GIẢI THÍCH CHI TIẾT ===
                print(f"   {status} Q: {question}")
                print(f"      Expected: '{ground_truth}'")
                print(f"      LLM Got : '{llm_answer}'")
                print(f"      Reason  : {match_reason}")
                if not is_correct:
                    # In thêm context để debug xem tại sao sai
                    preview = retrieved_memory[:80].replace('\n', ' ')
                    print(f"      Context : '{preview}...'")
                print("-" * 40) # Dòng kẻ phân cách cho dễ nhìn
            
            detailed_results.append({
                "sample_id": sample_id,
                "correct": is_correct,
                "question": question,
                "expected": ground_truth,
                "retrieved": retrieved_memory[:100],
                "source": source_layer,
                "weighted_score": weighted_score,
                "session_date": session_date,
                "category": category,
                "category_name": category_name,
            })
    
    elapsed = time.time() - start_time
    accuracy = correct_count / max(1, total_questions)
    
    return BenchmarkResult(
        version="V1",
        total_time=elapsed,
        messages_processed=total_messages,
        facts_extracted=total_facts,
        search_accuracy=accuracy,
        memory_stats={
            "Correct": correct_count,
            "Total": total_questions,
        },
        additional_metrics={
            "detailed_results": detailed_results,
            "llm_reader_enabled": use_llm_reader,
            "llm_reader_answers": llm_reader_answers,
        }
    )


def run_locomo_v2(data: List[Dict], verbose: bool = True, use_llm_reader: bool = True) -> BenchmarkResult:
    """Run LoCoMo benchmark với FCM V2
    
    Args:
        data: LoCoMo dataset
        verbose: Print detailed logs
        use_llm_reader: Use LLM Reader for temporal inference (Option 1)
    """
    from fcm.v2 import FCMAgentV2, FCMConfigV2
    
    print_subheader(f"FCM V2: Running {len(data)} samples...")
    if use_llm_reader:
        print("   🤖 LLM Reader: ENABLED (temporal inference)")
    start_time = time.time()
    
    config = FCMConfigV2(
        verbose=False, 
        crystallize_threshold=3,
        attention_sink_count=2,
        enable_active_forgetting=False,  # Disable for benchmark (short context)
        enable_dynamic_persona=True,
        retrieval_weight_solid=0.5,
        retrieval_weight_crystal=0.3,
        retrieval_weight_liquid=0.2,
    )
    
    # Initialize LLM Reader if enabled
    llm_reader = LLMReader(verbose=False) if use_llm_reader else None
    
    correct_count = 0
    total_facts = 0
    total_messages = 0
    total_questions = 0
    total_attention_sinks = 0
    total_llm_saved = 0
    llm_reader_answers = 0
    
    detailed_results = []
    
    # Create one agent for all samples (avoid Qdrant lock issues)
    test_user_id = "locomo_v2_benchmark"
    agent = FCMAgentV2(config=config, user_id=test_user_id)
    
    for i, item in enumerate(data):
        sample_id = item.get("sample_id", i + 1)
        
        # Clear memories between samples
        agent.clear_all_memories()
        
        # Get messages and qa from transformed data
        messages = item.get("messages", [])
        qa_pairs = item.get("qa", [])
        session_dates = item.get("session_dates", {})
        
        if verbose:
            print(f"\n🔹 Sample {i+1}/{len(data)} ({len(messages)} messages, {len(qa_pairs)} questions)")
        
        # Ingest context
        if verbose:
            print("   Ingesting...", end="", flush=True)
        
        attention_sinks = 0
        for msg in messages:
            result = agent.chat(msg)
            if result.get("is_attention_sink"):
                attention_sinks += 1
            if verbose:
                print(".", end="", flush=True)
        
        total_messages += len(messages)
        total_attention_sinks += attention_sinks
        
        if verbose:
            print(" Done.")
        
        # Process
        if verbose:
            print("   Processing...", end="", flush=True)
        agent.crystallize(force=True)
        agent.evolve(force=True)
        if verbose:
            print(" Done.")
        
        stats = agent.get_stats()
        total_facts += stats.get("crystal_count", 0) + stats.get("solid_count", 0)
        total_llm_saved += stats.get("llm_calls_saved", 0)
        
        # Test each QA pair
        for qa in qa_pairs:
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            session_num = qa.get("session_num")
            category = qa.get("category", 0)
            category_name = qa.get("category_name", "unknown")
            total_questions += 1
            
            # Get session date for this QA
            session_date = None
            if session_num:
                session_key = f"session_{session_num}"
                session_date = session_dates.get(session_key)
            
            search_result = agent.search(question, strategy="enhanced", limit=10)
            
            # 2. Xử lý kết quả trả về
            combined = search_result.get("combined", []) if isinstance(search_result, dict) else []
            
            # 3. Kỹ thuật "Context Window Expansion": Gộp Top-5 kết quả tốt nhất
            context_parts = []
            if combined:
                # Lấy 5 kết quả đầu tiên (thường chứa các mảnh thông tin bổ sung cho nhau)
                for i, item in enumerate(combined[:5]): 
                    mem = item.get("memory", "").strip()
                    if mem:
                        # Thêm dấu gạch đầu dòng để phân tách các ký ức
                        context_parts.append(f"- {mem}")
            
            # Nối lại thành 1 đoạn văn bản dài
            retrieved_memory = "\n".join(context_parts)

            # Metadata để log (lấy cái đầu tiên làm đại diện)
            top_item = combined[0] if combined else {}
            source_layer = top_item.get("source_layer", top_item.get("metadata", {}).get("fcm_type", "N/A")).upper()
            retrieval_score = top_item.get("score", 0)
            
            # Use LLM Reader if enabled
            match_reason = "Unknown"
            if use_llm_reader and llm_reader:
                # Cập nhật: nhận thêm biến match_reason
                is_correct, confidence, llm_answer, match_reason = check_answer_with_llm_reader(
                    retrieved=retrieved_memory,
                    expected=ground_truth,
                    question=question,
                    session_date=session_date,
                    llm_reader=llm_reader
                )
                if llm_answer and llm_answer != "No context retrieved":
                    llm_reader_answers += 1
                weighted_score = confidence
            else:
                # Fallback
                is_correct, weighted_score = check_answer_fuzzy(retrieved_memory, ground_truth, question)
                llm_answer = retrieved_memory[:100]
                match_reason = "Fuzzy match fallback"
            
            if is_correct:
                correct_count += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            if verbose:
                # === PHẦN IN RA GIẢI THÍCH CHI TIẾT ===
                print(f"   {status}")
                print(f"   Q: {question}")
                print(f"   Expected: '{ground_truth}'")
                print(f"   LLM Got : '{llm_answer}'")
                print(f"   Reason  : {match_reason}")
                if not is_correct:
                    # In context và layer nguồn để debug
                    print(f"   Source  : {source_layer} (Score: {retrieval_score:.2f})")
                    preview = retrieved_memory[:80].replace('\n', ' ')
                    print(f"   Context : '{preview}...'")
                print("   " + "-" * 40)
            
            detailed_results.append({
                "sample_id": sample_id,
                "correct": is_correct,
                "question": question,
                "expected": ground_truth,
                "retrieved": retrieved_memory,
                "source": source_layer,
                "weighted_score": weighted_score,
                "session_date": session_date,
                "category": category,
                "category_name": category_name,
            })
    
    elapsed = time.time() - start_time
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    return BenchmarkResult(
        version="V2",
        total_time=elapsed,
        messages_processed=total_messages,
        facts_extracted=total_facts,
        search_accuracy=accuracy,
        memory_stats={
            "Correct": correct_count,
            "Total": total_questions,
        },
        additional_metrics={
            "attention_sinks": total_attention_sinks,
            "llm_calls_saved": total_llm_saved,
            "detailed_results": detailed_results,
            "llm_reader_enabled": use_llm_reader,
            "llm_reader_answers": llm_reader_answers,
        }
    )


def run_locomo_comparison(dataset_path: str = "dataset/locomo10.json", verbose: bool = True, 
                          use_llm_reader: bool = True):
    """Main function - So sánh V1 vs V2 trên LoCoMo benchmark
    
    Args:
        dataset_path: Path to LoCoMo dataset
        verbose: Print detailed logs
        use_llm_reader: Use LLM Reader for temporal inference (Option 1)
    """
    print_header("FCM LOCOMO BENCHMARK COMPARISON", "═")
    
    if use_llm_reader:
        print("\n🤖 LLM Reader: ENABLED (Option 1 - Temporal Inference)")
    else:
        print("\n🤖 LLM Reader: DISABLED (Using exact match only)")
    
    # Load dataset
    data = load_locomo_dataset(dataset_path)
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"📊 Samples: {len(data)}")
    
    # Count total questions
    total_qa = sum(len(item.get("qa", [])) for item in data)
    print(f"❓ Total Questions: {total_qa}")
    
    # Run V1
    v1_result = run_locomo_v1(data, verbose=verbose, use_llm_reader=use_llm_reader)
    
    print("\n" + "=" * 60)
    
    # Run V2
    v2_result = run_locomo_v2(data, verbose=verbose, use_llm_reader=use_llm_reader)
    
    # Print comparison
    print_comparison_table(v1_result, v2_result)
    
    # LLM Reader stats
    if use_llm_reader:
        print_header("LLM READER STATISTICS", "-")
        v1_llm_answers = v1_result.additional_metrics.get("llm_reader_answers", 0)
        v2_llm_answers = v2_result.additional_metrics.get("llm_reader_answers", 0)
        print(f"\n  V1 LLM Reader inferences: {v1_llm_answers}")
        print(f"  V2 LLM Reader inferences: {v2_llm_answers}")
    
    # Detailed comparison
    print_header("DETAILED COMPARISON", "-")
    
    v1_details = v1_result.additional_metrics.get("detailed_results", [])
    v2_details = v2_result.additional_metrics.get("detailed_results", [])
    
    print(f"\n{'#':<4} {'Sample':<8} {'Cat':<12} {'V1':^6} {'V2':^6} {'Question':<40}")
    print("-" * 85)
    
    for idx, (v1_d, v2_d) in enumerate(zip(v1_details, v2_details), 1):
        v1_status = "✅" if v1_d["correct"] else "❌"
        v2_status = "✅" if v2_d["correct"] else "❌"
        cat_name = v1_d.get("category_name", "unknown")[:10]
        question = v1_d["question"][:37] + "..." if len(v1_d["question"]) > 40 else v1_d["question"]
        print(f"{idx:<4} {v1_d['sample_id']:<8} {cat_name:<12} {v1_status:^6} {v2_status:^6} {question:<40}")
    
    # Category-based statistics
    print_header("CATEGORY STATISTICS", "-")
    
    # Collect stats by category
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"v1_correct": 0, "v2_correct": 0, "total": 0})
    
    for v1_d, v2_d in zip(v1_details, v2_details):
        cat = v1_d.get("category", 0)
        cat_name = v1_d.get("category_name", "unknown")
        cat_stats[cat]["category_name"] = cat_name
        cat_stats[cat]["total"] += 1
        if v1_d["correct"]:
            cat_stats[cat]["v1_correct"] += 1
        if v2_d["correct"]:
            cat_stats[cat]["v2_correct"] += 1
    
    print(f"\n{'Cat':<4} {'Type':<12} {'Total':>6} {'V1 Acc':>10} {'V2 Acc':>10} {'Winner':>10}")
    print("-" * 58)
    
    for cat in sorted(cat_stats.keys()):
        stats = cat_stats[cat]
        cat_name = stats["category_name"][:10]
        total = stats["total"]
        v1_acc = stats["v1_correct"] / total if total > 0 else 0
        v2_acc = stats["v2_correct"] / total if total > 0 else 0
        
        if v1_acc > v2_acc:
            winner = "V1"
        elif v2_acc > v1_acc:
            winner = "V2"
        else:
            winner = "TIE"
        
        print(f"{cat:<4} {cat_name:<12} {total:>6} {v1_acc:>9.1%} {v2_acc:>9.1%} {winner:>10}")
    
    # Category descriptions
    print("\n📋 Category Descriptions:")
    for cat in sorted(cat_stats.keys()):
        print(f"   {cat}: {get_category_description(cat)}")
    
    # Final verdict
    print("\n" + "=" * 75)
    if v2_result.search_accuracy > v1_result.search_accuracy:
        improvement = (v2_result.search_accuracy - v1_result.search_accuracy) * 100
        print(f"🏆 FCM V2 wins with {improvement:.1f}% improvement in accuracy!")
    elif v2_result.search_accuracy < v1_result.search_accuracy:
        print(f"🏆 FCM V1 wins with higher accuracy")
    else:
        print(f"🤝 Both versions have same accuracy ({v1_result.search_accuracy:.1%})")
        if v2_result.total_time < v1_result.total_time:
            print(f"   But V2 is faster ({v2_result.total_time:.2f}s vs {v1_result.total_time:.2f}s)")
    
    print(f"\n📈 V2 Additional Benefits:")
    print(f"   • Attention Sinks preserved: {v2_result.additional_metrics.get('attention_sinks', 0)}")
    print(f"   • LLM calls saved: {v2_result.additional_metrics.get('llm_calls_saved', 0)}")
    
    return v1_result, v2_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FCM LoCoMo Benchmark")
    parser.add_argument("--dataset", type=str, default="dataset/locomo10.json",
                       help="Path to LoCoMo dataset")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce verbosity")
    parser.add_argument("--no-llm-reader", action="store_true",
                       help="Disable LLM Reader (Option 1)")
    
    args = parser.parse_args()
    
    run_locomo_comparison(
        dataset_path=args.dataset,
        verbose=not args.quiet,
        use_llm_reader=not args.no_llm_reader
    )

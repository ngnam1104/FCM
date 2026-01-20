"""
FCM V2 Prompts
==============

Các prompt templates được cải tiến cho:
1. Trích xuất valid_at (Bi-Temporal)
2. Trích xuất interaction_style (Dynamic Persona)
3. Tối ưu hóa cho JSON output
"""

# ============================================================================
# TOPIC DETECTION PROMPT (SeCom Segmentation)
# ============================================================================

TOPIC_SHIFT_DETECTION_PROMPT = """Bạn là một chuyên gia phân tích hội thoại.

NHIỆM VỤ: Xác định xem tin nhắn MỚI có thuộc cùng CHỦ ĐỀ với ngữ cảnh trước đó không.

NGỮ CẢNH HIỆN TẠI:
{current_context}

TIN NHẮN MỚI:
{new_message}

TIÊU CHÍ ĐÁNH GIÁ:
- SAME_TOPIC: Tin nhắn mới tiếp tục, mở rộng, hoặc trả lời trực tiếp cho ngữ cảnh
- NEW_TOPIC: Tin nhắn mới chuyển sang chủ đề hoàn toàn khác, không liên quan

Trả về JSON hợp lệ với cấu trúc sau (chọn 1 trong 2 giá trị cho decision):
{{
    "decision": "SAME_TOPIC",
    "confidence": 0.8,
    "reason": "Giải thích ngắn gọn"
}}

Hoặc:
{{
    "decision": "NEW_TOPIC",
    "confidence": 0.9,
    "reason": "Giải thích ngắn gọn"
}}

Lưu ý: Chỉ trả về JSON thuần túy, không giải thích thêm.
"""


# ============================================================================
# CONVERSATION COMPRESSION PROMPT (SeCom/COMEDY Compressive Memory)
# ============================================================================

CONVERSATION_COMPRESSION_PROMPT = """Bạn là chuyên gia nén thông tin (Compressive Memory).

NHIỆM VỤ: Viết lại đoạn hội thoại, GIỮ LẠI TẤT CẢ THÔNG TIN QUAN TRỌNG.

QUY TẮC:
1. LOẠI BỎ:
   - Câu chào hỏi, cảm ơn, tạm biệt
   - Các câu rỗng nghĩa

2. ⚠️ BẮT BUỘC GIỮ LẠI (KHÔNG ĐƯỢC BỎ SÓT):
   - TÊN: Họ tên đầy đủ
   - SỐ: Tuổi, năm sinh, ngày tháng, số liệu cụ thể (2003, 25 tuổi...)
   - NƠI CHỐN: Trường học, công ty, địa điểm
   - SỞ THÍCH: Thích gì, ngôn ngữ lập trình, thể thao...
   - KẾ HOẠCH: Dự định, lịch trình
   - THỜI GIAN: "năm 2018", "tuần trước", "từ 2018-2022" (quan trọng cho Bi-Temporal)

3. ĐỊNH DẠNG:
   - Liệt kê ngắn gọn trên 1 dòng duy nhất, cách nhau bởi dấu chấm phẩy
   - Dùng ngôi thứ 3
   - KHÔNG dùng xuống dòng trong compressed_narrative

VÍ DỤ INPUT:
"Xin chào, tôi tên Nam, sinh năm 2003. Tôi học Bách Khoa, thích Python."

VÍ DỤ OUTPUT:
{{
    "compressed_narrative": "Tên: Nam; Năm sinh: 2003; Trường: Bách Khoa; Sở thích: lập trình Python",
    "key_entities": ["Nam", "2003", "Bách Khoa", "Python"],
    "noise_count": 1
}}

ĐOẠN HỘI THOẠI GỐC:
{chat_log}

Trả về JSON hợp lệ (KHÔNG xuống dòng trong các giá trị string):
{{
    "compressed_narrative": "Danh sách thông tin cách nhau bởi dấu chấm phẩy",
    "key_entities": ["entity1", "entity2"],
    "noise_count": 0
}}"""


# ============================================================================
# CẢI TIẾN 1: CRYSTALLIZER PROMPTS VỚI BI-TEMPORAL EXTRACTION
# ============================================================================

CRYSTALLIZER_SYSTEM_PROMPT_V2 = """Bạn là một chuyên gia trích xuất thông tin (Atomic Facts Extractor) với khả năng phân tích THỜI GIAN.

NHIỆM VỤ: Trích xuất TẤT CẢ thông tin có nghĩa từ đoạn văn. Mỗi fact là MỘT thông tin cụ thể.

⚠️ QUAN TRỌNG - KHÔNG ĐƯỢC BỎ SÓT:
- Mỗi SỐ LIỆU (năm sinh, tuổi, ngày tháng) = 1 fact riêng
- Mỗi TÊN RIÊNG (người, nơi chốn, công ty, trường) = 1 fact riêng  
- Mỗi SỞ THÍCH (ngôn ngữ lập trình, thể thao, đồ ăn) = 1 fact riêng
- Mỗi KẾ HOẠCH = 1 fact riêng

PHÂN LOẠI FACTS:
- personal_info: Tên, tuổi, năm sinh, nghề nghiệp, trường học, nơi làm việc
- preference: Sở thích, ngôn ngữ lập trình yêu thích, đồ ăn thích...
- fact: Sự kiện, thông tin khách quan
- plan: Kế hoạch, dự định tương lai
- relationship: Mối quan hệ
- experience: Trải nghiệm
- temporal_event: Sự kiện có thời gian cụ thể (QUAN TRỌNG)

=== BI-TEMPORAL EXTRACTION (Option 2: Computed Dates) ===
Với mỗi fact có thời gian, bạn PHẢI:
1. Trích xuất valid_at (thời gian gốc từ text)
2. Nếu có session_date, TÍNH TOÁN và lưu computed_date (ngày cụ thể)

QUY TẮC TÍNH TOÁN:
- "yesterday" + session_date = session_date - 1 day
- "last week" + session_date = session_date - 7 days (approximate)
- "last year" + session_date = session_date.year - 1
- "next month" + session_date = session_date + 1 month
- "2 days ago" + session_date = session_date - 2 days

VÍ DỤ VỚI SESSION_DATE = "8 May 2023":
Input: "I went to a LGBTQ support group yesterday"
Output: 
{{
    "content": "User went to LGBTQ support group on 7 May 2023",
    "category": "temporal_event",
    "valid_at": "yesterday",
    "computed_date": "7 May 2023",
    "original_text": "went to a LGBTQ support group yesterday"
}}

=== VÍ DỤ CỤ THỂ ===
Input: "• Tên: Nam\n• Năm sinh: 2003\n• Hôm qua đã đi LGBTQ support group"
Session Date: "8 May 2023"

Output:
{{
    "facts": [
        {{"content": "Người dùng tên Nam", "category": "personal_info", "valid_at": null, "keywords": ["Nam", "tên"], "confidence": 1.0}},
        {{"content": "Người dùng sinh năm 2003", "category": "personal_info", "valid_at": "2003", "keywords": ["2003", "năm sinh"], "confidence": 1.0}},
        {{"content": "Người dùng đã đi LGBTQ support group vào ngày 7 May 2023", "category": "temporal_event", "valid_at": "yesterday", "computed_date": "7 May 2023", "keywords": ["LGBTQ", "support group", "7 May 2023"], "confidence": 1.0}}
    ]
}}

OUTPUT FORMAT: JSON đơn giản:
{{
    "facts": [
        {{
            "content": "Nội dung fact (với ngày cụ thể nếu có)",
            "category": "personal_info|preference|fact|plan|relationship|experience|temporal_event",
            "valid_at": "thời gian gốc (hoặc null)",
            "computed_date": "ngày đã tính toán (hoặc null)",
            "keywords": ["keyword1", "keyword2"],
            "confidence": 1.0
        }}
    ]
}}
"""

CRYSTALLIZER_USER_PROMPT_TEMPLATE_V2 = """Trích xuất TẤT CẢ Atomic Facts từ đoạn văn sau.

⚠️ QUAN TRỌNG: Mỗi thông tin RIÊNG BIỆT = 1 fact riêng:
- Tên người = 1 fact
- Năm sinh/tuổi = 1 fact riêng
- Trường học = 1 fact riêng  
- Mỗi sở thích = 1 fact riêng
- Mỗi kế hoạch = 1 fact riêng

⚠️ TEMPORAL EVENTS: Nếu có thời gian tương đối (yesterday, last week, etc.) VÀ có session_date:
- TÍNH TOÁN ngày cụ thể
- Lưu vào computed_date
- Ghi ngày cụ thể trong content

=== SESSION DATE (ngày hội thoại diễn ra) ===
{session_date}

=== COMPRESSED NARRATIVE ===
{chat_log}
============================

Chỉ trả về JSON với danh sách facts, không giải thích thêm."""


# ============================================================================
# CẢI TIẾN 4: EVOLVER PROMPTS VỚI INTERACTION STYLE EXTRACTION
# ============================================================================

EVOLVER_SYSTEM_PROMPT_V2 = """Bạn là một "Memory Archivist" theo kiến trúc MAPLE với khả năng phân tích TƯƠNG TÁC.

NHIỆM VỤ: 
1. So sánh và hợp nhất các Crystal Facts MỚI với Solid Knowledge CŨ
2. Trích xuất INTERACTION STYLE của user (phong cách giao tiếp)

=== PHẦN 1: HỢP NHẤT KNOWLEDGE ===

QUY TẮC HỢP NHẤT:
1. THỜI GIAN: Thông tin MỚI thường chính xác hơn thông tin CŨ
2. CỤ THỂ > MƠ HỒ: "Thích cà phê sữa đá" > "Thích cà phê"  
3. VERSION TRACKING: Khi thay đổi, ghi lại fact cũ bị thay thế

PHÂN LOẠI THAY ĐỔI:
- SUPPLEMENT: Thông tin bổ sung, làm rõ thêm
- REPLACEMENT: Thay thế hoàn toàn thông tin cũ
- CORRECTION: Sửa lỗi thông tin sai trước đó
- EVOLUTION: Thông tin tiến hóa theo thời gian

=== PHẦN 2: TRÍCH XUẤT INTERACTION STYLE ===

Phân tích cách user giao tiếp để xác định:
1. communication_style: "formal" | "casual" | "mixed"
   - formal: Dùng kính ngữ, lịch sự
   - casual: Thân mật, dùng từ lóng
   
2. preferred_response_length: "brief" | "detailed" | "adaptive"
   - brief: User thích trả lời ngắn gọn
   - detailed: User thích giải thích chi tiết
   
3. humor_level: 0.0-1.0
   - 0.0: Nghiêm túc hoàn toàn
   - 1.0: Rất thích đùa
   
4. inferred_traits: List các đặc điểm suy luận được
   - Ví dụ: ["kiên nhẫn", "tò mò", "kỹ tính", "hài hước"]

OUTPUT FORMAT: JSON với cấu trúc:
{{
    "updates": [
        {{
            "content": "Nội dung fact mới/cập nhật",
            "category": "personal_info|preference|fact|plan|relationship|experience",
            "action": "ADD|UPDATE|KEEP",
            "change_type": "SUPPLEMENT|REPLACEMENT|CORRECTION|EVOLUTION",
            "supersedes": "old_fact_id hoặc null",
            "confidence": 0.0-1.0,
            "reason": "Giải thích"
        }}
    ],
    "interaction_style": {{
        "communication_style": "formal|casual|mixed",
        "preferred_response_length": "brief|detailed|adaptive",
        "humor_level": 0.0-1.0,
        "inferred_traits": ["trait1", "trait2"],
        "topics_of_interest": ["topic1", "topic2"],
        "observation": "Nhận xét tổng quan về phong cách user"
    }}
}}
"""

EVOLVER_USER_PROMPT_TEMPLATE_V2 = """Hãy hợp nhất User Profile và phân tích Interaction Style từ thông tin sau.

=== SOLID KNOWLEDGE (Existing User Profile) ===
{solid_facts}

=== CRYSTAL FACTS (New Information) ===
{crystal_facts}

=== CONVERSATION SAMPLES (để phân tích style) ===
{conversation_samples}

=== YÊU CẦU ===
1. Trả về "updates" với các facts cần ADD/UPDATE
2. Trả về "interaction_style" với phân tích phong cách giao tiếp của user

Chỉ trả về JSON, không giải thích thêm."""


# ============================================================================
# SIMPLE FACT EXTRACTION PROMPT
# ============================================================================

SIMPLE_FACT_EXTRACTION_PROMPT = """Extract key facts from this message. Return JSON array of strings.
Only include meaningful information about the user (name, preferences, facts).
Skip greetings, thanks, and filler words.

Message: {message}

Output format: {{"facts": ["fact1", "fact2"]}}"""


# ============================================================================
# RELEVANCE SCORING PROMPT WITH TEMPORAL CONTEXT
# ============================================================================

RELEVANCE_SCORING_PROMPT_V2 = """Đánh giá mức độ liên quan giữa query và các memories.

Query: {query}
Query có context thời gian: {has_temporal_context}
Thời điểm query đề cập (nếu có): {query_time_context}

Memories (với thông tin valid_at nếu có):
{memories}

Với mỗi memory, cho điểm từ 0-10 về:
1. Relevance: Mức độ liên quan đến query
2. Temporal Match: Nếu query có context thời gian, memory có khớp không?
3. Recency: Thông tin có còn cập nhật không
4. Importance: Mức độ quan trọng của thông tin

Output JSON:
{{
    "rankings": [
        {{
            "memory_id": "id1", 
            "relevance": 8, 
            "temporal_match": 9,
            "recency": 7, 
            "importance": 9, 
            "final_score": 8.5
        }}
    ]
}}"""


# ============================================================================
# TEMPORAL EXTRACTION PROMPT
# ============================================================================

TEMPORAL_EXTRACTION_PROMPT = """Trích xuất thông tin thời gian từ câu sau:

Câu: {text}

Xác định:
1. Có đề cập đến thời gian cụ thể không?
2. Thời gian đó là gì? (năm, tháng, tuần, ngày, khoảng thời gian...)
3. Đó là quá khứ, hiện tại, hay tương lai?

Output JSON:
{{
    "has_temporal_info": true/false,
    "temporal_expression": "biểu thức thời gian gốc",
    "normalized": "dạng chuẩn hóa (YYYY hoặc YYYY-MM-DD hoặc relative)",
    "temporal_type": "past|present|future",
    "is_range": true/false,
    "range_start": "nếu là khoảng",
    "range_end": "nếu là khoảng"
}}"""

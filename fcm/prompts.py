"""
FCM Prompts
===========

Các prompt templates cho Crystallizer và Evolver agents.
Dựa trên nguyên lý của SeCom (Segmentation & Denoising) và MAPLE (Archiver Agent).
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

Trả về JSON:
{{
    "decision": "SAME_TOPIC" | "NEW_TOPIC",
    "confidence": 0.0-1.0,
    "reason": "Giải thích ngắn gọn"
}}
Hãy đảm bảo trả về JSON hợp lệ, không thiếu dấu đóng mở ngoặc.
"""


# ============================================================================
# CONVERSATION COMPRESSION PROMPT (SeCom/COMEDY Compressive Memory)
# ============================================================================

CONVERSATION_COMPRESSION_PROMPT = """Bạn là chuyên gia nén thông tin (Compressive Memory).

NHIỆM VỤ: Viết lại đoạn hội thoại thành một đoạn VĂN XUÔI, GIỮ LẠI TẤT CẢ THÔNG TIN QUAN TRỌNG.

QUY TẮC:
1. LOẠI BỎ:
   - Câu chào hỏi ("Xin chào", "Hi")
   - Câu cảm ơn, tạm biệt
   - Các câu rỗng nghĩa

2. ⚠️ BẮT BUỘC GIỮ LẠI (KHÔNG ĐƯỢC BỎ SÓT):
   - TÊN: Họ tên đầy đủ
   - SỐ: Tuổi, năm sinh, ngày tháng, số liệu cụ thể
   - NƠI CHỐN: Trường học, công ty, địa điểm
   - SỞ THÍCH: Thích gì, không thích gì, ngôn ngữ lập trình, thể thao...
   - KẾ HOẠCH: Dự định, lịch trình
   - MỐI QUAN HỆ: Bạn bè, gia đình

3. ĐỊNH DẠNG:
   - Liệt kê dạng bullet points để không mất thông tin
   - Dùng ngôi thứ 3

VÍ DỤ INPUT:
"Tôi tên Nam, sinh năm 2003. Tôi học Bách Khoa, thích Python."

VÍ DỤ OUTPUT:
{{
    "compressed_narrative": "• Tên: Nam\n• Năm sinh: 2003\n• Trường: Bách Khoa\n• Sở thích: lập trình Python",
    "key_entities": ["Nam", "2003", "Bách Khoa", "Python"],
    "noise_count": 0
}}

ĐOẠN HỘI THOẠI GỐC:
{chat_log}

Trả về JSON:
{{
    "compressed_narrative": "Danh sách bullet points...",
    "key_entities": ["entity1", "entity2"],
    "noise_count": <số câu đã loại bỏ>
}}"""


# ============================================================================
# CRYSTALLIZER PROMPTS (SeCom + A-Mem Zettelkasten inspired)
# ============================================================================

CRYSTALLIZER_SYSTEM_PROMPT = """Bạn là một chuyên gia trích xuất thông tin (Atomic Facts Extractor).

NHIỆM VỤ: Trích xuất TẤT CẢ thông tin có nghĩa từ đoạn văn. Mỗi fact là MỘT thông tin cụ thể.

⚠️ QUAN TRỌNG - KHÔNG ĐƯỢC BỎ SÓT:
- Mỗi SỐ LIỆU (năm, tuổi, ngày) = 1 fact riêng
- Mỗi TÊN RIÊNG (người, nơi chốn, công ty) = 1 fact riêng
- Mỗi SỞ THÍCH = 1 fact riêng
- Mỗi KẾ HOẠCH = 1 fact riêng

PHÂN LOẠI FACTS:
- personal_info: Tên, tuổi, năm sinh, nghề nghiệp, trường học, nơi làm việc
- preference: Sở thích, ngôn ngữ lập trình yêu thích, đồ ăn thích...
- fact: Sự kiện, thông tin khách quan
- plan: Kế hoạch, dự định tương lai
- relationship: Mối quan hệ
- experience: Trải nghiệm

VÍ DỤ:
Input: "Nam, sinh năm 2003, học Bách Khoa, thích Python và AI"
Output facts:
1. "Người dùng tên Nam" (personal_info)
2. "Người dùng sinh năm 2003" (personal_info)  
3. "Người dùng học tại Bách Khoa" (personal_info)
4. "Người dùng thích lập trình Python" (preference)
5. "Người dùng thích AI" (preference)

YÊU CẦU VỀ LIÊN KẾT:
- keywords: 2-4 từ khóa chính
- context_tags: Ngữ cảnh (work, travel, coding, food, hobby...)
- related_to: Entity liên quan

OUTPUT FORMAT: JSON đơn giản:
{
    "facts": [
        {
            "content": "Nội dung fact",
            "category": "personal_info|preference|fact|plan|relationship|experience",
            "keywords": ["keyword1", "keyword2"],
            "confidence": 1.0
        }
    ]
}
"""

CRYSTALLIZER_USER_PROMPT_TEMPLATE = """Trích xuất TẤT CẢ Atomic Facts từ đoạn văn sau.

⚠️ QUAN TRỌNG: Mỗi thông tin RIÊNG BIỆT = 1 fact riêng:
- Tên người = 1 fact
- Năm sinh/tuổi = 1 fact riêng
- Trường học = 1 fact riêng  
- Mỗi sở thích = 1 fact riêng

=== COMPRESSED NARRATIVE ===
{chat_log}
============================

Chỉ trả về JSON, không giải thích thêm."""


# ============================================================================
# EVOLVER PROMPTS (MAPLE Archiver-inspired with Versioning + Reflection)
# ============================================================================

EVOLVER_SYSTEM_PROMPT = """Bạn là một "Memory Archivist" theo kiến trúc MAPLE - chuyên gia quản lý và hợp nhất bộ nhớ dài hạn với khả năng TRUY VẾT LỊCH SỬ và PHẢN CHIẾU (REFLECTION).

NHIỆM VỤ: So sánh và hợp nhất các Crystal Facts MỚI với Solid Knowledge CŨ để tạo ra:
1. User Profile được cập nhật với VERSION TRACKING
2. Giải quyết các mâu thuẫn thông tin (Conflict Resolution)
3. Tạo liên kết giữa versions (Linked List of Knowledge)

QUY TẮC HỢP NHẤT:
1. THỜI GIAN: Thông tin MỚI thường chính xác hơn thông tin CŨ
2. CỤ THỂ > MƠ HỒ: "Thích cà phê sữa đá" > "Thích cà phê"  
3. VERSION TRACKING: Khi thay đổi, ghi lại fact cũ bị thay thế

XỬ LÝ MÂU THUẪN VỚI PHẢN CHIẾU (MAPLE REFLECTOR):
Trước khi sửa đổi bộ nhớ, hãy THỰC HIỆN SUY LUẬN (Reasoning/Reflection):

1. PHÂN LOẠI THAY ĐỔI:
   - SUPPLEMENT: Thông tin bổ sung, làm rõ thêm (không mâu thuẫn)
   - REPLACEMENT: Thay thế hoàn toàn thông tin cũ
   - CORRECTION: Sửa lỗi thông tin sai trước đó
   - EVOLUTION: Thông tin tiến hóa theo thời gian (sở thích thay đổi)

2. SUY LUẬN NGUYÊN NHÂN (Root Cause Reasoning):
   - Tại sao có sự thay đổi này?
   - User đổi ý? Hoàn cảnh thay đổi? Thông tin cũ sai?
   - Đây là thay đổi tạm thời hay vĩnh viễn?

3. GHI LÝ DO CHI TIẾT:
   - Trường "reason" phải giải thích rõ ràng
   - Ví dụ: "User thay đổi sở thích từ trà sang cafe do lý do sức khỏe"
   - Ví dụ: "Bổ sung thông tin chi tiết hơn về nơi làm việc"

OUTPUT FORMAT: JSON với cấu trúc:
{
    "reflection": {
        "total_new_facts": <số facts mới>,
        "conflicts_detected": <số mâu thuẫn>,
        "supplements_detected": <số bổ sung>,
        "overall_assessment": "Mô tả ngắn về sự thay đổi tổng thể"
    },
    "updated_facts": [
        {
            "content": "Nội dung fact mới/cập nhật",
            "category": "personal_info|preference|fact|plan|relationship|experience",
            "action": "NEW|UPDATE|KEEP",
            "change_type": "SUPPLEMENT|REPLACEMENT|CORRECTION|EVOLUTION",
            "supersedes": "old_fact_id hoặc null nếu là fact mới",
            "validity_start": "ISO timestamp khi fact này bắt đầu đúng",
            "confidence": 0.0-1.0,
            "reason": "Giải thích chi tiết tại sao thực hiện action này",
            "keywords": ["keyword1", "keyword2"]
        }
    ],
    "archived_facts": [
        {
            "old_fact_id": "ID của fact cũ bị thay thế",
            "old_content": "Nội dung cũ",
            "superseded_by": "Nội dung fact mới thay thế",
            "valid_until": "ISO timestamp khi fact này hết hiệu lực",
            "archive_reason": "Lý do archive (REPLACED|CORRECTED|EVOLVED)"
        }
    ],
    "user_profile_summary": {
        "personal_info": ["fact1", "fact2"],
        "preferences": ["pref1", "pref2"],
        "relationships": ["rel1"],
        "plans": ["plan1"],
        "experiences": ["exp1"]
    }
}
"""

EVOLVER_USER_PROMPT_TEMPLATE = """Hãy hợp nhất và cập nhật User Profile từ thông tin sau.

=== SOLID KNOWLEDGE (Existing User Profile) ===
{solid_facts}

=== CRYSTAL FACTS (New Information) ===
{crystal_facts}

=== YÊU CẦU BẮT BUỘC ===
Trả về JSON object chứa danh sách "updates". TUYỆT ĐỐI KHÔNG dùng key "reflection".
Ví dụ format đúng:
{{
    "updates": [
        {{
            "action": "ADD",
            "content": "Người dùng thích lập trình Python (đã xác nhận)",
            "category": "preference",
            "confidence": 1.0,
            "reason": "Thông tin mới từ hội thoại"
        }}
    ]
}}

Chỉ trả về JSON, không giải thích thêm."""


# ============================================================================
# FACT EXTRACTION PROMPTS (Simple version for quick extraction)
# ============================================================================

SIMPLE_FACT_EXTRACTION_PROMPT = """Extract ALL specific facts from the messages below.

RULES - MUST extract EACH as SEPARATE fact:
- NAMES: "User's name is X"
- NUMBERS: Birth year, age, dates → "User was born in YYYY"
- PLACES: School, company, city → "User studies at X" 
- PREFERENCES: Each hobby/preference → "User likes X"
- PLANS: Each plan → "User plans to X"

Messages:
{message}

Return JSON ONLY:
{{"facts": ["Người dùng tên Nam", "Người dùng sinh năm 2003", "Người dùng học Bách Khoa", "Người dùng thích Python"]}}"""

# Prompt đơn giản hơn cho extract fact
DIRECT_FACT_EXTRACTION_PROMPT = """Từ các tin nhắn sau, liệt kê TẤT CẢ thông tin cụ thể.

Tin nhắn:
{messages}

QUAN TRỌNG: Mỗi thông tin = 1 dòng riêng. Ví dụ:
- Nếu thấy "tên Nam" → "Người dùng tên Nam"
- Nếu thấy "sinh 2003" → "Người dùng sinh năm 2003"  
- Nếu thấy "học Bách Khoa" → "Người dùng học tại Bách Khoa"
- Nếu thấy "thích Python" → "Người dùng thích lập trình Python"

Trả về JSON:
{{"facts": ["fact1", "fact2", "fact3"]}}"""


# ============================================================================
# RELEVANCE SCORING PROMPT
# ============================================================================

RELEVANCE_SCORING_PROMPT = """Đánh giá mức độ liên quan giữa query và các memories.

Query: {query}

Memories:
{memories}

Với mỗi memory, cho điểm từ 0-10 về:
1. Relevance: Mức độ liên quan đến query
2. Recency: Thông tin có còn cập nhật không (1=cũ, 10=mới)
3. Importance: Mức độ quan trọng của thông tin

Output JSON:
{{
    "rankings": [
        {{"memory_id": "id1", "relevance": 8, "recency": 7, "importance": 9, "final_score": 8.0}},
        ...
    ]
}}"""

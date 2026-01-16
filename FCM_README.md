# Frequency-based Crystallizing Memory (FCM)

## A Hierarchical Memory Architecture for AI Agents

**Version:** 0.2.3  
**Authors:** Project III - HUST 2024-2025  
**Based on:** Nested Learning, SeCom, A-Mem, MAPLE, InfLLM

---

# PHẦN 1: LÝ THUYẾT KIẾN TRÚC

## 1.1. Giới thiệu

### 1.1.1. Bối cảnh nghiên cứu

Các hệ thống AI Agent hiện đại đối mặt với thách thức quản lý bộ nhớ trong các cuộc hội thoại dài. Kiến trúc truyền thống phân chia bộ nhớ theo **không gian** (Short-term Memory / Long-term Memory) gặp phải các vấn đề:

1. **Information Overload**: STM tràn nhanh chóng, mất context quan trọng
2. **Noise Accumulation**: LTM lưu cả thông tin không cần thiết
3. **Retrieval Confusion**: Không phân biệt được độ tin cậy của thông tin
4. **Conflict Blind**: Không xử lý được mâu thuẫn giữa thông tin cũ và mới

### 1.1.2. Đề xuất giải pháp

FCM (Frequency-based Crystallizing Memory) đề xuất phân chia bộ nhớ theo **tần số cập nhật** thay vì không gian, mô phỏng quá trình hình thành ký ức của con người:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HUMAN MEMORY ANALOGY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Sensory Input     →    Working Memory    →    Long-term       │
│   (raw perception)       (processing)           (consolidated)  │
│                                                                 │
│         ↓                     ↓                      ↓          │
│                                                                 │
│      LIQUID              CRYSTAL                 SOLID          │
│   (High-Freq)           (Mid-Freq)            (Low-Freq)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1.3. Đóng góp chính

1. **Three-Layer Architecture**: Phân tầng bộ nhớ theo tần số cập nhật
2. **LLM-driven Extraction**: Sử dụng LLM để trích xuất và lọc thông tin
3. **Semantic Segmentation**: Phân đoạn hội thoại dựa trên ngữ nghĩa (Topic Shift)
4. **Compressive Memory**: Nén hội thoại trước khi trích xuất facts
5. **Version Tracking**: Theo dõi lịch sử thay đổi của knowledge
6. **Hybrid Retrieval**: Tìm kiếm thông minh ưu tiên theo độ tin cậy

---

## 1.2. Các công trình liên quan

### 1.2.1. Nested Learning / CMS (Compressed Memory System)

**Nguồn:** "Learning to Compress: A Framework for Memory-Efficient Learning"

**Ý tưởng chính:** Learning as Compression - Quá trình học là quá trình nén thông tin từ dạng thô (raw) sang dạng tinh (compressed).

**Áp dụng vào FCM:**
- **Liquid Layer**: Raw input, không xử lý
- **Crystal Layer**: Compressed facts, loại bỏ noise
- **Solid Layer**: Highly compressed profile

```
Information Flow:
Raw Data (100%) → Filtered Facts (30%) → Core Knowledge (10%)
     ↓                    ↓                      ↓
  LIQUID              CRYSTAL                 SOLID
```

### 1.2.2. SeCom (Semantic Communication)

**Nguồn:** "SeCom: Semantic Communication for Multi-turn Dialogue"

**Ý tưởng chính:**
1. **Semantic Segmentation**: Phân đoạn hội thoại theo ngữ nghĩa, không phải theo số lượng
2. **Topic Shift Detection**: Phát hiện khi nào chủ đề thay đổi
3. **Denoising**: Loại bỏ các phát biểu không mang thông tin

**Áp dụng vào FCM:**
- `_detect_topic_shift()`: LLM phát hiện Topic Shift
- `_segment_conversation()`: Chia hội thoại thành segments có cùng chủ đề
- Crystallize trigger: Kích hoạt ngay khi Topic Shift (không đợi threshold)

```
Conversation:
[1] "Tôi tên Nam"           ─┐
[2] "Tôi học HUST"           │ Segment 1: Personal Info
[3] "Chuyên ngành CNTT"     ─┘
[4] "À mà hôm nay trời đẹp" ─┐ ← Topic Shift Detected!
[5] "Tôi thích thời tiết này"│ Segment 2: Weather/Preference
                            ─┘
```

### 1.2.3. COMEDY (Compressive Memory for Dialogue)

**Nguồn:** "COMEDY: A Compact Memory System for Dialogue"

**Ý tưởng chính:** Nén (compress) hội thoại thành narrative ngắn gọn trước khi xử lý tiếp.

**Áp dụng vào FCM:**
- `_compress_conversation()`: Viết lại hội thoại thành văn xuôi
- Loại bỏ: greetings, fillers, repetitions
- Giữ lại: facts, preferences, plans, relationships

```
Before Compression:
User: Xin chào!
Bot: Chào bạn!
User: Ờ... tôi muốn hỏi là... tôi tên Nam
User: À, tôi học ở HUST
User: Ừm, chuyên ngành CNTT

After Compression:
"Nam là sinh viên HUST, chuyên ngành CNTT."
```

### 1.2.4. A-Mem (Atomic Memory / Zettelkasten)

**Nguồn:** "A-Mem: Agentic Memory with Zettelkasten-Inspired Organization"

**Ý tưởng chính:**
1. **Atomic Notes**: Mỗi fact là một đơn vị độc lập (self-contained)
2. **Linking**: Các facts được liên kết với nhau qua keywords
3. **Emergence**: Knowledge mới xuất hiện từ các liên kết

**Áp dụng vào FCM:**
- Crystallizer output: JSON với `keywords`, `context_tags`, `related_to`
- `potential_links`: Liên kết giữa các facts trong cùng batch
- Metadata cho phép xây dựng Knowledge Graph sau này

```json
{
    "facts": [
        {
            "content": "Nam là sinh viên HUST",
            "category": "personal_info",
            "keywords": ["Nam", "sinh viên", "HUST"],
            "context_tags": ["education"],
            "related_to": ["HUST", "CNTT"],
            "confidence": 0.95
        }
    ],
    "potential_links": [
        {
            "fact_index": 0,
            "linked_to_index": 1,
            "link_type": "extends",
            "reason": "Cùng về thông tin học vấn"
        }
    ]
}
```

### 1.2.5. MAPLE (Memory Architecture with Programmable Layers)

**Nguồn:** "MAPLE: A Modular Architecture for Persistent Learning"

**Ý tưởng chính:**
1. **Archiver Agent**: Quản lý việc lưu trữ và cập nhật knowledge
2. **Version Tracking**: Theo dõi lịch sử thay đổi của mỗi fact
3. **Conflict Resolution**: Giải quyết mâu thuẫn giữa thông tin cũ/mới
4. **Reflection/Reasoning**: Suy luận về nguyên nhân thay đổi

**Áp dụng vào FCM:**
- `evolve()`: MAPLE Archiver tổng hợp Crystal → Solid
- `_archive_solid_memory()`: Lưu trữ version cũ với metadata
- `get_memory_history()`: Truy vết lịch sử thay đổi
- Evolver prompt yêu cầu `change_type` và `reason`

```
Version Tracking:
┌─────────────────────────────────────────────────────────────────┐
│ Fact: "Nam thích uống trà"                                      │
│ ID: fact_001, Created: 2024-01-01, Status: ARCHIVED             │
│ superseded_by: fact_002                                         │
├─────────────────────────────────────────────────────────────────┤
│ Fact: "Nam thích uống cà phê sữa đá"                            │
│ ID: fact_002, Created: 2024-03-15, Status: ACTIVE               │
│ supersedes: fact_001                                            │
│ change_type: EVOLUTION                                          │
│ reason: "User đổi sở thích từ trà sang cà phê"                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2.6. InfLLM (Infinite Context LLM)

**Nguồn:** "InfLLM: Training-Free Long-Context LLMs"

**Ý tưởng chính:**
1. **Representative Tokens**: Lưu các token đại diện thay vì toàn bộ context
2. **Hierarchical Retrieval**: Tìm kiếm phân cấp từ tổng quát → chi tiết
3. **Relevance Scoring**: Đánh giá độ liên quan theo ngữ cảnh

**Áp dụng vào FCM:**
- Solid memories = Representative tokens (knowledge cốt lõi)
- `search()` với strategy `hybrid`: Solid → Crystal → Liquid
- Score threshold để quyết định khi nào cần tìm sâu hơn

```
Hybrid Retrieval Flow:
Query: "Nam học trường nào?"
        │
        ▼
┌─────────────────┐
│  Search SOLID   │ → Found: "Nam là sinh viên HUST" (score=0.92)
└────────┬────────┘
         │ score >= 0.7? YES → Return
         ▼
    ┌────────────┐
    │   DONE     │
    └────────────┘

Query: "Hôm qua nói gì?"
        │
        ▼
┌─────────────────┐
│  Search SOLID   │ → No relevant result (score=0.3)
└────────┬────────┘
         │ score >= 0.7? NO → Continue
         ▼
┌─────────────────┐
│ Search CRYSTAL  │ → Found some facts (score=0.5)
└────────┬────────┘
         │ score >= 0.7? NO → Continue
         ▼
┌─────────────────┐
│  Search LIQUID  │ → Found recent messages
└────────┬────────┘
         ▼
    ┌────────────┐
    │   DONE     │
    └────────────┘
```

---

## 1.3. Kiến trúc FCM

### 1.3.1. Tổng quan hệ thống

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FCM SYSTEM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────┐                                                          │
│    │    USER     │                                                          │
│    │   MESSAGE   │                                                          │
│    └──────┬──────┘                                                          │
│           │                                                                 │
│           ▼                                                                 │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │                      LIQUID LAYER (t=1)                         │      │
│    │  ┌─────────────────────────────────────────────────────────┐    │      │
│    │  │  • Raw message storage                                  │    │      │
│    │  │  • Topic Shift Detection (SeCom)                        │    │      │
│    │  │  • Conversation Buffer                                  │    │      │
│    │  └─────────────────────────────────────────────────────────┘    │      │
│    └──────────────────────────┬──────────────────────────────────────┘      │
│                               │                                             │
│                               │ Trigger: N messages OR Topic Shift          │
│                               ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │                    CRYSTALLIZER (SeCom + A-Mem)                 │      │
│    │  ┌─────────────────────────────────────────────────────────┐    │      │
│    │  │  1. Segment by Topic (_segment_conversation)            │    │      │
│    │  │  2. Compress to Narrative (_compress_conversation)      │    │      │
│    │  │  3. Extract Atomic Facts (LLM + Zettelkasten format)    │    │      │
│    │  └─────────────────────────────────────────────────────────┘    │      │
│    └──────────────────────────┬──────────────────────────────────────┘      │
│                               │                                             │
│                               ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │                      CRYSTAL LAYER (t=N)                        │      │
│    │  ┌─────────────────────────────────────────────────────────┐    │      │
│    │  │  • Atomic Facts with keywords, context_tags             │    │      │
│    │  │  • Categories: personal_info, preference, plan, etc.    │    │      │
│    │  │  • Potential links between facts                        │    │      │
│    │  └─────────────────────────────────────────────────────────┘    │      │
│    └──────────────────────────┬──────────────────────────────────────┘      │
│                               │                                             │
│                               │ Trigger: End of Session OR M messages       │
│                               ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │                      EVOLVER (MAPLE Archiver)                   │      │
│    │  ┌─────────────────────────────────────────────────────────┐    │      │
│    │  │  1. Compare NEW Crystal facts with OLD Solid knowledge  │    │      │
│    │  │  2. Reflection: Analyze WHY changes happen              │    │      │
│    │  │  3. Resolve conflicts (SUPPLEMENT/REPLACE/CORRECT/EVOLVE)│   │      │
│    │  │  4. Archive old versions with linked history            │    │      │
│    │  └─────────────────────────────────────────────────────────┘    │      │
│    └──────────────────────────┬──────────────────────────────────────┘      │
│                               │                                             │
│                               ▼                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │                      SOLID LAYER (t=Session)                    │      │
│    │  ┌─────────────────────────────────────────────────────────┐    │      │
│    │  │  • User Profile (stable knowledge)                      │    │      │
│    │  │  • Version history with supersedes/superseded_by links  │    │      │
│    │  │  • Highest reliability for retrieval                    │    │      │
│    │  └─────────────────────────────────────────────────────────┘    │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │                    HYBRID RETRIEVER (InfLLM)                    │      │
│    │  ┌─────────────────────────────────────────────────────────┐    │      │
│    │  │  Strategies: hybrid, solid_first, all_layers, recent    │    │      │
│    │  │  Score-based fallback: Solid → Crystal → Liquid         │    │      │
│    │  └─────────────────────────────────────────────────────────┘    │      │
│    └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3.2. Liquid Layer

**Định nghĩa:** Lớp lưu trữ tần số cao (t=1), ghi nhận mọi tin nhắn nguyên bản.

**Đặc điểm:**
| Property | Value |
|----------|-------|
| Frequency | t=1 (sau mỗi message) |
| Content | Raw message, không xử lý |
| Metadata | fcm_type="liquid", role, timestamp |
| Retention | Cho đến khi được xử lý bởi Crystallizer |
| Purpose | Giữ context đầy đủ, hỗ trợ recent retrieval |

**SeCom Integration:**
- Mỗi message được kiểm tra Topic Shift trước khi lưu
- Nếu phát hiện Topic Shift (confidence >= 0.7), trigger Crystallize ngay

### 1.3.3. Crystal Layer

**Định nghĩa:** Lớp lưu trữ tần số trung bình (t=N), chứa Atomic Facts đã được lọc.

**Đặc điểm:**
| Property | Value |
|----------|-------|
| Frequency | t=N messages hoặc Topic Shift |
| Content | Atomic Facts (JSON structured) |
| Metadata | category, keywords, context_tags, confidence |
| Processing | SeCom Segmentation + COMEDY Compression + A-Mem Extraction |
| Purpose | Facts có cấu trúc, dễ retrieval |

**Crystallizer Pipeline:**
```
Input: Liquid Messages (buffer)
        │
        ▼
┌───────────────────────┐
│  1. SEGMENTATION      │  SeCom: Chia theo Topic
│     (if buffer > 10)  │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  2. COMPRESSION       │  COMEDY: Viết lại thành narrative
│     (LLM rewrite)     │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  3. EXTRACTION        │  A-Mem: Trích xuất Atomic Facts
│     (LLM + JSON)      │  với keywords, context_tags
└───────────┬───────────┘
            │
            ▼
Output: Crystal Facts (stored in Vector DB)
```

**Categories:**
- `personal_info`: Tên, tuổi, nghề nghiệp, quê quán
- `preference`: Sở thích, đồ ăn/uống yêu thích
- `fact`: Sự kiện khách quan
- `plan`: Kế hoạch, dự định tương lai
- `relationship`: Mối quan hệ với người khác
- `experience`: Trải nghiệm, kỷ niệm

### 1.3.4. Solid Layer

**Định nghĩa:** Lớp lưu trữ tần số thấp (t=Session), chứa User Profile bền vững.

**Đặc điểm:**
| Property | Value |
|----------|-------|
| Frequency | t=End of Session hoặc M messages |
| Content | Consolidated User Profile |
| Metadata | version, supersedes, validity_start, change_type |
| Processing | MAPLE Evolver với Reflection |
| Purpose | Knowledge tin cậy nhất, ưu tiên retrieval |

**Evolver Pipeline:**
```
Input: Crystal Facts (NEW) + Solid Knowledge (OLD)
        │
        ▼
┌───────────────────────┐
│  1. COMPARISON        │  So sánh NEW vs OLD
│                       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  2. REFLECTION        │  MAPLE: Phân tích WHY
│     (Root Cause)      │  - SUPPLEMENT: Bổ sung
│                       │  - REPLACEMENT: Thay thế
│                       │  - CORRECTION: Sửa lỗi
│                       │  - EVOLUTION: Tiến hóa
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  3. CONFLICT RESOLVE  │  Quyết định giữ/thay đổi
│                       │
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  4. VERSION ARCHIVE   │  Lưu version cũ với link
│                       │  supersedes ↔ superseded_by
└───────────┬───────────┘
            │
            ▼
Output: Updated Solid Profile + Archived History
```

### 1.3.5. Hybrid Retriever

**Định nghĩa:** Module tìm kiếm thông minh với nhiều chiến lược.

**Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `hybrid` | Solid → Crystal → Liquid (smart fallback) | Default, general queries |
| `solid_first` | Chỉ Solid, fallback Crystal nếu score thấp | Profile queries |
| `all_layers` | Search tất cả cùng lúc | Comprehensive search |
| `recent` | Ưu tiên Liquid (context gần đây) | "Câu hỏi trước là gì?" |

**Score-based Fallback:**
```python
if solid_score >= threshold (0.7):
    return solid_results
elif crystal_score >= threshold:
    return solid + crystal
else:
    return solid + crystal + liquid
```

---

## 1.4. Phân tích lý thuyết

### 1.4.1. Ưu điểm của FCM

1. **Noise Reduction**: Crystallizer loại bỏ 60-70% noise (greetings, fillers)
2. **Structured Output**: JSON format dễ xử lý, query
3. **Conflict Aware**: Evolver phát hiện và giải quyết mâu thuẫn
4. **Traceable**: Version history cho phép audit
5. **Efficient Retrieval**: Hybrid search giảm latency

### 1.4.2. Hạn chế

1. **LLM Dependency**: Cần LLM cho Crystallizer và Evolver
2. **Latency**: Crystallize/Evolve mất thời gian
3. **Cost**: Nhiều LLM calls nếu dùng paid API
4. **Error Propagation**: LLM extraction sai → propagate lên layers cao hơn

### 1.4.3. So sánh với kiến trúc truyền thống

| Aspect | Traditional (STM/LTM) | FCM |
|--------|----------------------|-----|
| Phân loại | Không gian (ngắn/dài hạn) | Tần số (High/Mid/Low) |
| Transition | Manual/Time-based | LLM-driven + Topic Shift |
| Noise Handling | Không xử lý | Crystallizer loại bỏ |
| Structure | Unstructured | JSON Atomic Facts |
| Conflict | Không xử lý | Evolver với Reflection |
| Version | Không track | Linked List History |
| Retrieval | Single layer | Hybrid multi-layer |

---

# PHẦN 2: TRIỂN KHAI CODE

## 2.1. Cấu trúc Module

```
fcm/
├── __init__.py          # Package exports, version
├── config.py            # FCMConfig dataclass
├── prompts.py           # LLM prompts (Crystallizer, Evolver, etc.)
├── agent.py             # FCMAgent - main implementation
└── utils.py             # Utility functions
```

## 2.2. Mapping Lý thuyết → Code

### 2.2.1. Config (`config.py`)

```python
@dataclass
class FCMConfig:
    # === Layer Types (Nested Learning) ===
    liquid_type: str = "liquid"      # High-frequency
    crystal_type: str = "crystal"    # Mid-frequency
    solid_type: str = "solid"        # Low-frequency
    
    # === Thresholds ===
    crystallize_threshold: int = 5   # N messages → Crystallize
    evolve_threshold: int = 20       # M messages → Evolve
    buffer_size: int = 5             # Conversation buffer
    
    # === Retrieval (InfLLM) ===
    solid_search_limit: int = 5
    crystal_search_limit: int = 5
    liquid_search_limit: int = 3
    hybrid_score_threshold: float = 0.7
    
    # === LLM Provider ===
    llm_provider: str = "groq"       # groq/openai/gemini
    llm_model: str = "llama-3.1-8b-instant"
    
    # === Embedder ===
    embedder_provider: str = "huggingface"
    embedder_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dims: int = 768
    
    # === Vector Store ===
    vector_store_provider: str = "qdrant"
    qdrant_path: str = "./fcm_qdrant"
    collection_name: str = "fcm_memory"
```

### 2.2.2. Prompts (`prompts.py`)

**Topic Shift Detection (SeCom):**
```python
TOPIC_SHIFT_DETECTION_PROMPT = """
Phân tích xem tin nhắn MỚI có thay đổi chủ đề so với CONTEXT không.

CONTEXT (các tin nhắn trước):
{context}

TIN NHẮN MỚI:
{new_message}

Output JSON:
{
    "topic_shifted": true/false,
    "old_topic": "chủ đề cũ",
    "new_topic": "chủ đề mới",
    "confidence": 0.0-1.0
}
"""
```

**Compression (COMEDY):**
```python
CONVERSATION_COMPRESSION_PROMPT = """
Viết lại đoạn hội thoại thành văn xuôi ngắn gọn.

LOẠI BỎ: Greetings, fillers, câu không có thông tin
GIỮ LẠI: Facts, preferences, plans, relationships

Output: Đoạn văn 2-5 câu tóm tắt thông tin quan trọng.
"""
```

**Crystallizer (A-Mem Zettelkasten):**
```python
CRYSTALLIZER_SYSTEM_PROMPT = """
Trích xuất Atomic Facts theo phương pháp Zettelkasten.

Mỗi fact phải:
1. ĐỘC LẬP (Self-contained)
2. CỤ THỂ (Specific)
3. NGẮN GỌN (Concise)

Output JSON với: content, category, keywords, context_tags, related_to, confidence
"""
```

**Evolver (MAPLE Archiver):**
```python
EVOLVER_SYSTEM_PROMPT = """
Memory Archivist với TRUY VẾT LỊCH SỬ và PHẢN CHIẾU.

PHÂN LOẠI THAY ĐỔI:
- SUPPLEMENT: Bổ sung
- REPLACEMENT: Thay thế
- CORRECTION: Sửa lỗi
- EVOLUTION: Tiến hóa

Output JSON với: reflection, updated_facts, archived_facts, user_profile_summary
"""
```

### 2.2.3. Agent (`agent.py`)

**Class FCMAgent:**
```python
class FCMAgent:
    def __init__(self, config, user_id):
        self.config = config
        self.user_id = user_id
        self.memory = Memory.from_config(mem0_config)
        self.llm_client = self._init_llm()
        self.conversation_buffer = []
        self.stats = FCMStats()
```

**Liquid Layer Methods:**
```python
def add_liquid_memory(self, content, role="user", detect_topic_shift=True):
    """
    [Nested Learning + SeCom]
    1. Detect Topic Shift (nếu enabled)
    2. Lưu vào Vector DB với metadata
    3. Update buffer và stats
    4. Return topic_shifted flag
    """
    
def get_liquid_memories(self, limit=10, status="raw"):
    """Lấy Liquid memories với filter"""
```

**Crystallizer Methods (SeCom + COMEDY + A-Mem):**
```python
def _detect_topic_shift(self, current_context, new_message):
    """[SeCom] LLM detect topic change"""
    
def _segment_conversation(self, messages):
    """[SeCom] Chia messages thành segments theo topic"""
    
def _compress_conversation(self, messages):
    """[COMEDY] Nén messages thành narrative"""
    
def crystallize(self, force=False):
    """
    Main Crystallizer Pipeline:
    1. Segment conversation (nếu buffer lớn)
    2. Compress mỗi segment
    3. Extract Atomic Facts từ compressed text
    4. Lưu vào Crystal layer với A-Mem metadata
    """
```

**Evolver Methods (MAPLE):**
```python
def evolve(self, force=False):
    """
    Main Evolver Pipeline:
    1. Lấy Crystal facts mới
    2. Lấy Solid knowledge cũ
    3. Gọi LLM để compare & merge
    4. Archive old versions
    5. Update Solid layer
    """
    
def _archive_solid_memory(self, memory_id, valid_until, superseded_by):
    """
    [MAPLE Version Tracking]
    1. Get old memory
    2. Delete from active
    3. Re-insert with status="archived" và [ARCHIVED] prefix
    """
    
def get_memory_history(self, content_query):
    """[MAPLE] Truy vết lịch sử thay đổi của một fact"""
```

**Retriever Methods (InfLLM):**
```python
def search(self, query, strategy="hybrid", limit=10):
    """
    Hybrid Retrieval với 4 strategies
    """
    
def _search_layer(self, query, fcm_type, limit, include_archived=False):
    """
    Search trong một layer với client-side filtering
    - Filter by fcm_type
    - Filter out archived (trừ khi include_archived=True)
    """
```

**High-level Methods:**
```python
def chat(self, user_message, auto_crystallize=True, return_context=False):
    """
    Main chat pipeline:
    1. Save to Liquid (với Topic Shift detection)
    2. Check trigger: Topic Shift OR Threshold
    3. Crystallize nếu triggered
    4. Return context nếu requested
    """
    
def end_session(self, auto_evolve=True):
    """
    Session cleanup:
    1. Force crystallize
    2. Force evolve
    3. Return final stats
    """
    
def get_user_profile(self):
    """Lấy User Profile từ Solid layer, organized by sections"""
```

## 2.3. Data Flow Chi tiết

### 2.3.1. Chat Flow

```python
# User gửi message
result = agent.chat("Tôi tên Nam, học HUST")

# Internal flow:
# 1. add_liquid_memory("Tôi tên Nam, học HUST")
#    → _detect_topic_shift() → topic_shifted=False
#    → memory.add() với metadata {fcm_type: "liquid", ...}
#    → buffer.append()
#
# 2. Check trigger
#    → topic_shifted? NO
#    → messages_since >= threshold? (1 >= 5)? NO
#    → crystallized = False
#
# 3. Return result
```

### 2.3.2. Crystallize Flow

```python
# Sau 5 messages hoặc Topic Shift
result = agent.crystallize()

# Internal flow:
# 1. Lấy liquid memories từ buffer
#
# 2. Nếu buffer > 10: _segment_conversation()
#    → LLM detect topic shifts
#    → Chia thành segments
#
# 3. Với mỗi segment: _compress_conversation()
#    → LLM viết lại thành narrative
#    → "Nam là sinh viên HUST, chuyên ngành CNTT, thích Python"
#
# 4. Extract facts từ compressed text
#    → LLM trả về JSON với facts
#    → [{content: "Nam là sinh viên HUST", keywords: [...], ...}]
#
# 5. Lưu mỗi fact vào Crystal layer
#    → memory.add() với metadata {fcm_type: "crystal", ...}
```

### 2.3.3. Evolve Flow

```python
# Cuối session
result = agent.evolve()

# Internal flow:
# 1. Lấy Crystal facts mới (chưa evolved)
#    → [{content: "Nam thích cà phê", ...}]
#
# 2. Lấy Solid knowledge cũ
#    → [{content: "Nam thích trà", ...}]  # Conflict!
#
# 3. Gọi LLM Evolver
#    → Compare: "thích trà" vs "thích cà phê"
#    → Reflection: EVOLUTION - "User đổi sở thích"
#    → Output: updated_facts, archived_facts
#
# 4. Archive old fact
#    → _archive_solid_memory("fact_001", ...)
#    → Delete "Nam thích trà"
#    → Insert "[ARCHIVED] Nam thích trà" với superseded_by
#
# 5. Add new solid
#    → memory.add("Nam thích cà phê") với supersedes link
```

---

# PHẦN 3: HƯỚNG DẪN CHẠY

## 3.1. Yêu cầu hệ thống

```
Python >= 3.9
pip (package manager)
~2GB disk space (for models and data)
```

## 3.2. Cài đặt

### 3.2.1. Clone và cài đặt

```bash
# Clone repository
git clone <repository-url>
cd mem0

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -e .

# Hoặc cài đặt từ requirements
pip install -r requirements.txt
```

### 3.2.2. Cài đặt thêm cho Qdrant

```bash
# Qdrant client (đã có trong requirements)
pip install qdrant-client
```

## 3.3. Cấu hình API Keys

### 3.3.1. Option 1: Groq (MIỄN PHÍ - Khuyến nghị)

**Bước 1:** Đăng ký tại https://console.groq.com

**Bước 2:** Tạo API Key

**Bước 3:** Set environment variable
```bash
# Linux/Mac
export GROQ_API_KEY="gsk_xxxxxxxxxxxx"

# Windows PowerShell
$env:GROQ_API_KEY = "gsk_xxxxxxxxxxxx"

# Windows CMD
set GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

**Bước 4:** Sử dụng trong code
```python
from fcm import FCMAgent, FCMConfig

config = FCMConfig(
    llm_provider="groq",
    llm_model="llama-3.1-8b-instant",  # Miễn phí, nhanh
)
agent = FCMAgent(config=config, user_id="my_user")
```

**Groq Models miễn phí:**
| Model | Speed | Quality |
|-------|-------|---------|
| llama-3.1-8b-instant | ⚡⚡⚡ Rất nhanh | ⭐⭐⭐ Tốt |
| llama-3.1-70b-versatile | ⚡⚡ Nhanh | ⭐⭐⭐⭐ Rất tốt |
| mixtral-8x7b-32768 | ⚡⚡ Nhanh | ⭐⭐⭐⭐ Rất tốt |

### 3.3.2. Option 2: OpenAI (Trả phí)

**Bước 1:** Đăng ký tại https://platform.openai.com

**Bước 2:** Tạo API Key

**Bước 3:** Set environment variable
```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxx"
```

**Bước 4:** Sử dụng trong code
```python
config = FCMConfig(
    llm_provider="openai",
    llm_model="gpt-4o-mini",  # Rẻ nhất
    # llm_model="gpt-4o",     # Chất lượng cao nhất
)
```

**OpenAI Pricing (tham khảo):**
| Model | Input | Output |
|-------|-------|--------|
| gpt-4o-mini | $0.15/1M tokens | $0.60/1M tokens |
| gpt-4o | $2.50/1M tokens | $10/1M tokens |

### 3.3.3. Option 3: Google Gemini (Free tier có giới hạn)

**Bước 1:** Đăng ký tại https://aistudio.google.com

**Bước 2:** Tạo API Key

**Bước 3:** Set environment variable
```bash
export GOOGLE_API_KEY="AIzaxxxxxx"
```

**Bước 4:** Sử dụng trong code
```python
config = FCMConfig(
    llm_provider="gemini",
    llm_model="gemini-1.5-flash",  # Nhanh, rẻ
    # llm_model="gemini-1.5-pro",  # Chất lượng cao
)
```

### 3.3.4. Option 4: Ollama (Local - Miễn phí hoàn toàn)

**Bước 1:** Cài đặt Ollama từ https://ollama.ai

**Bước 2:** Pull model
```bash
ollama pull llama3.1:8b
```

**Bước 3:** Sử dụng trong code
```python
config = FCMConfig(
    llm_provider="ollama",
    llm_model="llama3.1:8b",
    ollama_base_url="http://localhost:11434",  # Default
)
```

## 3.4. Chạy Demo

### 3.4.1. Quick Start

```python
# quick_start.py
import os
os.environ["GROQ_API_KEY"] = "your-key-here"

from fcm import FCMAgent, FCMConfig

# Khởi tạo
config = FCMConfig(
    llm_provider="groq",
    llm_model="llama-3.1-8b-instant",
    verbose=True,  # Xem logs
)
agent = FCMAgent(config=config, user_id="demo_user")

# Chat
agent.chat("Xin chào, tôi tên là Nam")
agent.chat("Tôi là sinh viên Bách Khoa Hà Nội")
agent.chat("Tôi học ngành Công nghệ thông tin")
agent.chat("Tôi thích lập trình Python")
agent.chat("Cuối tuần này tôi sẽ đi Đà Lạt")

# Crystallize (tự động trigger sau 5 messages)
# Hoặc force:
agent.crystallize(force=True)

# Search
results = agent.search("Nam học trường nào?")
print(results["combined"][0]["memory"])

# End session (auto evolve)
agent.end_session()

# Xem User Profile
profile = agent.get_user_profile()
print(profile)
```

### 3.4.2. Interactive Demo

```bash
python fcm_demo.py
```

**Menu:**
```
=== FCM Demo Menu ===
1. Basic Flow (Happy Path)
2. Conflict Resolution Demo
3. Search Strategies Demo
4. Interactive Chat
5. Exit
```

### 3.4.3. Demo Script Chi tiết

```python
# demo_full.py
import os
os.environ["GROQ_API_KEY"] = "your-key-here"

from fcm import FCMAgent, FCMConfig

def demo_basic_flow():
    """Demo luồng cơ bản"""
    print("=" * 50)
    print("DEMO 1: Basic Flow")
    print("=" * 50)
    
    config = FCMConfig(verbose=True)
    agent = FCMAgent(config=config, user_id="demo1")
    
    # Liquid Layer
    print("\n--- Adding to Liquid Layer ---")
    agent.chat("Tôi tên Nam, 22 tuổi")
    agent.chat("Tôi học HUST ngành CNTT")
    agent.chat("Tôi thích Python và AI")
    agent.chat("Sở thích của tôi là đọc sách")
    agent.chat("Cuối tuần tôi hay đi cafe")
    
    # Crystallize (triggered automatically)
    print("\n--- Crystal Layer ---")
    crystals = agent.get_crystal_memories(limit=10)
    for c in crystals:
        print(f"  - {c['memory']}")
    
    # Evolve
    print("\n--- Evolving to Solid Layer ---")
    agent.end_session()
    
    # User Profile
    print("\n--- User Profile ---")
    profile = agent.get_user_profile()
    for section, facts in profile.items():
        if facts:
            print(f"\n{section}:")
            for f in facts:
                print(f"  - {f}")

def demo_conflict_resolution():
    """Demo xử lý mâu thuẫn"""
    print("=" * 50)
    print("DEMO 2: Conflict Resolution")
    print("=" * 50)
    
    config = FCMConfig(verbose=True, crystallize_threshold=3)
    agent = FCMAgent(config=config, user_id="demo2")
    
    # Session 1: Thông tin ban đầu
    print("\n--- Session 1: Initial Info ---")
    agent.chat("Tôi thích uống trà")
    agent.chat("Trà ô long là loại tôi thích nhất")
    agent.chat("Tôi uống trà mỗi sáng")
    agent.end_session()
    
    # Session 2: Thông tin mới (conflict)
    print("\n--- Session 2: Updated Info (Conflict!) ---")
    agent.chat("Gần đây tôi chuyển sang uống cà phê")
    agent.chat("Cà phê sữa đá là thức uống yêu thích mới")
    agent.chat("Tôi bỏ uống trà rồi")
    agent.end_session()
    
    # Check history
    print("\n--- Memory History ---")
    history = agent.get_memory_history("uống")
    for h in history:
        print(f"  [{h['metadata'].get('fcm_status', 'active')}] {h['memory']}")

def demo_search_strategies():
    """Demo các chiến lược search"""
    print("=" * 50)
    print("DEMO 3: Search Strategies")
    print("=" * 50)
    
    config = FCMConfig(verbose=True)
    agent = FCMAgent(config=config, user_id="demo3")
    
    # Add data
    agent.chat("Tôi tên Minh, 25 tuổi")
    agent.chat("Tôi làm kỹ sư phần mềm")
    agent.chat("Tôi thích chơi game và xem phim")
    agent.chat("Hôm qua tôi đi xem phim Avengers")
    agent.chat("Tôi đang học Machine Learning")
    agent.crystallize(force=True)
    agent.evolve(force=True)
    
    # Different search strategies
    query = "Minh làm nghề gì?"
    
    print(f"\n--- Query: '{query}' ---")
    
    for strategy in ["hybrid", "solid_first", "all_layers", "recent"]:
        results = agent.search(query, strategy=strategy)
        print(f"\n{strategy}:")
        print(f"  Best source: {results.get('best_source', 'N/A')}")
        if results["combined"]:
            print(f"  Top result: {results['combined'][0]['memory']}")

def demo_topic_shift():
    """Demo Topic Shift Detection"""
    print("=" * 50)
    print("DEMO 4: Topic Shift Auto-trigger")
    print("=" * 50)
    
    config = FCMConfig(
        verbose=True, 
        crystallize_threshold=10  # Cao để thấy Topic Shift trigger
    )
    agent = FCMAgent(config=config, user_id="demo4")
    
    # Segment 1: Personal info
    print("\n--- Segment 1: Personal Info ---")
    agent.chat("Tôi tên Hùng")
    agent.chat("Tôi 23 tuổi")
    
    # Topic Shift! → Should trigger crystallize
    print("\n--- Topic Shift: Weather ---")
    result = agent.chat("À mà hôm nay trời đẹp quá!")
    print(f"  Topic shifted: {result.get('topic_shifted', False)}")
    print(f"  Crystallize trigger: {result.get('crystallize_trigger', 'none')}")

if __name__ == "__main__":
    demo_basic_flow()
    print("\n" + "=" * 70 + "\n")
    demo_conflict_resolution()
    print("\n" + "=" * 70 + "\n")
    demo_search_strategies()
    print("\n" + "=" * 70 + "\n")
    demo_topic_shift()
```

## 3.5. Troubleshooting

### 3.5.1. Lỗi API Key

```
Error: No API key found
```

**Giải pháp:**
```python
import os
os.environ["GROQ_API_KEY"] = "your-key"  # Trước khi import FCM
```

### 3.5.2. Lỗi Embedding Dimension

```
Error: Embedding dimension mismatch
```

**Giải pháp:**
```python
config = FCMConfig(
    embedding_dims=768,  # Phải khớp với model
    # all-mpnet-base-v2 = 768
    # all-MiniLM-L6-v2 = 384
)
```

### 3.5.3. Lỗi JSON Parse

```
Error: Failed to parse LLM response as JSON
```

**Giải pháp:**
- Bật `verbose=True` để xem raw response
- Thử model khác (llama-3.1-70b-versatile thường tốt hơn với JSON)
- Kiểm tra prompt có yêu cầu JSON rõ ràng không

### 3.5.4. Lỗi Qdrant

```
Error: Collection not found
```

**Giải pháp:**
```python
config = FCMConfig(
    qdrant_path="./fcm_qdrant",  # Đường dẫn tồn tại
    collection_name="fcm_memory",
)
```

### 3.5.5. Rate Limit (Groq)

```
Error: Rate limit exceeded
```

**Giải pháp:**
- Groq free tier: 30 requests/minute
- Thêm delay giữa các requests
- Hoặc nâng cấp plan

## 3.6. Best Practices

### 3.6.1. Cho Demo/Presentation

```python
config = FCMConfig(
    crystallize_threshold=3,  # Thấp để thấy effect nhanh
    verbose=True,             # Xem logs
    llm_provider="groq",      # Miễn phí, nhanh
)
```

### 3.6.2. Cho Production

```python
config = FCMConfig(
    crystallize_threshold=10,
    evolve_threshold=50,
    verbose=False,
    llm_provider="openai",
    llm_model="gpt-4o-mini",
)
```

### 3.6.3. Cho Testing

```python
config = FCMConfig(
    crystallize_threshold=2,
    evolve_threshold=5,
    verbose=True,
    llm_provider="ollama",  # Local, không tốn tiền
)
```

---

## Phụ lục

### A. Mapping Table: Paper → Code

| Paper Concept | FCM Module | File | Function/Class |
|---------------|------------|------|----------------|
| Nested Learning Layers | 3 Layer Architecture | config.py | liquid_type, crystal_type, solid_type |
| SeCom Segmentation | Topic Shift Detection | agent.py | `_detect_topic_shift()` |
| SeCom Segmentation | Conversation Segmentation | agent.py | `_segment_conversation()` |
| COMEDY Compression | Compressive Memory | agent.py | `_compress_conversation()` |
| A-Mem Atomic Notes | Crystallizer | agent.py | `crystallize()` |
| A-Mem Zettelkasten | Keyword Linking | prompts.py | CRYSTALLIZER_SYSTEM_PROMPT |
| MAPLE Archiver | Evolver | agent.py | `evolve()` |
| MAPLE Version Track | Archive System | agent.py | `_archive_solid_memory()` |
| MAPLE Reflection | Reasoning | prompts.py | EVOLVER_SYSTEM_PROMPT |
| InfLLM Retrieval | Hybrid Search | agent.py | `search()`, `_search_layer()` |

### B. Glossary

| Term | Definition |
|------|------------|
| **Liquid** | High-frequency layer, raw message storage |
| **Crystal** | Mid-frequency layer, atomic facts |
| **Solid** | Low-frequency layer, user profile |
| **Crystallize** | Process: Liquid → Crystal |
| **Evolve** | Process: Crystal → Solid |
| **Topic Shift** | Change of conversation topic |
| **Atomic Fact** | Self-contained unit of information |
| **Zettelkasten** | Note-taking method with linking |
| **Supersedes** | New fact replaces old fact |

### C. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12 | Initial release |
| 0.2.0 | 2024-12 | Added SeCom, COMEDY, MAPLE |
| 0.2.1 | 2024-12 | A-Mem Zettelkasten linking |
| 0.2.2 | 2024-12 | Fixed archive logic, search filtering |
| 0.2.3 | 2024-12 | Topic Shift auto-trigger |

---

## License

MIT License - Free to use for educational purposes.

## References

1. Nested Learning / CMS Framework
2. SeCom: Semantic Communication for Dialogue
3. COMEDY: Compact Memory for Dialogue
4. A-Mem: Agentic Memory with Zettelkasten
5. MAPLE: Modular Architecture for Persistent Learning
6. InfLLM: Training-Free Long-Context LLMs
7. Mem0: Memory Layer for AI Agents

---

# PHẦN 4: ĐÁNH GIÁ TRÊN TẬP LOCOMO

## 4.1. Giới thiệu tập locomo

Tập dữ liệu **locomo10.json** và **locomo10_rag.json** (trong thư mục `dataset/`) là bộ câu hỏi kiểm tra khả năng ghi nhớ, truy xuất và tổng hợp thông tin của agent. Mỗi mục gồm:
- `context`: Đoạn hội thoại hoặc thông tin nền
- `question`: Câu hỏi kiểm tra
- `answer`: Đáp án chuẩn

## 4.2. Quy trình đánh giá

1. **Khởi tạo agent** với cấu hình phù hợp (ưu tiên Groq hoặc OpenAI, bật `verbose=True` để xem log).
2. **Nạp context**: Dùng `agent.chat()` để đưa từng câu trong `context` vào agent (mỗi câu là một lượt chat).
3. **Crystallize/Evolve**: Đảm bảo agent đã thực hiện crystallize và evolve (có thể gọi thủ công nếu cần).
4. **Truy vấn**: Dùng `agent.search(question)` để lấy câu trả lời.
5. **So sánh kết quả**: So sánh kết quả trả về với trường `answer`.
6. **Ghi nhận điểm**: Tính điểm chính xác (accuracy) hoặc các chỉ số khác (F1, recall, v.v.).

## 4.3. Script đánh giá mẫu

```python
import json
from fcm import FCMAgent, FCMConfig

# Load dataset
with open("dataset/locomo10.json", encoding="utf-8") as f:
    data = json.load(f)

config = FCMConfig(verbose=True)
agent = FCMAgent(config=config, user_id="eval")

correct = 0
for i, item in enumerate(data):
    print(f"\n=== Sample {i+1} ===")
    # Reset agent/session nếu cần
    agent.end_session()
    # Nạp context
    for msg in item["context"]:
        agent.chat(msg)
    agent.crystallize(force=True)
    agent.evolve(force=True)
    # Truy vấn
    result = agent.search(item["question"])
    answer = result["combined"][0]["memory"] if result["combined"] else ""
    print(f"Q: {item['question']}")
    print(f"Predicted: {answer}")
    print(f"Ground Truth: {item['answer']}")
    if item["answer"].strip().lower() in answer.strip().lower():
        correct += 1

print(f"\nAccuracy: {correct}/{len(data)} = {correct/len(data):.2%}")
```

## 4.4. Lưu ý khi đánh giá

- Có thể cần tinh chỉnh so khớp đáp án (normalize, lower, loại bỏ dấu câu).
- Đánh giá thêm các chỉ số như recall, precision nếu cần.
- Có thể thử nghiệm với các cấu hình khác nhau (model, threshold, v.v.).

---

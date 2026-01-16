# Frequency-based Crystallizing Memory (FCM) V2

## A Hierarchical Memory Architecture for AI Agents

**Version:** 2.0.0  
**Authors:** Project III - HUST 2024-2025  
**Based on:** Nested Learning, SeCom, A-Mem, MAPLE, InfLLM, **Zep, MemLoRA, Ebbinghaus**

---

# 🚀 PHẦN MỚI: FCM V2 - 5 CẢI TIẾN

## Tổng quan các cải tiến

FCM V2 bổ sung 5 cải tiến quan trọng dựa trên các nghiên cứu mới nhất về memory systems:

| # | Cải tiến | Lý thuyết | Lợi ích |
|---|----------|-----------|---------|
| 1 | **Bi-Temporal Schema** | Temporal databases | Trả lời câu hỏi về quá khứ chính xác hơn |
| 2 | **Attention Sinks** | InfLLM, Zep | Giữ context ban đầu, tiết kiệm LLM calls |
| 3 | **Active Forgetting** | Ebbinghaus Curve | Tự động dọn dẹp memories yếu |
| 4 | **Dynamic Persona** | MemLoRA | Bot thích nghi với phong cách user |
| 5 | **Weighted Retrieval** | Ensemble Learning | Kết quả tìm kiếm chính xác hơn |

---

## Cải tiến 1: Bi-Temporal Schema (Crystal Layer)

### Vấn đề
Hệ thống cũ chỉ lưu `created_at` - thời điểm ghi nhận. Khi user nói "Năm 2018 tôi làm ở Google", không phân biệt được đây là sự kiện quá khứ.

### Giải pháp
Thêm 2 trường thời gian:
- **valid_at**: Thời gian sự kiện XẢY RA (trích xuất từ text)
- **observed_at**: Thời gian hệ thống GHI NHẬN

### Schema mới

```json
{
  "content": "User worked at Google",
  "valid_at": "2018-2022",
  "observed_at": "2024-01-15",
  "confidence": 0.95,
  "decay_score": 1.0,
  "metadata": {
    "category": "experience",
    "keywords": ["Google", "work"],
    "context_tags": ["career"]
  }
}
```

### Sử dụng

```python
from fcm_v2 import FCMAgentV2

agent = FCMAgentV2(user_id="user1")

# Add message với temporal info
agent.chat("Năm 2018 tôi làm ở Google")

# Search với temporal context
result = agent.search("Tôi làm gì năm 2018?", temporal_context="2018")
# → Ưu tiên facts có valid_at="2018"
```

---

## Cải tiến 2: Attention Sinks & Semantic Grouping (Liquid Layer)

### Vấn đề
1. Buffer trôi đi làm mất context ban đầu (tên user, vai trò)
2. SeCom gọi LLM liên tục để check topic shift → tốn kém

### Giải pháp

#### Attention Sinks
Luôn giữ lại **K tin nhắn đầu tiên** của phiên (System prompt + Greeting):

```
┌─────────────────────────────────────────────────────────────────┐
│                        LIQUID BUFFER                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 🔒 ATTENTION SINKS (không trôi đi)                        │  │
│  │   [1] "Tôi là Nam"                                        │  │
│  │   [2] "Tôi cần hỗ trợ về Python"                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 📝 BUFFER (trôi bình thường)                              │  │
│  │   [3] "Tôi đang học ML"                                   │  │
│  │   [4] "Tensorflow hay Pytorch?"                           │  │
│  │   [5] ...                                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

#### Semantic Grouping
Trước khi gọi LLM check topic shift, tính **cosine similarity** giữa embedding:

```python
# Tính similarity trước
similarity = cosine_similarity(new_message_embedding, buffer_avg_embedding)

if similarity >= 0.6:  # threshold
    # Cùng topic → SKIP LLM call
    topic_shifted = False
else:
    # Khác topic → Gọi LLM xác nhận
    topic_shifted = llm_detect_topic_shift(...)
```

### Config

```python
from fcm_v2 import FCMConfigV2

config = FCMConfigV2(
    attention_sink_count=3,  # Giữ 3 tin nhắn đầu tiên
    topic_shift_embedding_threshold=0.6,  # Ngưỡng similarity
)
```

---

## Cải tiến 3: Active Forgetting (Solid Layer)

### Vấn đề
Vector DB tích lũy rác theo thời gian - memories không còn relevant nhưng vẫn chiếm không gian và ảnh hưởng search.

### Giải pháp: Ebbinghaus Forgetting Curve

Công thức tính sức mạnh ký ức:

$$S(t) = S_0 \cdot e^{-\frac{\Delta t}{\tau}} + R \cdot N_{access}$$

Trong đó:
- $S_0 = 1.0$: Sức mạnh ban đầu
- $\Delta t$: Thời gian trôi qua (ngày)
- $\tau = 7$: Hằng số suy giảm (ngày)
- $R = 0.1$: Hệ số củng cố khi được truy xuất
- $N_{access}$: Số lần truy xuất

### Logic

```
Nếu memory được RETRIEVE:
    → Reset S = 1.0 (củng cố ký ức)
    → access_count += 1

Khi END SESSION:
    → Tính S cho tất cả solid memories
    → Nếu S < 0.2 → Move to Cold Storage (file log)
```

### Ví dụ

```
Memory: "User thích ăn pizza"
Created: 30 ngày trước
Access count: 0

S(30) = 1.0 × e^(-30/7) + 0.1 × 0
      = 1.0 × 0.013 + 0
      = 0.013 < 0.2

→ PRUNE (move to cold storage)
```

### Config

```python
config = FCMConfigV2(
    enable_active_forgetting=True,
    decay_constant_days=7.0,  # τ
    reinforcement_factor=0.1,  # R
    forget_threshold=0.2,  # Ngưỡng quên
    cold_storage_path="./fcm_cold_storage",
)
```

---

## Cải tiến 4: Dynamic Persona (Solid Layer)

### Vấn đề
Bot không thay đổi thái độ dù đã tương tác lâu với user. Phản hồi luôn generic.

### Giải pháp
Trong quá trình `evolve()`, trích xuất thêm **interaction_style**:

```json
{
  "interaction_style": {
    "communication_style": "casual",
    "preferred_response_length": "brief",
    "humor_level": 0.7,
    "inferred_traits": ["curious", "tech-savvy"],
    "topics_of_interest": ["AI", "Python"]
  }
}
```

### Sử dụng

```python
agent = FCMAgentV2(user_id="user1")

# Chat nhiều lần...
agent.chat("Yo! Tui thích code Python nè :D")
agent.chat("Mấy cái AI này hay ghê!")

# End session → Extract persona
agent.end_session()

# Inject vào system prompt của phiên sau
persona_injection = agent.get_persona_prompt_injection()
# → "- User thích giao tiếp thoải mái, có thể xưng hô thân mật"
# → "- User thích câu trả lời ngắn gọn"
# → "- User thích đùa, có thể dùng humor"
```

### Schema

```python
class UserPersona(BaseModel):
    user_id: str
    communication_style: Literal["formal", "casual", "mixed"]
    preferred_response_length: Literal["brief", "detailed", "adaptive"]
    humor_level: float  # 0.0 = serious, 1.0 = playful
    familiarity_level: float  # Tăng theo thời gian
    inferred_traits: List[str]
    topics_of_interest: List[str]
```

---

## Cải tiến 5: Weighted Ensemble Retrieval

### Vấn đề
Fallback tuần tự (Solid → Crystal → Liquid) có thể bỏ sót thông tin từ layers khác.

### Giải pháp
Search **song song** trên cả 3 layers và áp dụng **trọng số**:

$$Score_{final} = (w_s \cdot S_{solid}) + (w_c \cdot S_{crystal}) + (w_l \cdot S_{liquid})$$

Mặc định: $w_s=0.5, w_c=0.3, w_l=0.2$

### Flow

```
Query: "Nam làm gì?"
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌─────────┐
   │  SOLID  │   │  CRYSTAL │   │  LIQUID │
   │ Search  │   │  Search  │   │  Search │
   └────┬────┘   └────┬─────┘   └────┬────┘
        │             │              │
        ▼             ▼              ▼
   Normalize     Normalize      Normalize
   (0-1 scale)   (0-1 scale)    (0-1 scale)
        │             │              │
        ▼             ▼              ▼
   × 0.5 weight  × 0.3 weight   × 0.2 weight
        │             │              │
        └──────────────┴──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  Merge &     │
              │  Re-rank     │
              └──────┬───────┘
                     │
                     ▼
              Final Results
```

### Config

```python
config = FCMConfigV2(
    retrieval_weight_solid=0.5,
    retrieval_weight_crystal=0.3,
    retrieval_weight_liquid=0.2,
    parallel_search=True,  # Search song song
)
```

---

## Cấu trúc Module V2

```
fcm_v2/
├── __init__.py              # Package exports
├── config.py                # FCMConfigV2 với tham số mới
├── prompts.py               # Prompts cải tiến (Bi-Temporal, Persona)
├── agent.py                 # FCMAgentV2 - main implementation
├── demo.py                  # Demo script
│
├── schemas/                 # Pydantic models
│   ├── __init__.py
│   └── base.py              # AtomicFact, MemoryStrength, UserPersona
│
├── liquid/                  # Liquid Layer
│   ├── __init__.py
│   └── layer.py             # Attention Sinks, Semantic Grouping
│
├── crystal/                 # Crystal Layer
│   ├── __init__.py
│   └── layer.py             # Bi-Temporal extraction
│
├── solid/                   # Solid Layer
│   ├── __init__.py
│   └── layer.py             # Active Forgetting, Dynamic Persona
│
└── retrieval/               # Retrieval System
    ├── __init__.py
    └── weighted_retriever.py  # Weighted Ensemble
```

---

## Quick Start V2

```python
import os
os.environ["GROQ_API_KEY"] = "gsk_xxxxx"

from fcm_v2 import FCMAgentV2, FCMConfigV2

# Config với các cải tiến
config = FCMConfigV2(
    verbose=True,
    # Cải tiến 2: Attention Sinks
    attention_sink_count=3,
    topic_shift_embedding_threshold=0.6,
    # Cải tiến 3: Active Forgetting
    enable_active_forgetting=True,
    decay_constant_days=7.0,
    forget_threshold=0.2,
    # Cải tiến 4: Dynamic Persona
    enable_dynamic_persona=True,
    # Cải tiến 5: Weighted Retrieval
    retrieval_weight_solid=0.5,
    retrieval_weight_crystal=0.3,
    retrieval_weight_liquid=0.2,
)

agent = FCMAgentV2(config=config, user_id="demo")

# Chat
agent.chat("Xin chào, tôi là Nam")  # → Attention Sink
agent.chat("Năm 2018 tôi làm ở Google")  # → Bi-Temporal
agent.chat("Hiện tại tôi làm AI research")

# Search với Weighted Ensemble + Bi-Temporal boost
result = agent.search("Nam làm gì năm 2018?", temporal_context="2018")

# End session → Evolve + Prune + Extract Persona
agent.end_session()

# Get persona injection cho phiên sau
persona = agent.get_persona_prompt_injection()
print(persona)
```

---

## So sánh V1 vs V2

| Feature | V1 | V2 |
|---------|----|----|
| Temporal | Chỉ có `created_at` | Bi-Temporal (`valid_at` + `observed_at`) |
| Buffer | Trôi hoàn toàn | Attention Sinks giữ K đầu |
| Topic Shift | Luôn gọi LLM | Embedding similarity trước |
| Memory Cleanup | Manual | Active Forgetting tự động |
| Persona | Không có | Dynamic Persona extraction |
| Retrieval | Fallback tuần tự | Weighted Ensemble song song |
| Cost | Cao (nhiều LLM calls) | Tối ưu (Semantic Grouping) |

---

## Chạy Demo

```bash
cd fcm_v2
python demo.py
```

Menu:
1. Basic Flow (Attention Sinks, Crystallize)
2. Bi-Temporal Search
3. Weighted Retrieval
4. Active Forgetting
5. Dynamic Persona

---

# PHẦN GỐC: LÝ THUYẾT KIẾN TRÚC FCM V1

(Giữ nguyên nội dung gốc từ đây trở xuống...)

---

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

### 1.1.3. Đóng góp chính (V1)

1. **Three-Layer Architecture**: Phân tầng bộ nhớ theo tần số cập nhật
2. **LLM-driven Extraction**: Sử dụng LLM để trích xuất và lọc thông tin
3. **Semantic Segmentation**: Phân đoạn hội thoại dựa trên ngữ nghĩa (Topic Shift)
4. **Compressive Memory**: Nén hội thoại trước khi trích xuất facts
5. **Version Tracking**: Theo dõi lịch sử thay đổi của knowledge
6. **Hybrid Retrieval**: Tìm kiếm thông minh ưu tiên theo độ tin cậy

### 1.1.4. Đóng góp bổ sung (V2)

7. **Bi-Temporal Schema**: Phân biệt thời gian sự kiện vs thời gian ghi nhận
8. **Attention Sinks**: Giữ context ban đầu quan trọng
9. **Active Forgetting**: Tự động dọn dẹp memories yếu theo Ebbinghaus
10. **Dynamic Persona**: Thích nghi phong cách với user
11. **Weighted Retrieval**: Search ensemble với trọng số

---

## Phụ lục

### A. Công thức toán học

#### Active Forgetting (Ebbinghaus)

$$S(t) = S_0 \cdot e^{-\frac{\Delta t}{\tau}} + R \cdot N_{access}$$

| Parameter | Description | Default |
|-----------|-------------|---------|
| $S_0$ | Sức mạnh ban đầu | 1.0 |
| $\tau$ | Hằng số suy giảm (ngày) | 7.0 |
| $R$ | Hệ số củng cố | 0.1 |
| $N_{access}$ | Số lần truy xuất | 0 |

#### Weighted Retrieval

$$Score_{final} = (w_s \cdot S_{solid}) + (w_c \cdot S_{crystal}) + (w_l \cdot S_{liquid})$$

| Layer | Weight | Reason |
|-------|--------|--------|
| Solid | 0.5 | Knowledge đã kiểm chứng |
| Crystal | 0.3 | Facts đã trích xuất |
| Liquid | 0.2 | Context gần đây |

### B. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2024-12 | Initial release |
| 0.2.0 | 2024-12 | Added SeCom, COMEDY, MAPLE |
| 0.2.1 | 2024-12 | A-Mem Zettelkasten linking |
| 0.2.2 | 2024-12 | Fixed archive logic, search filtering |
| 0.2.3 | 2024-12 | Topic Shift auto-trigger |
| **2.0.0** | **2025-01** | **5 cải tiến: Bi-Temporal, Attention Sinks, Active Forgetting, Dynamic Persona, Weighted Retrieval** |

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
8. **Zep: Long-Term Memory for AI Assistants** (V2)
9. **MemLoRA: Memory-Efficient Personalization** (V2)
10. **Ebbinghaus Forgetting Curve** (V2)

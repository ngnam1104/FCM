# 🧠 FCM - Frequency-based Crystallizing Memory

> **A Hierarchical Memory Architecture for AI Agents**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Giới thiệu

FCM (Frequency-based Crystallizing Memory) là kiến trúc bộ nhớ phân tầng theo **tần số cập nhật**, mô phỏng quá trình hình thành ký ức của con người:

```
┌─────────────────────────────────────────────────────────────┐
│                    HUMAN MEMORY ANALOGY                     │
├─────────────────────────────────────────────────────────────┤
│   Sensory Input  →   Working Memory   →    Long-term        │
│         ↓                  ↓                   ↓             │
│      LIQUID            CRYSTAL              SOLID            │
│   (High-Freq)         (Mid-Freq)         (Low-Freq)          │
└─────────────────────────────────────────────────────────────┘
```

### 🎯 Vấn đề giải quyết

| Vấn đề | Kiến trúc truyền thống (STM/LTM) | FCM |
|--------|----------------------------------|-----|
| Information Overload | STM tràn nhanh | Crystallizer lọc noise |
| Noise Accumulation | LTM lưu cả rác | Compression + Extraction |
| Retrieval Confusion | Không phân biệt độ tin cậy | Hybrid multi-layer search |
| Conflict Blind | Không xử lý mâu thuẫn | Evolver với Reflection |

---

## 🏗️ Kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                      FCM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   USER MESSAGE                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────┐                                               │
│   │   LIQUID    │  Raw messages, Topic Shift Detection          │
│   │  (t = 1)    │                                               │
│   └──────┬──────┘                                               │
│          │  Trigger: N messages OR Topic Shift                  │
│          ▼                                                      │
│   ┌─────────────┐                                               │
│   │ CRYSTALLIZER│  Segment → Compress → Extract Atomic Facts    │
│   └──────┬──────┘                                               │
│          ▼                                                      │
│   ┌─────────────┐                                               │
│   │   CRYSTAL   │  Atomic Facts với keywords, context_tags      │
│   │  (t = N)    │                                               │
│   └──────┬──────┘                                               │
│          │  Trigger: End Session OR M messages                  │
│          ▼                                                      │
│   ┌─────────────┐                                               │
│   │   EVOLVER   │  Compare → Reflect → Resolve → Archive        │
│   └──────┬──────┘                                               │
│          ▼                                                      │
│   ┌─────────────┐                                               │
│   │    SOLID    │  User Profile + Version History               │
│   │ (t = Session)│                                              │
│   └─────────────┘                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 📚 Dựa trên các công trình

| Paper | Ý tưởng | Áp dụng trong FCM |
|-------|---------|-------------------|
| **Nested Learning** | Learning as Compression | 3-layer architecture |
| **SeCom** | Semantic Segmentation | Topic Shift Detection |
| **COMEDY** | Compressive Memory | Conversation Compression |
| **A-Mem** | Zettelkasten Atomic Notes | Keywords, context_tags linking |
| **MAPLE** | Archiver + Version Tracking | Evolver với Reflection |
| **InfLLM** | Hierarchical Retrieval | Hybrid multi-layer search |

---

## 🚀 Quick Start

### 1. Cài đặt

```bash
# Clone repository
git clone <repository-url>
cd Frequency-based-Crystallizing-Memory

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình API Key

```bash
# Option 1: Groq (MIỄN PHÍ - Khuyến nghị)
# Đăng ký tại: https://console.groq.com
export GROQ_API_KEY="gsk_xxxxxxxxxxxx"

# Option 2: OpenAI (Trả phí)
export OPENAI_API_KEY="sk-xxxxxxxxxxxx"

# Option 3: Google Gemini (Free tier có giới hạn)
export GOOGLE_API_KEY="AIzaxxxxxx"
```

### 3. Chạy Demo

```python
import os
os.environ["GROQ_API_KEY"] = "your-key-here"

from fcm import FCMAgent, FCMConfig

# Khởi tạo
config = FCMConfig(verbose=True)
agent = FCMAgent(config=config, user_id="demo_user")

# Chat
agent.chat("Xin chào, tôi tên là Nam")
agent.chat("Tôi là sinh viên Bách Khoa Hà Nội")
agent.chat("Tôi học ngành Công nghệ thông tin")
agent.chat("Tôi thích lập trình Python")
agent.chat("Cuối tuần này tôi sẽ đi Đà Lạt")

# Search
results = agent.search("Nam học trường nào?")
print(results["combined"][0]["memory"])

# Kết thúc session (auto crystallize + evolve)
agent.end_session()

# Xem User Profile
profile = agent.get_user_profile()
print(profile)
```

### 4. Chạy Demo Script

```bash
python fcm_eval/fcm_demo.py
```

**Menu:**
```
1. Basic Flow (Happy Path)
2. Conflict Resolution
3. Search Strategies
4. Interactive Chat
5. Topic Shift (SeCom)
```

---

## 📁 Cấu trúc Project

```
📂 Frequency-based Crystallizing Memory/
├── 📂 fcm/                     # Core FCM implementation
│   ├── __init__.py             # Package exports
│   ├── agent.py                # FCMAgent - main class
│   ├── config.py               # FCMConfig dataclass
│   ├── prompts.py              # LLM prompts
│   ├── utils.py                # Utility functions
│   └── FCM_README.md           # Documentation chi tiết
├── 📂 fcm_eval/                # Evaluation & demo scripts
│   ├── fcm_demo.py             # Interactive demo
│   ├── fcm_locomo.py           # LoCoMo benchmark
│   └── fcm_quick_start.py      # Quick start script
├── 📂 fcm_data/                # Vector store data
├── 📂 mem0/                    # Backend memory library
├── 📂 dataset/                 # Evaluation datasets
├── 📂 eval_results/            # Evaluation results
├── 📄 config_free_llm.py       # LLM config helpers
├── 📄 locomo_evaluation.ipynb  # Evaluation notebook
├── 📄 requirements.txt         # Dependencies
└── 📄 pyproject.toml           # Package config
```

---

## ⚙️ Cấu hình

```python
from fcm import FCMConfig

config = FCMConfig(
    # === Layer Thresholds ===
    crystallize_threshold=5,      # N messages → Crystallize
    evolve_threshold=20,          # M messages → Evolve
    buffer_size=5,                # Conversation buffer
    
    # === Retrieval ===
    hybrid_score_threshold=0.7,   # Score để fallback
    
    # === LLM Provider ===
    llm_provider="groq",          # groq/openai/gemini/ollama
    llm_model="llama-3.1-8b-instant",
    
    # === Embedder ===
    embedder_provider="huggingface",
    embedder_model="sentence-transformers/all-mpnet-base-v2",
    embedding_dims=768,
    
    # === Vector Store ===
    vector_store_provider="qdrant",
    qdrant_path="./fcm_data",
    
    # === Debug ===
    verbose=True,
)
```

---

## 🔍 API Reference

### FCMAgent

```python
agent = FCMAgent(config=config, user_id="user_123")

# Chat & Memory
agent.chat(message, auto_crystallize=True)    # Thêm message
agent.crystallize(force=False)                 # Liquid → Crystal
agent.evolve(force=False)                      # Crystal → Solid
agent.end_session()                            # Kết thúc session

# Search
agent.search(query, strategy="hybrid")         # Tìm kiếm
# Strategies: "hybrid", "solid_first", "all_layers", "recent"

# Profile & Stats
agent.get_user_profile()                       # Lấy user profile
agent.get_stats()                              # Thống kê
agent.get_memory_history(query)                # Lịch sử thay đổi
```

---

## 📊 Evaluation

### LoCoMo Benchmark

```bash
python fcm_eval/fcm_locomo.py
```

### Jupyter Notebook

```bash
jupyter notebook locomo_evaluation.ipynb
```

---

## 🛠️ Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `No API key found` | Set `os.environ["GROQ_API_KEY"]` trước khi import |
| `Embedding dimension mismatch` | Đảm bảo `embedding_dims` khớp với model (768 cho all-mpnet-base-v2) |
| `Rate limit exceeded` | Groq free tier: 30 req/min. Thêm delay hoặc nâng cấp |
| `Collection not found` | Kiểm tra `qdrant_path` tồn tại |

---

## 📖 Documentation

Xem chi tiết tại [fcm/FCM_README.md](fcm/FCM_README.md):
- Lý thuyết kiến trúc
- Các công trình liên quan
- Code implementation
- Hướng dẫn chi tiết

---

## 📄 License

MIT License - Free to use for educational purposes.

---

## 🙏 Acknowledgments

- [Mem0](https://github.com/mem0ai/mem0) - Memory Layer for AI Agents
- [Qdrant](https://qdrant.tech/) - Vector Database
- [Groq](https://groq.com/) - Fast LLM Inference
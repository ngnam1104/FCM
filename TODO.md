# FCM Project - TODO & Changelog

## 📅 Last Updated: January 17, 2026

---

## ✅ Đã Hoàn Thành

### 🎯 Option 1: LLM Reader (Temporal Inference)
**Status:** ✅ Hoàn thành - 100% accuracy trên test cases

**Mục đích:** Giải quyết vấn đề LoCoMo benchmark yêu cầu suy luận thời gian (temporal inference) mà retrieval system không thể làm được.

**Files đã sửa:**
- `fcm_eval/locomo.py`

**Các thành phần mới:**
| Component | Mô tả |
|-----------|-------|
| `LLMReader` class | Component sử dụng LLM để trả lời câu hỏi từ retrieved context |
| `LLM_READER_PROMPT` | Prompt hướng dẫn LLM tính toán temporal dates |
| `check_answer_with_llm_reader()` | Hàm kiểm tra answer với LLM Reader |
| `_parse_date()` | Parse date string (e.g., "8 May, 2023") |
| `_extract_temporal_answer()` | Tính toán temporal references |
| `_dates_match()` | So sánh 2 date strings |

**Temporal patterns được hỗ trợ:**
- `yesterday` → session_date - 1 day
- `last week` → session_date - 7 days
- `last year` → session_date.year - 1
- `next month` → session_date.month + 1

**Kết quả test:**
```
Test Case 1: "When did Caroline go to LGBTQ support group?"
  Expected: "7 May 2023"
  Context: "went to LGBTQ support group yesterday" (session: 8 May 2023)
  Result: ✅ PASS (LLM computed: 7 May 2023)

Test Case 2: "When did Melanie paint a sunrise?"
  Expected: "2022"
  Context: "painted that lake sunrise last year" (session: May 2023)
  Result: ✅ PASS (LLM computed: 2022)

Test Case 3: "When is Melanie planning on going camping?"
  Expected: "June 2023"
  Context: "planning to go camping next month" (session: 25 May 2023)
  Result: ✅ PASS (LLM computed: June 2023)
```

---

### 🔮 Option 2: Computed Temporal Facts (Crystallization Enhancement)
**Status:** ✅ Implemented - Cần test với LLM (rate limit)

**Mục đích:** Lưu temporal facts với ngày tháng đã được tính toán sẵn trong quá trình crystallization.

**Files đã sửa:**
- `fcm_v2/prompts.py`
- `fcm_v2/crystal/layer.py`
- `fcm_v2/agent.py`

**Cải tiến Prompt (CRYSTALLIZER_SYSTEM_PROMPT_V2):**
- Thêm category mới: `temporal_event`
- Thêm field mới: `computed_date`
- Hướng dẫn LLM tính toán ngày cụ thể từ relative time + session_date

**Ví dụ:**
```
Input: "I went to LGBTQ support group yesterday"
Session Date: "8 May 2023"

Output:
{
  "content": "User went to LGBTQ support group on 7 May 2023",
  "category": "temporal_event",
  "valid_at": "yesterday",
  "computed_date": "7 May 2023"
}
```

**Cải tiến Crystal Layer:**
- `extract_facts_with_bitemporal()` nhận `session_date` parameter
- Trích xuất và lưu `computed_date` vào CrystalFact
- Log với computed date info

**Cải tiến Agent:**
- `crystallize()` method hỗ trợ `session_date` parameter
- Lấy messages từ attention sinks khi buffer empty (force=True)

---

### 📊 LoCoMo Benchmark Updates
**Status:** ✅ Hoàn thành

**Files đã sửa:**
- `fcm_eval/locomo.py`

**Cải tiến:**
1. **Session dates extraction:** Parse `session_X_date_time` từ LoCoMo dataset
2. **Evidence parsing:** Trích xuất session number từ evidence (e.g., "D1:3" → session_1)
3. **LLM Reader integration:** Tích hợp vào `run_locomo_v1()` và `run_locomo_v2()`
4. **New CLI arguments:**
   - `--no-llm-reader`: Disable LLM Reader
5. **LLM Reader statistics:** Hiển thị số lần sử dụng LLM inference

---

### 🔧 Bug Fixes
- ✅ Fixed Windows date format issue (`%-d` không hoạt động trên Windows)
- ✅ Fixed GROQ LLM initialization (sử dụng GroqLLM thay vì Memory.from_config)
- ✅ Fixed LiquidMessage object handling (Pydantic model vs dict)

---

### 📁 Files Mới
| File | Mô tả |
|------|-------|
| `test_options.py` | Test script cho Option 1 và Option 2 |
| `debug_locomo.py` | Debug script cho LoCoMo dataset analysis |

---

## 🔄 Công Việc Sắp Tới

### High Priority
- [ ] **Test Option 2 với LLM đầy đủ** - Chờ GROQ rate limit reset
- [ ] **Chạy full LoCoMo benchmark** với LLM Reader enabled
- [ ] **So sánh accuracy** trước và sau khi thêm LLM Reader

### Medium Priority
- [ ] **Optimize LLM calls** - Cache results, batch processing
- [ ] **Thêm fallback LLM providers** - OpenAI, Anthropic khi GROQ rate limit
- [ ] **Improve temporal pattern matching** - Thêm patterns: "2 days ago", "last Monday", etc.

### Low Priority
- [ ] **Documentation** - Viết README chi tiết cho LLM Reader
- [ ] **Unit tests** - Thêm unit tests cho LLMReader class
- [ ] **Metrics dashboard** - Visualize benchmark results

---

## 📈 Benchmark Results Summary

### Trước khi fix (0% accuracy)
```
V1: 0.0% accuracy (50/50 failed)
V2: 0.0% accuracy (50/50 failed)
Reason: Exact match metric + Temporal inference required
```

### Sau khi fix với LLM Reader
```
Option 1 Test: 3/3 (100%) ✅
Full benchmark: Pending (rate limit)
```

---

## 🔗 References

- **LoCoMo Paper:** Long Context Memory benchmark
- **GROQ API:** Rate limit 100K tokens/day (free tier)
- **Temporal Inference:** Session-based date computation

---

## 📝 Notes

### Rate Limit Issue
GROQ API có rate limit 100K tokens/day. Khi exhausted:
- LLM Reader sẽ fallback sang pattern matching
- Crystallization sẽ skip fact extraction

### Workarounds
1. Chờ rate limit reset (~24h)
2. Sử dụng API key khác
3. Upgrade lên GROQ Dev Tier

---

## 🚀 Quick Start

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run LoCoMo benchmark với LLM Reader
python -m fcm_eval.locomo

# Run benchmark không có LLM Reader
python -m fcm_eval.locomo --no-llm-reader

# Test Option 1 và Option 2
python test_options.py
```

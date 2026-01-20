"""
FCM Agent Demo UI - Enhanced Version
=====================================

Giao diện Streamlit cải tiến với:
- Chọn giữa FCM V1 và FCM V2
- Vùng ném nội dung (Bulk Import)
- Vùng chat chính
- Hiển thị Memory Layers
- Tránh lỗi .lock trên Windows

Chạy: streamlit run demoUI.py
"""

import gc
import logging
import sys
import os
import time
import atexit

# Tắt warning
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

import streamlit as st

# Thêm đường dẫn - FCM root folder (parent of demo/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Check API Key
if "api_checked" not in st.session_state:
    print("=" * 60)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"✅ GROQ_API_KEY found: {api_key[:20]}...")
        st.session_state["api_checked"] = True
    else:
        print("❌ GROQ_API_KEY NOT FOUND!")
        st.error("❌ GROQ_API_KEY NOT FOUND! Vui lòng kiểm tra file .env")
        st.stop()
    print("=" * 60)

# Cấu hình trang
st.set_page_config(
    page_title="FCM Agent Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .stChatMessage {padding: 1rem; border-radius: 10px; margin-bottom: 10px;}
    .version-badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .v1-badge {background-color: #ff6b6b; color: white;}
    .v2-badge {background-color: #4ecdc4; color: white;}
    .bulk-import-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def get_agent(version: str, user_id: str):
    """Initialize FCM Agent based on version"""
    if version == "V1":
        from fcm.v1 import FCMAgent, FCMConfig
        config = FCMConfig(
            crystallize_threshold=3,
            verbose=True
        )
        return FCMAgent(config=config, user_id=user_id)
    else:
        from fcm.v2 import FCMAgentV2, FCMConfigV2
        config = FCMConfigV2(
            crystallize_threshold=3,
            attention_sink_count=2,
            semantic_similarity_threshold=0.75,
            enable_active_forgetting=False,
            verbose=True
        )
        return FCMAgentV2(config=config, user_id=user_id)


def is_question(text: str) -> bool:
    """Detect if text is a question"""
    text = text.strip().lower()
    # Vietnamese question words
    question_words = ['ai', 'gì', 'nào', 'đâu', 'sao', 'bao nhiêu', 'khi nào', 'như thế nào', 
                      'tại sao', 'có phải', 'có không', 'làm sao', 'mấy', 'bao giờ', 'ở đâu',
                      'what', 'who', 'where', 'when', 'why', 'how', 'which', 'is', 'are', 'do', 'does']
    return text.endswith('?') or any(text.startswith(w) or f' {w} ' in text for w in question_words)


def generate_answer_from_context(query: str, results: list, agent) -> str:
    """Generate answer using Top-K RAG (Updated to match Locomo logic)"""
    if not results:
        return "Tôi chưa có thông tin về điều này trong bộ nhớ."
    
    # 1. Context Window Expansion: Lấy Top 5 kết quả tốt nhất
    context_parts = []
    # Dùng .get() an toàn cho cả Dict và Object
    for i, r in enumerate(results[:5]):
        if isinstance(r, dict):
            mem = r.get('memory', '')
        else:
            # Fallback nếu r là object Pydantic
            mem = getattr(r, 'memory', str(r))
            
        if mem:
            context_parts.append(f"- {mem}")
    
    context_text = "\n".join(context_parts)
    
    # 2. Gọi LLM (Dùng model 8b-instant để tránh Rate Limit và Fallback)
    try:
        from mem0.llms.groq import GroqLLM
        # CHUYỂN VỀ MODEL NHẸ HƠN ĐỂ TRÁNH LỖI RATE LIMIT -> TRÁNH FALLBACK TOP 1
        llm = GroqLLM(config={
            "model": "llama-3.1-8b-instant", 
            "temperature": 0.1,
            "api_key": os.getenv("GROQ_API_KEY")
        })
        
        # Prompt nâng cao giống Locomo
        prompt = f"""Bạn là trợ lý AI hữu ích. Dựa VÀO BỘ NHỚ ĐƯỢC CUNG CẤP bên dưới, hãy trả lời câu hỏi.

BỘ NHỚ (Context):
{context_text}

CÂU HỎI: {query}

YÊU CẦU:
1. Chỉ trả lời dựa trên thông tin trong bộ nhớ.
2. Nếu bộ nhớ không có thông tin, hãy nói "Tôi chưa có thông tin này".
3. Trả lời ngắn gọn, đi thẳng vào vấn đề.

TRẢ LỜI:"""
        
        response = llm.generate_response([{"role": "user", "content": prompt}])
        return response.strip()
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        # Fallback thông minh hơn: Trả về toàn bộ context tìm được thay vì chỉ dòng đầu
        return f"Tìm thấy thông tin liên quan:\n{context_text}"


def cleanup_agent():
    """Cleanup agent trước khi exit"""
    if "agent" in st.session_state and st.session_state.agent is not None:
        try:
            del st.session_state.agent
        except:
            pass
    gc.collect()


def reset_agent():
    """Reset agent và clear session state - Windows safe"""
    cleanup_agent()
    
    if "agent" in st.session_state:
        del st.session_state["agent"]
    
    # Đợi để Windows mở khóa file
    time.sleep(1.5)
    
    # Reset các biến khác
    st.session_state.messages = []
    st.session_state.initialized = False
    st.session_state.bulk_messages = []
    
    st.rerun()


def switch_version(new_version: str):
    """Switch giữa V1 và V2 an toàn"""
    if st.session_state.get("version") == new_version:
        return
    
    # Cleanup old agent
    cleanup_agent()
    time.sleep(1.0)
    
    # Reset state
    st.session_state.version = new_version
    st.session_state.agent = None
    st.session_state.initialized = False
    st.session_state.messages = []
    
    st.rerun()


# Register cleanup
atexit.register(cleanup_agent)


# ===================== SIDEBAR =====================
with st.sidebar:
    st.title("🧠 FCM Control Panel")
    st.markdown("---")
    
    # Version Selection
    st.subheader("📦 Chọn Phiên Bản")
    
    current_version = st.session_state.get("version", "V2")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        if st.button("🔴 V1", use_container_width=True, 
                    type="primary" if current_version == "V1" else "secondary"):
            switch_version("V1")
    with col_v2:
        if st.button("🟢 V2", use_container_width=True,
                    type="primary" if current_version == "V2" else "secondary"):
            switch_version("V2")
    
    version = st.session_state.get("version", "V2")
    
    # Version info
    if version == "V1":
        st.markdown('<div class="version-badge v1-badge">FCM V1 - Basic</div>', unsafe_allow_html=True)
        st.caption("Features: Liquid → Crystal → Solid")
    else:
        st.markdown('<div class="version-badge v2-badge">FCM V2 - Enhanced</div>', unsafe_allow_html=True)
        st.caption("Features: Attention Sinks, Semantic Grouping, Active Forgetting")
    
    st.markdown("---")
    
    # Initialize Agent
    user_id = f"demo_{version.lower()}_user"
    
    if "agent" not in st.session_state or st.session_state.agent is None:
        try:
            with st.spinner(f"Initializing FCM {version}..."):
                st.session_state.agent = get_agent(version, user_id)
                st.session_state.messages = []
                st.session_state.bulk_messages = []
                st.session_state.initialized = True
            st.success("✅ Khởi tạo thành công!")
        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo: {str(e)}")
            st.code(str(e))
            st.stop()
    
    # Actions
    st.subheader("🎮 Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💎 Crystallize", use_container_width=True):
            with st.spinner("Crystallizing..."):
                res = st.session_state.agent.crystallize(force=True)
                st.success(f"Done! {res.get('facts_extracted', 0)} facts")
    
    with col2:
        if st.button("🧬 Evolve", use_container_width=True):
            with st.spinner("Evolving..."):
                res = st.session_state.agent.evolve(force=True)
                st.success("Done!")
    
    if st.button("🗑️ Reset All", use_container_width=True, type="secondary"):
        if st.session_state.agent:
            st.session_state.agent.clear_all_memories()
        reset_agent()
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    if st.session_state.get("agent"):
        stats = st.session_state.agent.get_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", stats.get('total_messages', 0))
            st.metric("Liquid", stats.get('liquid_count', 0))
        with col2:
            st.metric("Crystal", stats.get('crystal_count', 0))
            st.metric("Solid", stats.get('solid_count', 0))
        
        if version == "V2":
            st.markdown("---")
            st.caption("V2 Metrics:")
            additional = stats.get('additional', {})
            if additional:
                st.caption(f"• Attention Sinks: {additional.get('attention_sinks', 0)}")
                st.caption(f"• LLM Calls Saved: {additional.get('llm_calls_saved', 0)}")


# ===================== MAIN AREA =====================
col_import, col_chat = st.columns([1, 2])

# === CỘT TRÁI: BULK IMPORT ===
with col_import:
    st.header("📥 Bulk Import")
    st.caption("Ném nội dung vào đây để nhập hàng loạt")
    
    bulk_text = st.text_area(
        "Nhập nội dung (mỗi dòng = 1 message)",
        height=200,
        placeholder="Ví dụ:\nTôi tên Nam, sinh năm 2003\nTôi học Bách Khoa\nTôi thích Python và AI"
    )
    
    col_import_btn, col_clear_btn = st.columns(2)
    with col_import_btn:
        if st.button("📥 Import", use_container_width=True, type="primary"):
            if bulk_text.strip():
                lines = [l.strip() for l in bulk_text.strip().split('\n') if l.strip()]
                
                progress = st.progress(0)
                status = st.empty()
                
                for i, line in enumerate(lines):
                    status.text(f"Processing {i+1}/{len(lines)}...")
                    st.session_state.agent.chat(line)
                    progress.progress((i + 1) / len(lines))
                
                # Auto crystallize
                st.session_state.agent.crystallize(force=True)
                
                status.success(f"✅ Imported {len(lines)} messages!")
                st.rerun()
    
    with col_clear_btn:
        if st.button("🧹 Clear", use_container_width=True):
            st.rerun()
    

# === CỘT PHẢI: CHAT ===
with col_chat:
    version_emoji = "🔴" if version == "V1" else "🟢"
    st.header(f"{version_emoji} Chat với FCM {version}")
    st.caption("💬 Nhập thông tin để lưu nhớ | ❓ Đặt câu hỏi để truy vấn")
    
    # Chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "debug_info" in message:
                    with st.expander("🛠️ Debug Log"):
                        st.json(message["debug_info"])
    
    # Chat input
    if prompt := st.chat_input("Nhập thông tin hoặc đặt câu hỏi... (VD: Nam sinh năm bao nhiêu?)"):
        # User message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Agent response
        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý..."):
                start_time = time.time()
                
                # Detect if this is a question
                user_is_asking = is_question(prompt)
                
                # Always save to memory first
                response_data = st.session_state.agent.chat(
                    prompt,
                    auto_crystallize=True,
                    return_context=True
                )
                
                elapsed = time.time() - start_time
                
                # Search for relevant memories
                search_results = st.session_state.agent.search(
                    prompt, 
                    strategy="enhanced", 
                    limit=10  # <--- QUAN TRỌNG: Lấy 10 kết quả để lọc
                )
                combined = search_results.get('combined', []) if isinstance(search_results, dict) else []
                
                # Log top results to console
                print("\n" + "="*60)
                print(f"🔍 SEARCH: '{prompt[:50]}...'" if len(prompt) > 50 else f"🔍 SEARCH: '{prompt}'")
                print("="*60)
                if combined:
                    for i, r in enumerate(combined[:5]):
                        score = r.get('score', r.get('enhanced_score', 0))
                        mem_type = r.get('metadata', {}).get('fcm_type', 'unknown')
                        mem = r.get('memory', '')[:100]
                        print(f"  {i+1}. [{mem_type}] (score={score:.3f}) {mem}")
                else:
                    print("  (Không có kết quả)")
                print("="*60 + "\n")
                
                # Generate response based on question or statement
                if user_is_asking:
                    # User is asking a question -> generate answer
                    if combined:
                        bot_reply = generate_answer_from_context(prompt, combined, st.session_state.agent)
                    else:
                        bot_reply = "Tôi chưa có thông tin về điều này trong bộ nhớ."
                    mode = "❓ Trả lời câu hỏi"
                else:
                    # User is providing information -> confirm storage
                    bot_reply = f"✅ Đã ghi nhớ: *{prompt}*"
                    mode = "💾 Lưu thông tin"
                
                # Debug info
                debug_info = {
                    "⏱️ Thời gian": f"{elapsed:.2f}s",
                    "🎯 Mode": mode,
                    "💧 Saved to Liquid": response_data.get("liquid_saved", True),
                    "💎 Crystallized": response_data.get("crystallized", False),
                    "🔍 Results found": len(combined),
                }
                
                with st.expander("🛠️ Debug Log", expanded=False):
                    st.json(debug_info)
                    if combined:
                        st.caption("Top memories:")
                        for i, r in enumerate(combined[:3]):
                            score = r.get('score', r.get('enhanced_score', 0))
                            st.caption(f"{i+1}. (score={score:.2f}) {r.get('memory', '')[:80]}...")
                
                st.markdown(bot_reply)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_reply,
            "debug_info": debug_info
        })
        
        st.rerun()


# ===================== MEMORY VISUALIZATION =====================
st.markdown("---")
st.header("🧠 Memory Visualization")

if st.session_state.get("agent"):
    agent = st.session_state.agent
    
    tab1, tab2, tab3, tab4 = st.tabs(["📌 Solid (L3)", "💎 Crystal (L2)", "💧 Liquid (L1)", "⚙️ Config"])
    
    # SOLID LAYER
    with tab1:
        st.info("Long-term Memory: User Profile")
        try:
            profile = agent.get_user_profile()
            if not any(profile.values()):
                st.warning("Trống. Chat nhiều hơn và nhấn 'Evolve'.")
            else:
                for section, facts in profile.items():
                    if facts:
                        with st.expander(f"📌 {section.upper()}", expanded=True):
                            for f in facts:
                                st.write(f"• {f}")
        except Exception as e:
            st.error(f"Lỗi: {e}")
    
    # CRYSTAL LAYER
    with tab2:
        st.info("Mid-term Memory: Facts cụ thể")
        try:
            if hasattr(agent, 'get_crystal_memories'):
                crystals = agent.get_crystal_memories(limit=20)
            else:
                crystals = agent.crystal_layer.get_memories(limit=20)
            
            if crystals:
                for c in crystals:
                    mem = c.get('memory', '')
                    meta = c.get('metadata') or {}
                    category = meta.get('category', 'fact')
                    created_at = meta.get('created_at', '')
                    
                    st.markdown(f"**[{category}]** {mem}")
                    st.caption(f"🕒 {created_at}")
                    st.divider()
            else:
                st.write("*(Chưa có dữ liệu)*")
        except Exception as e:
            st.error(f"Lỗi: {e}")
    
    # LIQUID LAYER
    with tab3:
        st.info("Short-term Memory: Raw messages")
        try:
            if hasattr(agent, 'get_liquid_memories'):
                liquids = agent.get_liquid_memories(limit=20, status="all")
            else:
                liquids = agent.liquid_layer.get_messages(limit=20)
            
            if liquids:
                for l in liquids:
                    mem = l.get('memory', '')
                    meta = l.get('metadata') or {}
                    role = meta.get('role', 'user')
                    ts = str(meta.get('timestamp', ''))[:19]
                    
                    if role == "user":
                        st.info(f"👤 **User** ({ts}):\n{mem}")
                    else:
                        st.success(f"🤖 **Agent** ({ts}):\n{mem}")
            else:
                st.write("*(Bộ đệm trống)*")
        except Exception as e:
            st.error(f"Lỗi: {e}")
    
    # CONFIG
    with tab4:
        st.write("**Current Config:**")
        st.json(agent.config.__dict__)
        
        if st.button("🔄 Refresh"):
            st.rerun()


# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        FCM Demo v0.4.0 | Running <b>FCM {version}</b> | 
        <a href='https://github.com/ngnam1104/FCM'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

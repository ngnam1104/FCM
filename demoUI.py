"""
FCM Agent Demo UI
==================

Giao diện Streamlit để demo FCM Agent với khả năng:
- Chọn giữa FCM V1 và FCM V2
- Chat với agent
- Xem Memory Visualization
- Force Crystallize/Evolve
- Search test

Chạy: streamlit run demoUI.py
"""


import gc
import logging
import sys
import os

# Tắt warning ngay lập tức trước khi import bất kỳ thư viện nào khác
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR) # Tắt thêm logger gốc cho chắc chắn

# --- 2. SAU ĐÓ MỚI IMPORT STREAMLIT VÀ CÁC LIBS KHÁC ---
import streamlit as st
import time

# Thêm đường dẫn (giữ nguyên code cũ của bạn)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

# Chỉ in log kiểm tra API Key một lần duy nhất khi khởi động session
if "api_checked" not in st.session_state:
    print("=" * 60)
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"✅ GROQ_API_KEY found: {api_key[:20]}...")
        # Đánh dấu là đã kiểm tra để không in lại lần sau
        st.session_state["api_checked"] = True
    else:
        print("❌ GROQ_API_KEY NOT FOUND!")
        st.error("❌ GROQ_API_KEY NOT FOUND! Vui lòng kiểm tra file .env")
        st.stop() # Dùng st.stop() thay vì sys.exit() để dừng UI nhẹ nhàng hơn
    print("=" * 60)

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="FCM Agent Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH ---
st.markdown("""
<style>
    .stChatMessage {padding: 1rem; border-radius: 10px; margin-bottom: 10px;}
    .stCodeBlock {font-size: 0.9em;}
    .version-badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }
    .v1-badge {background-color: #ff6b6b; color: white;}
    .v2-badge {background-color: #4ecdc4; color: white;}
</style>
""", unsafe_allow_html=True)


def get_agent(version: str, user_id: str):
    """Initialize FCM Agent based on version"""
    if version == "V1":
        from fcm import FCMAgent, FCMConfig
        config = FCMConfig(
            crystallize_threshold=3,
            verbose=True
        )
        return FCMAgent(config=config, user_id=user_id)
    else:
        from fcm_v2 import FCMAgentV2, FCMConfigV2
        config = FCMConfigV2(
            crystallize_threshold=3,
            attention_sink_count=2,
            semantic_similarity_threshold=0.75,
            enable_active_forgetting=False,
            verbose=True
        )
        return FCMAgentV2(config=config, user_id=user_id)


def reset_agent():
    """Reset agent and clear session state SAFE for Windows"""
    
    # 1. Xóa reference đến Agent cũ
    if "agent" in st.session_state and st.session_state.agent is not None:
        # Nếu agent có hàm close, hãy gọi nó (ví dụ Qdrant client có close())
        # Tuy nhiên mem0 wrap khá kín, nên ta dùng cách xóa object
        del st.session_state.agent
        
    if "agent" in st.session_state:
        del st.session_state["agent"]
        
    # 2. Ép buộc Garbage Collection chạy để giải phóng file handle
    gc.collect()
    
    # 3. Chờ một chút để Windows kịp mở khóa (Hack quan trọng trên Windows)
    time.sleep(1.0) 
    
    # 4. Reset các biến khác
    st.session_state.messages = []
    st.session_state.initialized = False
    
    # 5. Rerun để Streamlit chạy lại từ đầu với state sạch
    st.rerun()


# --- SIDEBAR: BẢNG ĐIỀU KHIỂN ---
with st.sidebar:
    st.title("🧠 FCM Control Panel")
    st.markdown("---")
    
    # Version Selection
    st.subheader("📦 Chọn Phiên Bản")
    
    # Check if version changed
    current_version = st.session_state.get("version", "V2")
    version = st.radio(
        "Architecture",
        ["V1", "V2"],
        index=1 if current_version == "V2" else 0,
        format_func=lambda x: f"FCM {x}" + (" (Basic)" if x == "V1" else " (Enhanced)"),
        help="V1: Basic 3-layer | V2: Enhanced với Attention Sinks, Semantic Grouping, Active Forgetting"
    )
    
    # Handle version change
    if version != st.session_state.get("version"):
        reset_agent()
        st.session_state.version = version
    
    # Version info
    if version == "V1":
        st.markdown('<div class="version-badge v1-badge">FCM V1 - Basic</div>', unsafe_allow_html=True)
        st.caption("Features: Liquid → Crystal → Solid")
    else:
        st.markdown('<div class="version-badge v2-badge">FCM V2 - Enhanced</div>', unsafe_allow_html=True)
        st.caption("Features: Attention Sinks, Semantic Grouping, Active Forgetting, Weighted Retrieval")
    
    st.markdown("---")
    
    # Initialize Agent
    user_id = f"demo_{version.lower()}_user"
    st.session_state.user_id = user_id
    
    if "agent" not in st.session_state or st.session_state.agent is None:
        try:
            with st.spinner(f"Initializing FCM {version}..."):
                st.session_state.agent = get_agent(version, user_id)
                st.session_state.messages = []
                st.session_state.initialized = True
            st.success("Khởi tạo thành công!") # Báo thành công nếu qua được
        except Exception as e:
            st.error(f"❌ Lỗi khởi tạo Agent: {str(e)}")
            st.code(str(e)) # In chi tiết lỗi ra màn hình
            st.stop() # Dừng app lại an toàn
    
    # Control buttons
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
                st.success(f"Done!")
    
    if st.button("🗑️ Reset Memory", use_container_width=True, type="secondary"):
        reset_agent()
        st.rerun()
    
    st.markdown("---")
    
    # Statistics
    st.subheader("📊 Statistics")
    stats = st.session_state.agent.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", stats.get('total_messages', 0))
        st.metric("Liquid", stats.get('liquid_count', 0))
    with col2:
        st.metric("Crystal", stats.get('crystal_count', 0))
        st.metric("Solid", stats.get('solid_count', 0))
    
    # V2 specific metrics
    if version == "V2":
        st.markdown("---")
        st.caption("V2 Metrics:")
        additional = stats.get('additional', {})
        if additional:
            st.caption(f"• Attention Sinks: {additional.get('attention_sinks', 0)}")
            st.caption(f"• LLM Calls Saved: {additional.get('llm_calls_saved', 0)}")
    
    st.markdown("---")
    
    # Search Test
    st.subheader("🔍 Search Test")
    search_query = st.text_input("Query", placeholder="VD: Nam sinh năm bao nhiêu?")
    if st.button("Search", use_container_width=True):
        if search_query:
            results = st.session_state.agent.search(search_query, strategy="enhanced")
            st.write("Results:")
            combined = results.get('combined', [])
            if combined:
                for i, r in enumerate(combined[:3]):
                    score = r.get('score', 0)
                    mem_type = r.get('metadata', {}).get('fcm_type', 'unknown')
                    st.caption(f"{i+1}. [{mem_type}] Score: {score:.2f}")
                    st.code(r.get('memory', '')[:100], language="text")
            else:
                st.warning("Không tìm thấy kết quả")


# --- GIAO DIỆN CHÍNH ---
col_chat, col_brain = st.columns([3, 2])

# === CỘT TRÁI: CHAT ===
with col_chat:
    version_emoji = "🔴" if version == "V1" else "🟢"
    st.header(f"{version_emoji} Chat với FCM {version}")
    
    # Hiển thị lịch sử chat
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Nếu là tin nhắn bot và có log xử lý đi kèm (lưu trong lịch sử)
            if "debug_info" in message:
                with st.expander("🛠️ Chi tiết xử lý (Processing Log)"):
                    st.json(message["debug_info"])

    # Input chat
    if prompt := st.chat_input("Nhập tin nhắn... (VD: Tôi tên Nam, sinh năm 2003)"):
        # 1. Hiển thị tin nhắn User
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Agent xử lý
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ & Ghi nhớ..."):
                start_time = time.time()
                
                # Gọi agent chat
                response_data = st.session_state.agent.chat(
                    prompt,
                    auto_crystallize=True,
                    return_context=True
                )
                
                elapsed = time.time() - start_time
                
                # --- PHẦN HIỂN THỊ LOG CẬP NHẬT ---
                # Tạo một dictionary chứa thông tin cập nhật trong lượt này
                debug_info = {
                    "⏱️ Thời gian xử lý": f"{elapsed:.2f}s",
                    "💧 Saved to Liquid": response_data.get("liquid_saved", False),
                    "🔄 Topic Shift": response_data.get("topic_shifted", False),
                    "💎 Crystallized": response_data.get("crystallized", False),
                    "⚠️ Trigger Reason": response_data.get("crystallize_trigger", "None")
                }
                
                # Hiển thị Log ngay lập tức dưới tin nhắn
                with st.expander("🛠️ Chi tiết xử lý (Processing Log)", expanded=True):
                    st.write(f"**Trạng thái:** {'✅ Đã lưu Liquid' if debug_info['💧 Saved to Liquid'] else '❌ Lỗi lưu'}")
                    if debug_info['🔄 Topic Shift']:
                        st.warning(f"⚡ Phát hiện đổi chủ đề! -> Kích hoạt Crystallize")
                    if debug_info['💎 Crystallized']:
                        st.success(f"💎 Đã kết tinh ký ức vào Crystal Layer!")
                    
                    st.caption("Raw Data:")
                    st.json(debug_info)

                # --- PHẦN HIỂN THỊ CÂU TRẢ LỜI (Đã sửa logic) ---
                context = response_data.get("context", {})
                
                # Xử lý lấy list kết quả an toàn (dict hoặc object)
                combined = []
                if isinstance(context, dict):
                    combined = context.get("combined") or context.get("combined_results", [])
                
                # Logic hiển thị thông minh hơn
                if combined:
                    top_mem = combined[0].get("memory", "")
                    bot_reply = f"Tôi đã ghi nhận. \n\n*Dựa trên ký ức liên quan tìm thấy:* \n> {top_mem}"
                else:
                    bot_reply = "Đã ghi nhận thông tin mới vào bộ nhớ ngắn hạn."

                st.markdown(bot_reply)
        
        # Lưu tin nhắn bot kèm debug_info vào lịch sử
        st.session_state.messages.append({
            "role": "assistant", 
            "content": bot_reply,
            "debug_info": debug_info # Lưu log để render lại khi rerun
        })
        
        # Rerun để cập nhật thống kê bên Sidebar và Cột Brain ngay lập tức
        st.rerun()


# === CỘT PHẢI: BỘ NÃO ===
# === CỘT PHẢI: BỘ NÃO ===
with col_brain:
    st.header("🧠 Memory Visualization")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📌 Solid (L3)", "💎 Crystal (L2)", "💧 Liquid (L1)", "⚙️ Config"])
    
    agent = st.session_state.agent

    # TAB 1: SOLID LAYER (Profile)
    with tab1:
        st.info("Long-term Memory: Hồ sơ người dùng (User Profile)")
        try:
            profile = agent.get_user_profile()
            if not any(profile.values()):
                st.warning("Trống. Hãy chat nhiều hơn và nhấn 'Evolve'.")
            else:
                for section, facts in profile.items():
                    if facts:
                        with st.expander(f"📌 {section.upper()}", expanded=True):
                            for f in facts:
                                st.write(f"• {f}")
        except Exception as e:
            st.error(f"Lỗi lấy Solid Layer: {e}")
    
    # TAB 2: CRYSTAL LAYER (Facts)
    with tab2:
        st.info("Mid-term Memory: Các sự kiện/facts cụ thể")
        try:
            # Gọi hàm riêng cho Crystal
            if hasattr(agent, 'get_crystal_memories'):
                crystals = agent.get_crystal_memories(limit=20)
            else:
                # Fallback nếu chưa update agent.py
                crystals = agent.crystal_layer.get_memories(limit=20)

            if crystals:
                for c in crystals:
                    with st.container():
                        mem = c.get('memory', '')
                        meta = c.get('metadata') or {}
                        category = meta.get('category', 'fact')
                        created_at = meta.get('created_at', '')
                        
                        st.markdown(f"**[{category}]** {mem}")
                        st.caption(f"🕒 {created_at}")
                        st.divider()
            else:
                st.write("*(Chưa có dữ liệu kết tinh)*")
        except Exception as e:
            st.error(f"Lỗi lấy Crystal Layer: {e}")
    
    # TAB 3: LIQUID LAYER (Raw Messages)
    with tab3:
        st.info("Short-term Memory: Bộ đệm hội thoại")
        try:
            # Gọi hàm riêng cho Liquid
            if hasattr(agent, 'get_liquid_memories'):
                liquids = agent.get_liquid_memories(limit=20, status="all")
            else:
                # Fallback
                liquids = agent.liquid_layer.get_messages(limit=20)

            if liquids:
                for l in liquids:
                    mem = l.get('memory', '')
                    meta = l.get('metadata') or {}
                    role = meta.get('role', 'user')
                    ts = str(meta.get('timestamp', ''))[:19]
                    
                    # Style khác nhau cho User/Bot
                    if role == "user":
                        st.info(f"👤 **User** ({ts}):\n{mem}")
                    else:
                        st.success(f"🤖 **Agent** ({ts}):\n{mem}")
            else:
                st.write("*(Bộ đệm trống)*")
        except Exception as e:
            st.error(f"Lỗi lấy Liquid Layer: {e}")

    # TAB 4: CONFIG & DEBUG
    with tab4:
        st.write("**Current Config:**")
        st.json(agent.config.__dict__)
        
        if st.button("Refresh View"):
            st.rerun()


# --- FOOTER ---
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        FCM Demo v0.3.0 | Running <b>FCM {version}</b> | 
        <a href='https://github.com/ngnam1104/FCM'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

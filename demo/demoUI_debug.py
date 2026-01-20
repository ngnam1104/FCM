"""
FCM Agent Demo UI - DEBUG VERSION
"""
import logging
import sys
import os
import shutil

# --- 1. TẮT LOG RÁC ---
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)

import streamlit as st
import time
import gc

# Add FCM root folder to path (parent of demo/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="FCM Debug", layout="wide")

# --- DEBUG HELPER ---
def debug_log(msg):
    """In log ra cả terminal và sidebar để dễ theo dõi"""
    print(f"[DEBUG] {msg}")
    with st.sidebar:
        st.text(f"⏳ {msg}")

# --- CHECK API KEY ---
if "api_checked" not in st.session_state:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ Thiếu GROQ_API_KEY trong file .env")
        st.stop()
    st.session_state["api_checked"] = True

# --- HÀM KHỞI TẠO AN TOÀN ---
def get_agent_safe(version: str, user_id: str):
    debug_log(f"Bắt đầu khởi tạo {version}...")
    
    try:
        if version == "V1":
            debug_log("Importing fcm V1...")
            from fcm import FCMAgent, FCMConfig
            debug_log("Configuring V1...")
            config = FCMConfig(crystallize_threshold=3, verbose=True)
            debug_log("Instantiating V1 Agent (Có thể treo ở đây nếu DB lock)...")
            return FCMAgent(config=config, user_id=user_id)
        else:
            debug_log("Importing fcm.v2...")
            from fcm.v2 import FCMAgentV2, FCMConfigV2
            
            debug_log("Configuring V2...")
            # CẤU HÌNH QDRANT CHẠY RAM ĐỂ TRÁNH LỖI LOCK FILE
            # Lưu ý: Dữ liệu sẽ mất khi tắt app, nhưng đảm bảo chạy được demo
            config = FCMConfigV2(
                crystallize_threshold=3,
                attention_sink_count=2,
                semantic_similarity_threshold=0.75,
                enable_active_forgetting=False,
                verbose=True
            )
            
            # --- HACK: Cưỡng ép dùng In-Memory DB nếu thư viện hỗ trợ ---
            # Nếu class config của bạn có thuộc tính vector_store_config, hãy set path=":memory:"
            # config.vector_store_config = {"path": ":memory:"} 
            
            debug_log("Instantiating V2 Agent...")
            agent = FCMAgentV2(config=config, user_id=user_id)
            debug_log("✅ V2 Agent Ready!")
            return agent

    except Exception as e:
        st.error(f"❌ Lỗi chi tiết: {str(e)}")
        print(f"CRITICAL ERROR: {e}")
        raise e

def reset_agent_safe():
    debug_log("Đang reset...")
    if "agent" in st.session_state:
        del st.session_state.agent
    gc.collect()
    time.sleep(0.5)
    st.session_state.messages = []
    st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛠️ Debug Mode")
    
    version = st.radio("Version", ["V1", "V2"], index=1)
    
    # Nút Force Reset
    if st.button("♻️ FORCE RESET APP"):
        reset_agent_safe()

    # --- KHỞI TẠO ---
    user_id = f"demo_{version}_debug"
    
    if "agent" not in st.session_state:
        st.write("--- LOG KHỞI TẠO ---")
        try:
            st.session_state.agent = get_agent_safe(version, user_id)
            st.session_state.messages = []
            st.success("Khởi tạo xong!")
        except Exception as e:
            st.error("Dừng chương trình do lỗi.")
            st.stop()

    # Actions
    st.markdown("---")
    if st.button("Clear Memory"):
        if st.session_state.agent:
            st.session_state.agent.clear_all_memories()
            st.success("Đã xóa sạch bộ nhớ!")
            time.sleep(1)
            st.rerun()

# --- GIAO DIỆN CHÍNH ---
st.title(f"Chat với FCM {version}")

# Input chat
if prompt := st.chat_input("Nhập tin nhắn..."):
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        if st.session_state.agent:
            # Chat
            debug_log("Agent đang xử lý...")
            res = st.session_state.agent.chat(prompt, return_context=True)
            
            # Show kết quả
            context = res.get("context", {})
            # Xử lý an toàn cho dict
            combined = []
            if isinstance(context, dict):
                 combined = context.get("combined") or context.get("combined_results", [])

            if combined:
                mem = combined[0].get("memory", "...")
                st.write(f"Ký ức: {mem}")
            else:
                st.write("Đã ghi nhớ (Chưa có context cũ).")
            
            debug_log("Xử lý xong.")
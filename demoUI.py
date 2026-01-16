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

import streamlit as st
import os
import sys
import time

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
    """Reset agent and clear session state"""
    if "agent" in st.session_state:
        try:
            st.session_state.agent.memory.delete_all(user_id=st.session_state.user_id)
        except:
            pass
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.initialized = False


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
        with st.spinner(f"Initializing FCM {version}..."):
            st.session_state.agent = get_agent(version, user_id)
            st.session_state.messages = []
            st.session_state.initialized = True
    
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
    
    # Input chat
    if prompt := st.chat_input("Nhập tin nhắn... (VD: Tôi tên Nam, sinh năm 2003)"):
        # 1. Hiển thị tin nhắn User
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 2. Agent xử lý
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                start_time = time.time()
                
                # Gọi agent chat
                response_data = st.session_state.agent.chat(
                    prompt,
                    auto_crystallize=True,
                    return_context=True
                )
                
                elapsed = time.time() - start_time
                
                # Tạo response dựa trên context
                context = response_data.get("context", {})
                combined = context.get("combined", []) if context else []
                
                if combined:
                    # Có context - tạo response từ memory
                    top_memory = combined[0].get("memory", "")
                    bot_reply = f"Dựa trên ký ức của tôi: {top_memory}"
                else:
                    bot_reply = f"Đã ghi nhớ: '{prompt[:50]}...'" if len(prompt) > 50 else f"Đã ghi nhớ: '{prompt}'"
                
                st.markdown(bot_reply)
                
                # Stats
                st.caption(f"⏱️ {elapsed:.2f}s | 📊 {response_data.get('stats', {})}")
                
                # Topic Shift notification
                if response_data.get("topic_shifted"):
                    st.toast("⚡ Topic Shift detected! Crystallizing...", icon="⚡")
        
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()


# === CỘT PHẢI: BỘ NÃO ===
with col_brain:
    st.header("🧠 Memory Visualization")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📌 Solid (Profile)", "💎 Crystal", "💧 Liquid", "🔧 Debug"])
    
    # TAB 1: USER PROFILE (Solid Layer)
    with tab1:
        st.info("Long-term Memory - User Profile consolidated từ các facts")
        
        profile = st.session_state.agent.get_user_profile()
        
        if not any(profile.values()):
            st.warning("Chưa có thông tin Profile. Hãy chat thêm và nhấn Evolve.")
        else:
            for section, facts in profile.items():
                if facts:
                    with st.expander(f"📌 {section.upper()}", expanded=True):
                        for f in facts:
                            st.write(f"• {f}")
    
    # TAB 2: CRYSTAL LAYER
    with tab2:
        st.info("Mid-term Memory - Atomic Facts được trích xuất từ hội thoại")
        
        crystals = st.session_state.agent.get_crystal_memories(limit=20)
        if crystals:
            for c in crystals:
                with st.container():
                    mem = c.get('memory', '')
                    category = c.get('metadata', {}).get('category', 'fact')
                    st.markdown(f"**[{category}]** {mem}")
                    st.divider()
        else:
            st.warning("Chưa có Crystal facts. Hãy chat thêm và nhấn Crystallize.")
    
    # TAB 3: LIQUID LAYER
    with tab3:
        st.info("Short-term Memory - Raw messages chưa xử lý")
        
        liquids = st.session_state.agent.get_liquid_memories(limit=20, status="all")
        if liquids:
            for l in liquids:
                mem = l.get('memory', '')
                role = l.get('metadata', {}).get('role', 'user')
                ts = l.get('metadata', {}).get('timestamp', '')[:19]
                
                icon = "👤" if role == "user" else "🤖"
                st.caption(f"{icon} {ts}")
                st.text(mem[:200])
                st.divider()
        else:
            st.warning("Chưa có Liquid messages.")
    
    # TAB 4: DEBUG
    with tab4:
        st.info("Debug Information")
        
        # All memories by layer
        if st.button("📋 Show All Memories"):
            all_mems = st.session_state.agent.get_all_memories_by_layer()
            
            st.write(f"**Liquid:** {len(all_mems.get('liquid', []))} items")
            st.write(f"**Crystal:** {len(all_mems.get('crystal', []))} items")
            st.write(f"**Solid:** {len(all_mems.get('solid', []))} items")
            
            with st.expander("Raw Data"):
                st.json(all_mems)
        
        # Config
        with st.expander("⚙️ Current Config"):
            config = st.session_state.agent.config
            st.write(f"- LLM: {config.llm_provider}/{config.llm_model}")
            st.write(f"- Crystallize Threshold: {config.crystallize_threshold}")
            st.write(f"- Evolve Threshold: {config.evolve_threshold}")
            if version == "V2":
                st.write(f"- Attention Sinks: {getattr(config, 'attention_sink_count', 'N/A')}")
                st.write(f"- Semantic Threshold: {getattr(config, 'semantic_similarity_threshold', 'N/A')}")


# --- FOOTER ---
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        FCM Demo v0.3.0 | Running <b>FCM {version}</b> | 
        <a href='https://github.com/your-repo/fcm'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

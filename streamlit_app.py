import streamlit as st
import asyncio
import os
import sys
import nest_asyncio

nest_asyncio.apply()

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

st.set_page_config(
    page_title="Project Omega",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .omega-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .omega-sub {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .stStatusWidget { border-radius: 10px; }
    div[data-testid="stChatMessage"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="Free key: aistudio.google.com/apikey"
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ Key loaded")
    elif os.environ.get("GOOGLE_API_KEY"):
        st.success("✅ Key from Cloud Secrets")
    else:
        st.warning("⚠️ Enter key to begin")
    
    st.divider()
    st.markdown("### 🧠 Architecture")
    st.markdown("""
    | Layer | Technology |
    |-------|-----------|
    | Search | SearXNG + DDG |
    | Extraction | Crawl4AI |
    | Reasoning | Gemini 2.5 Pro |
    | Halting | CMI Satiation |
    | Verification | Temporal Grounding |
    | Cost | **$0.00** |
    """)
    
    st.divider()
    st.markdown("**Project Omega** v1.0")
    st.caption("100% Free • Epistemic Verification • Zero Paid APIs")

# --- MAIN ---
st.markdown('<p class="omega-title">🧠 Project Omega</p>', unsafe_allow_html=True)
st.markdown('<p class="omega-sub">Autonomous Deep Research Swarm • Big-Tech Grade • 100% Free Stack</p>', unsafe_allow_html=True)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "report_count" not in st.session_state:
    st.session_state.report_count = 0

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("What should we research deeply today?"):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("🔑 Please enter your Gemini API Key in the sidebar first.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        status = st.status("🚀 Initializing Research Swarm...", expanded=True)
        
        try:
            status.update(label="📦 Loading deep_researcher graph...")
            from open_deep_research.deep_researcher import deep_researcher_builder
            from langgraph.checkpoint.memory import MemorySaver
            from langchain_core.messages import HumanMessage
            
            status.update(label="🔧 Compiling with checkpointing...")
            memory = MemorySaver()
            graph = deep_researcher_builder.compile(checkpointer=memory)
            
            st.session_state.report_count += 1
            config = {
                "configurable": {
                    "thread_id": f"omega_session_{st.session_state.report_count}"
                }
            }
            
            status.update(label="🔍 Dispatching parallel researchers across the web...")
            
            async def run_research():
                return await graph.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config
                )
            
            result = asyncio.run(run_research())
            
            status.update(label="✅ Research Complete!", state="complete", expanded=False)
            
            report = result.get("final_report", "Research finished but no report was generated.")
            st.markdown(report)
            
            st.session_state.messages.append({"role": "assistant", "content": report})
            
            # Evidence graph stats
            evidence = result.get("evidence_graph", [])
            if evidence:
                st.divider()
                cols = st.columns(3)
                cols[0].metric("📚 Sources Found", len(evidence))
                cols[1].metric("🔍 Research Iterations", result.get("research_iterations", "N/A"))
                cols[2].metric("💰 Total Cost", "$0.00")
            
            # Download
            st.download_button(
                label="📥 Download Report as Markdown",
                data=report,
                file_name=f"omega_research_{st.session_state.report_count}.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        except Exception as e:
            status.update(label="❌ Research Failed", state="error")
            error_msg = f"**Error:** {str(e)}\n\nCheck that your Gemini API key is valid and has quota remaining."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

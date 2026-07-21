import streamlit as st
import asyncio
import os
import sys
import concurrent.futures

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

st.set_page_config(page_title="Project Omega", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .omega-title { font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; }
    .omega-sub { color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    api_key = st.text_input("🔑 Gemini API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""), help="Free: aistudio.google.com/apikey")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ Key loaded")
    elif os.environ.get("GOOGLE_API_KEY"):
        st.success("✅ Key from Cloud Secrets")
    else:
        st.warning("⚠️ Enter key to begin")
    st.divider()
    st.markdown("### 🧠 Stack")
    st.markdown("| Layer | Tech |\n|-------|------|\n| Search | SearXNG + DDG |\n| Extract | Crawl4AI |\n| Reason | Gemini 2.5 Pro |\n| Halt | CMI Satiation |\n| Cost | **$0.00** |")
    st.divider()
    st.markdown("**Project Omega** v1.0")

st.markdown('<p class="omega-title">🧠 Project Omega</p>', unsafe_allow_html=True)
st.markdown('<p class="omega-sub">Autonomous Deep Research Swarm • 100% Free</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "report_count" not in st.session_state:
    st.session_state.report_count = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What should we research deeply today?"):
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("🔑 Enter Gemini API Key first.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        status = st.status("🚀 Initializing Swarm...", expanded=True)
        try:
            status.update(label="📦 Loading graph...")
            from open_deep_research.deep_researcher import deep_researcher
            from langgraph.checkpoint.memory import MemorySaver
            from langchain_core.messages import HumanMessage
            
            # MemorySaver is already compiled in deep_researcher.py now, 
            # but we ensure it's fresh for the session
            config = {"configurable": {"thread_id": "omega_" + str(st.session_state.report_count)}}
            
            status.update(label="🔍 Dispatching researchers...")
            
            # UVLOOP SAFE ASYNC EXECUTION
            def _run():
                async def _inner():
                    return await deep_researcher.ainvoke(
                        {"messages": [HumanMessage(content=prompt)]}, 
                        config=config
                    )
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _inner()).result()
                    
            result = _run()
            st.session_state.report_count += 1
            
            status.update(label="✅ Complete!", state="complete", expanded=False)
            report = result.get("final_report", "No report generated.")
            st.markdown(report)
            st.session_state.messages.append({"role": "assistant", "content": report})
            
            evidence = result.get("evidence_graph", [])
            if evidence:
                st.divider()
                c = st.columns(3)
                c[0].metric("📚 Sources", len(evidence))
                c[1].metric("🔍 Iterations", result.get("research_iterations", "N/A"))
                c[2].metric("💰 Cost", "$0.00")
            st.download_button("📥 Download MD", report, "omega_report.md", "text/markdown", use_container_width=True)
        except Exception as e:
            status.update(label="❌ Failed", state="error")
            err = "Error: " + str(e)
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
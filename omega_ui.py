"""Omega Supremacy Streamlit Control Plane."""
import asyncio, os, sys, traceback, uuid, streamlit as st
from langchain_core.messages import HumanMessage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path: sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
st.set_page_config(page_title="Omega Supremacy | Deep Research", layout="wide", page_icon="🌌", initial_sidebar_state="expanded")
st.markdown("""<style>.stStatus{border:1px solid #4CAF50;border-radius:8px;padding:10px}.stMetric{background-color:#1E1E1E;padding:15px;border-radius:8px;border:1px solid #333}.omega-title{font-size:2.8rem;font-weight:800;background:linear-gradient(135deg,#667eea 0%,#764ba2 50%,#f093fb 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem}.omega-sub{color:#888;font-size:1rem;margin-bottom:2rem;font-weight:300}.telemetry-text{font-family:'Courier New',monospace;font-size:0.85rem;color:#00FF00}</style>""", unsafe_allow_html=True)
st.markdown('<p class="omega-title">🌌 Omega Supremacy Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="omega-sub">Algorithmic Ruthlessness. Zero-Token Fact-Checking. Groq LPU Supremacy.</p>', unsafe_allow_html=True)
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "telemetry" not in st.session_state: st.session_state.telemetry = []
if "final_report" not in st.session_state: st.session_state.final_report = None
if "audit_metrics" not in st.session_state: st.session_state.audit_metrics = {}
if "last_error" not in st.session_state: st.session_state.last_error = None
def _env_preview(name):
    value = os.environ.get(name, "")
    if not value: return "Offline"
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return "Online (" + str(len(parts)) + " key" + ("s" if len(parts) != 1 else "") + ")"
def _run_graph(query):
    from open_deep_research.deep_researcher import deep_researcher
    config = {"recursion_limit": 250, "configurable": {"thread_id": st.session_state.thread_id, "search_api": "jina", "allow_clarification": True}}
    input_state = {"messages": [HumanMessage(content=query)]}
    try: return asyncio.run(deep_researcher.ainvoke(input_state, config=config))
    except RuntimeError as err:
        msg = str(err).lower()
        if "asyncio.run()" in msg or "event loop" in msg:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(deep_researcher.ainvoke(input_state, config=config))
            finally: loop.close()
        raise
def _as_dict(state_obj):
    if isinstance(state_obj, dict): return state_obj
    values = getattr(state_obj, "values", None)
    if isinstance(values, dict): return values
    try: return dict(state_obj)
    except Exception: return {}
with st.sidebar:
    st.markdown("## Control Plane")
    st.success("Thread ID: " + str(st.session_state.thread_id[:8]) + "...")
    st.info("**Groq Brain:** " + _env_preview("GROQ_API_KEYS"))
    st.info("**Jina Eyes:** " + _env_preview("JINA_API_KEY"))
    if st.button("Purge Memory (New Session)", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.telemetry = []
        st.session_state.final_report = None
        st.session_state.audit_metrics = {}
        st.session_state.last_error = None
        st.rerun()
    st.divider()
    st.markdown("### Active Architecture")
    st.markdown("- **Pure Groq Supremacy** *(Zero Gemini Dependency)*")
    st.markdown("- **VFS Artifact Pattern** *(Context Rot Immunity)*")
    st.markdown("- **Algorithmic Epistemic Halting** *(Zero-Token Math)*")
query = st.chat_input("Enter your complex research directive...")
if query:
    st.session_state.telemetry = ["Directive Received: " + query]
    st.session_state.final_report = None
    st.session_state.audit_metrics = {}
    st.session_state.last_error = None
    status = st.status("Running deep research...", expanded=True)
    report_placeholder = st.empty()
    try:
        with st.spinner("Research engine is running..."):
            final_state = _run_graph(query)
        values = _as_dict(final_state)
        if values:
            st.session_state.final_report = values.get("final_report", st.session_state.final_report)
            if values.get("research_plan"): st.session_state.telemetry.append("DAG Topology: " + str(len(values["research_plan"])) + " research vectors.")
            if values.get("evidence_graph"): st.session_state.telemetry.append("Evidence Graph: " + str(len(values["evidence_graph"])) + " atomic nodes.")
            if values.get("virtual_filesystem"): st.session_state.telemetry.append("VFS Artifacts: " + str(len(values["virtual_filesystem"])) + " heavy payloads secured.")
            st.session_state.audit_metrics = {"confidence": float(values.get("confidence_score", 0.0) or 0.0), "nodes": len(values.get("evidence_graph", [])), "vfs": len(values.get("virtual_filesystem", {})), "consensus": values.get("consensus_report", "N/A"), "red_team": values.get("red_team_findings", "N/A")}
        st.session_state.telemetry.append("Graph Execution: completed via async API.")
        status.update(label="Epistemic Synthesis Complete", state="complete")
    except Exception as e:
        st.session_state.last_error = str(e)
        status.update(label="System Fault: " + str(e), state="error")
        st.error("Execution Error: " + str(e))
        with st.expander("Traceback"): st.code(traceback.format_exc())
if st.session_state.telemetry:
    st.markdown('<div class="telemetry-text">' + "<br>".join(st.session_state.telemetry[-20:]) + '</div>', unsafe_allow_html=True)
if st.session_state.final_report:
    report_placeholder.markdown(st.session_state.final_report)
    st.download_button(label="Download Report (MD)", data=st.session_state.final_report, file_name="omega_report.md", mime="text/markdown", use_container_width=True)
if st.session_state.last_error and not st.session_state.final_report:
    st.warning("Last error: " + str(st.session_state.last_error))

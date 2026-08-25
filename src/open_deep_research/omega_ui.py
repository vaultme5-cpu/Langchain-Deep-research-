"""Omega Supremacy Streamlit Control Plane."""
import asyncio
import os
import sys
import traceback
import uuid
import html
import streamlit as st
from langchain_core.messages import HumanMessage
import logging as _lg
class _DropCkpt(_lg.Filter):
    def filter(self, r):
        return "Deserializing unregistered type" not in r.getMessage()
_lg.getLogger().addFilter(_DropCkpt())
import time as _ui_time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PATH_CANDIDATES = [
    BASE_DIR,
    os.path.dirname(BASE_DIR),
    os.path.dirname(os.path.dirname(BASE_DIR)),
    os.path.join(BASE_DIR, "src"),
    os.path.join(os.path.dirname(BASE_DIR), "src"),
    os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "src")
]
for _p in _PATH_CANDIDATES:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

st.set_page_config(
    page_title="Omega Supremacy | Deep Research",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
.stStatus { border: 1px solid #4CAF50; border-radius: 8px; padding: 10px; }
.stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #333; }
.omega-title {
font-size: 2.8rem; font-weight: 800;
background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
-webkit-background-clip: text; -webkit-text-fill-color: transparent;
margin-bottom: 0.2rem;
}
.omega-sub { color: #888; font-size: 1rem; margin-bottom: 2rem; font-weight: 300; }
.telemetry-text { font-family: Courier New, monospace; font-size: 0.85rem; color: #00FF00; }
</style>
""",
    unsafe_allow_html=True
)

st.markdown("<p class=omega-title>🌌 Omega Supremacy Engine</p>", unsafe_allow_html=True)
st.markdown("<p class=omega-sub>Algorithmic Ruthlessness. Zero-Token Fact-Checking. Groq LPU Supremacy.</p>", unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "telemetry" not in st.session_state:
    st.session_state.telemetry = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "audit_metrics" not in st.session_state:
    st.session_state.audit_metrics = {}
if "last_error" not in st.session_state:
    st.session_state.last_error = None

def _env_preview(name: str) -> str:
    value = os.environ.get(name, "")
    if not value: return "Offline"
    parts = [p.strip() for p in value.split(",") if p.strip()]
    suffix = "s" if len(parts) != 1 else ""
    return "Online (" + str(len(parts)) + " key" + suffix + ")"

def _run_graph(query: str):
    from open_deep_research.deep_researcher import deep_researcher
    config = {
        "recursion_limit": 250,
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "allow_clarification": True
        }
    }
    input_state = {"messages": [HumanMessage(content=query)]}
    try:
        return asyncio.run(deep_researcher.ainvoke(input_state, config=config))
    except RuntimeError as err:
        msg = str(err).lower()
        if "asyncio.run()" in msg or "event loop" in msg:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(deep_researcher.ainvoke(input_state, config=config))
            finally:
                loop.close()
        raise

def _as_dict(state_obj):
    if isinstance(state_obj, dict): return state_obj
    values = getattr(state_obj, "values", None)
    if isinstance(values, dict): return values
    try: return dict(state_obj)
    except Exception: return {}

with st.sidebar:
    st.markdown("## Control Plane")
    st.success("**Thread ID:** " + str(st.session_state.thread_id[:8]) + "...")
    st.info("**Groq Brain:** " + _env_preview("GROQ_API_KEYS"))
    st.info("**Jina Eyes:** " + _env_preview("JINA_API_KEY"))
    try:
        from open_deep_research import deep_researcher as _dr

        # I15: UI telemetry must read the active run context,
        # never the legacy process-global compatibility mirrors.
        _ctx = _dr._get_q()
        _run_budget = getattr(_ctx, "run_budget", {}) or {}
        _brain_health = getattr(_ctx, "brain_health", {}) or {}

        st.caption(
            "Token budget: "
            + str(int(_run_budget.get("used", 0)))
            + " / "
            + str(int(_run_budget.get("cap", 0)))
            + " used"
        )

        _lk = [
            k
            for k in list(_brain_health.keys())
            if _dr._brain_is_open(k)
        ]

        st.caption(
            "Locked brains: "
            + (", ".join(_lk) if _lk else "none")
        )
    except Exception:
        pass
    if st.button("New Research Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.telemetry = []
        st.session_state.final_report = None
        st.session_state.audit_metrics = {}
        st.session_state.last_error = None
        st.rerun()
    st.divider()
    st.markdown("### Active Architecture")
    st.markdown("- VFS Artifact Pattern (Context Rot Immunity)")
    st.markdown("- Search-as-Code (Sandboxed Execution)")
    st.markdown("- Algorithmic Epistemic Halting (Zero-Token Math)")
    st.markdown("- Local/Jina PDF Engine (Groq-only Path B)")
    st.markdown("- GroqShield Multi-Key Pool (Burst Protection)")

query = st.chat_input("Enter your complex research directive...")
if query:
    previous_error = str(st.session_state.get("last_error", "") or "")
    previous_fail_ts = float(st.session_state.get("last_fail_ts", 0.0))

    if (
        ("exhausted" in previous_error.lower() or "locked" in previous_error.lower())
        and (_ui_time.time() - previous_fail_ts < 300.0)
    ):
        st.warning(
            "Capacity temporarily locked after a quota failure. "
            "This run was not started."
        )
        st.stop()

    st.session_state.telemetry = [
        "Directive Received: " + str(query)
    ]
    st.session_state.final_report = None
    st.session_state.audit_metrics = {}
    st.session_state.last_error = None
    status = st.status("Running deep research...", expanded=True)
    try:
        import time as _rt
        _t0 = _rt.time()
        with st.spinner("Research engine is running..."):
            final_state = _run_graph(query)
        st.session_state.telemetry.append("Clock: " + str(int(_rt.time() - _t0)) + "s wall time.")
        values = _as_dict(final_state)
        if values:
            st.session_state.final_report = values.get("final_report", st.session_state.final_report)
            if values.get("research_plan"):
                st.session_state.telemetry.append("**DAG Topology:** " + str(len(values.get("research_plan", []))) + " research vectors.")
            if values.get("evidence_graph"):
                st.session_state.telemetry.append("**Evidence Graph:** " + str(len(values.get("evidence_graph", []))) + " atomic nodes.")
            if values.get("virtual_filesystem"):
                st.session_state.telemetry.append("**VFS Artifacts:** " + str(len(values.get("virtual_filesystem", {}))) + " heavy payloads secured.")
            if values.get("red_team_findings") not in [None, "", "N/A", "Skipped"]:
                st.session_state.telemetry.append("**Adversarial Verification** complete.")
            st.session_state.audit_metrics = {
                "confidence": float(values.get("confidence_score", 0.0) or 0.0),
                "nodes": len(values.get("evidence_graph", [])),
                "vfs": len(values.get("virtual_filesystem", {})),
                "consensus": values.get("consensus_report", "N/A"),
                "red_team": values.get("red_team_findings", "N/A")
            }
        st.session_state.telemetry.append("**Graph Execution:** completed via async API.")
        status.update(label="Epistemic Synthesis Complete", state="complete")
    except Exception as e:
        st.session_state.last_error = str(e)
        st.session_state.last_fail_ts = _ui_time.time()
        status.update(label="System Fault: " + str(e), state="error")
        st.error("Execution Error: " + str(e))
        if os.environ.get("OMEGA_DEBUG", "").lower() in {"1", "true", "yes"}:
            with st.expander("Traceback"):
                st.code(traceback.format_exc(), language="text")

if st.session_state.telemetry:
    safe_lines = [
        html.escape(str(x))
        for x in st.session_state.telemetry[-20:]
    ]
    st.markdown(
        "<div class=telemetry-text>"
        + "<br>".join(safe_lines)
        + "</div>",
        unsafe_allow_html=True
    )

if st.session_state.final_report:
    st.markdown(st.session_state.final_report)
    st.download_button(
        label="Download Report (MD)",
        data=st.session_state.final_report,
        file_name="omega_report.md",
        mime="text/markdown",
        use_container_width=True
    )
    metrics = st.session_state.audit_metrics
    if metrics:
        with st.expander("Epistemic Audit & Telemetry Graph", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", str(round(float(metrics.get("confidence", 0.0)), 2)))
            col2.metric("Evidence Nodes", metrics.get("nodes", 0))
            col3.metric("VFS Artifacts", metrics.get("vfs", 0))
            st.markdown("**Consensus Report:** " + str(metrics.get("consensus", "N/A")))
            st.markdown("**Red Team Findings:** " + str(metrics.get("red_team", "N/A")))
            try:
                from open_deep_research.deep_researcher import deep_researcher
                final_state = deep_researcher.get_state({"configurable": {"thread_id": st.session_state.thread_id}})
                values = _as_dict(final_state)
                evidence = values.get("evidence_graph", [])
                if evidence:
                    st.markdown("### Top Evidence Nodes")
                    for i, node in enumerate(evidence[:10]):
                        st.markdown("- **Fact " + str(i+1) + ":** " + str(getattr(node, "claim", "N/A")) + " ([Source](" + str(getattr(node, "url", "#")) + "))")
            except Exception:
                pass

if st.session_state.last_error and not st.session_state.final_report:
    st.warning("Last error: " + str(st.session_state.last_error))


# ============================================================
# I17.14: UI SEMANTIC CORRECTNESS
# Rule: new session != delete persistent memory.
# The "New Research Session" button ONLY starts a new run context.
# It MUST NOT call _memory_delete_all() or destroy persistent records.
# If persistent-memory deletion is provided, it MUST be a separate,
# explicitly labeled destructive button.
# ============================================================

import streamlit as st
import uuid
import os
import sys

# God-Tier Pathing Bypass: Ensures Streamlit finds the src/ folder regardless of launch directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from langchain_core.messages import HumanMessage
from open_deep_research.deep_researcher import deep_researcher

# ==========================================
# 1. UI CONFIGURATION & CSS INJECTION
# ==========================================
st.set_page_config(
    page_title="Omega Supremacy | Deep Research", 
    layout="wide", 
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stStatus { border: 1px solid #4CAF50; border-radius: 8px; padding: 10px; }
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #333; }
    .omega-title { 
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; 
    }
    .omega-sub { color: #888; font-size: 1rem; margin-bottom: 2rem; font-weight: 300; }
    .telemetry-text { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #00FF00; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "telemetry" not in st.session_state:
    st.session_state.telemetry = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "audit_metrics" not in st.session_state:
    st.session_state.audit_metrics = {}

# ==========================================
# 3. SIDEBAR CONTROL PLANE
# ==========================================
with st.sidebar:
    st.markdown("## ⚙️ Control Plane")
    
    st.success(f"🧠 **Thread ID:** `{st.session_state.thread_id[:8]}...`")
    
    if st.button("🔄 Purge Memory (New Session)", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.telemetry = []
        st.session_state.final_report = None
        st.session_state.audit_metrics = {}
        st.rerun()
        
    st.divider()
    st.markdown("### 🏗️ Active Architecture")
    st.markdown("- **VFS Artifact Pattern** *(Context Rot Immunity)*")
    st.markdown("- **Search-as-Code (SaC)** *(Sandboxed Execution)*")
    st.markdown("- **Algorithmic Epistemic Halting** *(Zero-Token Math)*")
    st.markdown("- **3-Tier Gemini PDF Engine** *(Cloud OCR Offload)*")
    st.markdown("- **GroqShield Multi-Key Pool** *(Burst Protection)*")

# ==========================================
# 4. MAIN HEADER
# ==========================================
st.markdown('<p class="omega-title">🌌 Omega Supremacy Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="omega-sub">Algorithmic Ruthlessness. Zero-Token Fact-Checking. Groq LPU Supremacy.</p>', unsafe_allow_html=True)

# ==========================================
# 5. CHAT INPUT & EXECUTION LOOP
# ==========================================
query = st.chat_input("Enter your complex research directive...")

if query:
    st.session_state.telemetry = [f"🚀 **Directive Received:** {query}"]
    st.session_state.final_report = None
    
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "search_api": "jina",
            "allow_clarification": True
        }
    }
    
    input_state = {"messages": [HumanMessage(content=query)]}
    
    status = st.status("⏳ Initializing Meta-Cognitive Router...", expanded=True)
    telemetry_container = status.container()
    report_placeholder = st.empty()
    
    try:
        # Stream state updates from LangGraph (The UX Killer Feature)
        for chunk in deep_researcher.stream(input_state, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                st.session_state.telemetry.append(f"⚙️ **Executed Node:** `{node_name}`")
                
                # Parse Evidence Graph Expansion
                if "evidence_graph" in update and update["evidence_graph"]:
                    ev_len = len(update["evidence_graph"])
                    st.session_state.telemetry.append(f"🕸️ **Evidence Graph** expanded to **{ev_len}** atomic nodes.")
                    
                # Parse DAG Topology
                if "research_plan" in update and update["research_plan"]:
                    st.session_state.telemetry.append(f"🗺️ **DAG Topology** mapped with **{len(update['research_plan'])}** research vectors.")
                    
                # Parse VFS Artifacts
                if "virtual_filesystem" in update and update["virtual_filesystem"]:
                    vfs_len = len(update["virtual_filesystem"])
                    st.session_state.telemetry.append(f"📂 **VFS Artifacts** secured: **{vfs_len}** heavy payloads bypassed from context.")
                    
                # Parse Adversarial Verification
                if "red_team_findings" in update and update["red_team_findings"] not in ["Skipped", "N/A", ""]:
                    st.session_state.telemetry.append(f"🛡️ **Adversarial Verification** complete.")
                    
                # Capture Final Report
                if "final_report" in update and update["final_report"]:
                    st.session_state.final_report = update["final_report"]
                
                # Keep telemetry buffer manageable
                if len(st.session_state.telemetry) > 60:
                    st.session_state.telemetry = st.session_state.telemetry[-60:]
                    
                # Render Telemetry
                telemetry_container.markdown(
                    f'<div class="telemetry-text">{"".join(["<br>".join(st.session_state.telemetry[-20:])])}</div>', 
                    unsafe_allow_html=True
                )
                
        status.update(label="✅ Epistemic Synthesis Complete", state="complete")
        
        # Fetch Final Audit Metrics
        try:
            final_state = deep_researcher.get_state(config)
            values = final_state.values
            st.session_state.audit_metrics = {
                "confidence": values.get("confidence_score", 0.0),
                "nodes": len(values.get("evidence_graph", [])),
                "vfs": len(values.get("virtual_filesystem", {})),
                "consensus": values.get("consensus_report", "N/A"),
                "red_team": values.get("red_team_findings", "N/A")
            }
        except Exception:
            pass

    except Exception as e:
        status.update(label=f"❌ System Fault: {str(e)}", state="error")
        st.error(f"Execution Error: {str(e)}")

# ==========================================
# 6. REPORT RENDERING & EPISTEMIC AUDIT
# ==========================================
if st.session_state.final_report:
    report_placeholder.markdown(st.session_state.final_report)
    
    st.download_button(
        label="📥 Download Report (MD)", 
        data=st.session_state.final_report, 
        file_name="omega_report.md", 
        mime="text/markdown", 
        use_container_width=True
    )
    
    metrics = st.session_state.audit_metrics
    if metrics:
        with st.expander("🛡️ Epistemic Audit & Telemetry Graph", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", f"{float(metrics.get('confidence', 0.0)):.2f}")
            col2.metric("Evidence Nodes", metrics.get("nodes", 0))
            col3.metric("VFS Artifacts", metrics.get("vfs", 0))
            
            st.markdown(f"**Consensus Report:** {metrics.get('consensus', 'N/A')}")
            st.markdown(f"**Red Team Findings:** {metrics.get('red_team', 'N/A')}")
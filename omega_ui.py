import streamlit as st
import uuid
import os
from langchain_core.messages import HumanMessage
from open_deep_research.deep_researcher import deep_researcher

st.set_page_config(page_title="Omega Supremacy | Deep Research", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    .stStatus { border: 1px solid #4CAF50; border-radius: 5px; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    .omega-title { font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; }
    .omega-sub { color: #888; font-size: 0.95rem; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="omega-title">🌌 Omega Supremacy Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="omega-sub">Algorithmic Ruthlessness. Zero-Token Fact-Checking. Groq LPU Supremacy.</p>', unsafe_allow_html=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "telemetry" not in st.session_state:
    st.session_state.telemetry = []
if "final_report" not in st.session_state:
    st.session_state.final_report = None

with st.sidebar:
    st.header("⚙️ Control Plane")
    
    # Environment Check
    groq_keys = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
    jina_key = os.environ.get("JINA_API_KEY", "")
    
    if groq_keys:
        st.success(f"🧠 Groq Brain: Online ({len(groq_keys.split(','))} Keys)")
    else:
        st.error("🧠 Groq Brain: Offline (Add GROQ_API_KEYS to secrets)")
        
    if jina_key:
        st.success("👁️ Jina Eyes: Online (Cloudflare Piercer)")
    else:
        st.warning("👁️ Jina Eyes: Offline (Add JINA_API_KEY to secrets)")

    if st.button("🔄 Purge Memory (New Session)"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.telemetry = []
        st.session_state.final_report = None
        st.rerun()
        
    st.divider()
    st.markdown("### 🧠 Active Architecture")
    st.markdown("- **VFS Artifact Pattern** (Context Rot Immunity)")
    st.markdown("- **Search-as-Code (SaC)** Sandbox")
    st.markdown("- **Algorithmic Epistemic Halting**")
    st.markdown("- **3-Tier Gemini PDF Engine**")

query = st.chat_input("Enter your complex research directive...")

if query:
    st.session_state.telemetry.append(f"🚀 **Directive Received:** {query}")
    
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "search_api": "searxng", # Defaults to Searxng/Jina fallback chain
            "allow_clarification": True
        }
    }
    
    input_state = {"messages": [HumanMessage(content=query)]}
    
    status = st.status("⏳ Initializing Meta-Cognitive Router...", expanded=True)
    telemetry_container = status.container()
    report_placeholder = st.empty()
    
    try:
        # Stream state updates from LangGraph (The UX Killer Feature)
        for chunk in deep_researcher.stream(input_state, config=config):
            for node_name, update in chunk.items():
                st.session_state.telemetry.append(f"⚙️ **Executed Node:** `{node_name}`")
                
                if "evidence_graph" in update and update["evidence_graph"]:
                    ev_len = len(update["evidence_graph"])
                    st.session_state.telemetry.append(f"🕸️ **Evidence Graph** expanded to **{ev_len}** atomic nodes.")
                    
                if "research_plan" in update and update["research_plan"]:
                    st.session_state.telemetry.append(f"🗺️ **DAG Topology** mapped with **{len(update['research_plan'])}** research vectors.")
                    
                if "virtual_filesystem" in update and update["virtual_filesystem"]:
                    vfs_len = len(update["virtual_filesystem"])
                    st.session_state.telemetry.append(f"📂 **VFS Artifacts** secured: **{vfs_len}** heavy payloads bypassed from context.")
                    
                if "red_team_findings" in update and update["red_team_findings"] not in ["Skipped", "N/A", ""]:
                    st.session_state.telemetry.append(f"🛡️ **Adversarial Verification** complete.")
                    
                if len(st.session_state.telemetry) > 60:
                    st.session_state.telemetry = st.session_state.telemetry[-60:]
                    
                telemetry_container.markdown("\n\n".join(st.session_state.telemetry[-20:]))
                
                if "final_report" in update and update["final_report"]:
                    st.session_state.final_report = update["final_report"]
                    
        status.update(label="✅ Epistemic Synthesis Complete", state="complete")
        
    except Exception as e:
        status.update(label=f"❌ System Fault: {str(e)}", state="error")
        st.error(f"Details: {str(e)}")

if st.session_state.final_report:
    report_placeholder.markdown(st.session_state.final_report)
    
    try:
        final_state = deep_researcher.get_state({"configurable": {"thread_id": st.session_state.thread_id}})
        values = final_state.values
        
        with st.expander("🛡️ Epistemic Audit & Telemetry Graph"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Score", f"{values.get('confidence_score', 0.0):.2f}")
            col2.metric("Evidence Nodes", len(values.get("evidence_graph", [])))
            col3.metric("VFS Artifacts", len(values.get("virtual_filesystem", {})))
            
            st.markdown(f"**Consensus Report:** {values.get('consensus_report', 'N/A')}")
            st.markdown(f"**Red Team Findings:** {values.get('red_team_findings', 'N/A')}")
            
            evidence = values.get("evidence_graph", [])
            if evidence:
                st.markdown("### 🕸️ Top Evidence Nodes")
                for i, node in enumerate(evidence[:10]):
                    st.markdown(f"- **Fact {i+1}:** {getattr(node, 'claim', 'N/A')} ([Source]({getattr(node, 'url', '#')}))")
    except Exception as e:
        st.warning(f"Could not load final telemetry state: {str(e)}")
        
    st.download_button("📥 Download Report (MD)", st.session_state.final_report, "omega_report.md", "text/markdown", use_container_width=True)
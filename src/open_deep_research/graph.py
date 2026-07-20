"""Deep Research Graph Construction - Upgraded for Parallel Swarm & Gap Analysis"""
import hashlib
import re
from typing import cast, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field

# Import your patched state
from .state import AgentState, EvidenceNode 

# ==========================================
# MOCK SEARCH TOOL (Replace with Tavily/Exa)
# ==========================================
def web_search(query: str) -> list[dict]:
    """Mock search tool. Replace this with your actual Tavily/Serper API call."""
    # Returns format: [{"url": "...", "content": "...", "title": "..."}]
    return [
        {"url": f"https://example.com/{hash(query) % 1000}", "title": f"Article about {query}", "content": f"This is deep content about {query}..."}
    ]

# ==========================================
# NODE 1: THE BRAIN (Supervisor)
# Solves Flaw 1 (Concurrency) & Flaw 6 (Gap Analysis)
# ==========================================
class SupervisorDecision(BaseModel):
    reasoning: str = Field(description="Analysis of what we know and what is missing.")
    next_steps: list[str] = Field(description="List of highly specific sub-topics to research next. Empty if done.")
    is_complete: bool = Field(description="True if we have enough diverse, verified evidence to write the final report.")

def supervisor_node(state: AgentState) -> list[Send] | dict:
    """Analyzes gaps and spawns parallel researchers."""
    print("🧠 Supervisor: Analyzing knowledge gaps...")
    
    evidence_count = len(state.get("evidence_graph", []))
    compressed_notes = state.get("compressed_research", [])
    
    # Get the original user prompt
    user_query = "Deep Research Task"
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # Format current knowledge (Data Diet - only read the last 5 summaries to save context)
    knowledge_summary = f"Total Unique Sources Found: {evidence_count}\n\n"
    if compressed_notes:
        knowledge_summary += "Recent Findings:\n" + "\n---\n".join(compressed_notes[-5:])
    else:
        knowledge_summary += "No research conducted yet."

    llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(SupervisorDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Lead Research Supervisor. Your goal is to gather comprehensive evidence to answer the user's prompt.
        Analyze the research gathered so far. 
        If there are missing perspectives, contradictions, or lack of depth, output the specific sub-topics we need to research next.
        If we have comprehensive data covering all aspects, set is_complete to True."""),
        ("human", "User Prompt: {query}\n\nCurrent Knowledge:\n{knowledge}")
    ])
    
    decision = llm.invoke(prompt.format(query=user_query, knowledge=knowledge_summary))
    
    # 1. If we need more research -> Spawn the Swarm! (Parallel Execution)
    if not decision.is_complete and decision.next_steps:
        print(f"🚀 Supervisor: Spawning {len(decision.next_steps)} parallel researchers...")
        
        # LangGraph intercepts this list and runs 'researcher_node' in parallel
        return [
            Send("researcher", {"research_topic": topic, "messages": [AIMessage(content=f"Starting research on: {topic}")]}) 
            for topic in decision.next_steps
        ]
        
    # 2. If research is complete -> Route to Synthesizer
    print("✅ Supervisor: Research complete. Routing to Synthesizer.")
    return {
        "research_brief": decision.reasoning,
        "messages": [AIMessage(content=f"Research phase complete. Reasoning: {decision.reasoning}")]
    }

# ==========================================
# NODE 2: THE WORKER (Researcher)
# Solves Flaw 2 (Context Diet), Flaw 3 (Deduplication), Flaw 4 (Citations)
# ==========================================
class ResearcherOutput(BaseModel):
    compressed_summary: str = Field(description="Compressed summary. MUST use <cit>X</cit> tags.")
    key_excerpts: list[str] = Field(description="Direct quotes with <cit>X</cit> tags.")

def researcher_node(state: dict) -> dict:
    """Searches, compresses, and grounds data."""
    topic = state["research_topic"]
    print(f"🔍 Researcher: Investigating '{topic}'...")
    
    # 1. Search the web
    search_results = web_search(topic) 
    
    # 2. Create Evidence Nodes & Hash URLs (Auto-Deduplication)
    new_evidence = []
    raw_text_for_llm = ""
    
    for i, res in enumerate(search_results):
        # Create unique doc_id hash so the state reducer catches duplicates
        doc_id = hashlib.md5(res["url"].encode()).hexdigest()
        local_citation = i + 1 
        
        # Truncate content to save tokens (The Data Diet)
        raw_text_for_llm += f"\n\nSource [{local_citation}] ({res['url']}):\n{res['content'][:2000]}" 
        
        new_evidence.append({
            "doc_id": doc_id,
            "url": res["url"],
            "title": res["title"],
            "snippet": res["content"][:200],
            "citation_index": local_citation # Global reducer will re-assign final index
        })

    # 3. Compress & Enforce Citations
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2).with_structured_output(ResearcherOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert researcher. Summarize the search results. You MUST cite sources using the exact format <cit>X</cit> where X is the source number."),
        ("human", "Topic: {topic}\n\nSearch Results:\n{results}")
    ])
    
    response = llm.invoke(prompt.format(topic=topic, results=raw_text_for_llm))

    # 4. Return updates (Flaw 5 Solved: Returning standard dict maps to TypedDict)
    return {
        "compressed_research": [f"### Research on: {topic}\n{response.compressed_summary}"],
        "evidence_graph": new_evidence, # Reducer auto-deduplicates!
        "messages": [AIMessage(content=f"Completed research on: {topic}")]
    }

# ==========================================
# NODE 3: THE WRITER (Synthesizer)
# Solves Flaw 4 (Hallucination / Grounding)
# ==========================================
def synthesizer_node(state: AgentState) -> dict:
    """Compiles compressed research into a grounded Markdown report."""
    print("📝 Synthesizer: Writing final report...")
    
    compressed_notes = state.get("compressed_research", [])
    evidence_graph = state.get("evidence_graph", [])
    research_brief = state.get("research_brief", "General deep research.")
    
    if not compressed_notes:
        return {"final_report": "No research data gathered.", "messages": [AIMessage(content="Failed.")]}

    # Format References
    reference_list = "\n".join([f"[{node.citation_index}] {node.title} ({node.url})" for node in evidence_graph])
    all_notes = "\n\n---\n\n".join(compressed_notes)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a comprehensive Markdown report. STRICTLY use <cit>X</cit> for every factual claim. Do not invent citations."),
        ("human", "Brief: {brief}\n\nReferences:\n{references}\n\nNotes:\n{notes}")
    ])
    
    raw_report = llm.invoke(prompt.format(brief=research_brief, references=reference_list, notes=all_notes)).content
    
    # Cryptographic Grounding: Replace <cit>X</cit> with Markdown links [[X]](url)
    url_map = {str(node.citation_index): node.url for node in evidence_graph}
    def replacer(match):
        idx = match.group(1)
        return f"[[{idx}]]({url_map.get(idx, '#')})"
        
    final_report = re.sub(r"<cit>(\d+)</cit>", replacer, raw_report)
    final_report += "\n\n## References\n" + "\n".join([f"- [[{n.citation_index}]]({n.url}) {n.title}" for n in evidence_graph])

    return {
        "final_report": final_report,
        "messages": [AIMessage(content="Final research report generated.")]
    }

# ==========================================
# GRAPH ASSEMBLY & ROUTING
# ==========================================
def route_supervisor(state: AgentState) -> str:
    """
    If the supervisor returned a list of Send() objects, LangGraph automatically 
    routes to the 'researcher' node in parallel. 
    This router ONLY triggers if the supervisor returned a dict (meaning research is done).
    """
    return "synthesizer"

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # Wire Edges
    workflow.add_edge(START, "supervisor")
    
    # If supervisor returns Send(), it goes to researcher. 
    # If it returns dict, it hits this conditional edge and goes to synthesizer.
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"synthesizer": "synthesizer"}
    )
    
    # After parallel researchers finish, they loop back to supervisor for gap analysis
    workflow.add_edge("researcher", "supervisor")
    
    # Synthesizer finishes the graph
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()

# Export the compiled graph
graph = build_graph()
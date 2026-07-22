"""Graph state definitions aligned with deep_researcher.py and Omega Supremacy."""
import operator, re
from typing import Annotated, Optional, Any, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph.message import add_messages

###################
# 1. Pydantic Models (Structured Outputs & Nodes)
###################
class EvidenceNode(BaseModel):
    doc_id: str = Field(default="")
    url: str = Field(default="")
    title: str = Field(default="")
    snippet: str = Field(default="")
    claim: str = Field(default="")
    date_published: Optional[str] = Field(default=None)
    citation_index: int = Field(default=0)
    supports: list[int] = Field(default_factory=list)
    contradicts: list[int] = Field(default_factory=list)

class EvidenceGraphExtraction(BaseModel):
    nodes: List[EvidenceNode] = Field(default_factory=list)

class ResearchNode(BaseModel):
    node_id: str = Field(description="Unique ID (e.g., 'N1', 'N2')")
    topic: str = Field(description="The specific research topic for this node")
    depends_on: list[str] = Field(default_factory=list, description="List of node_ids that must complete before this node can start")

class RouterDecision(BaseModel):
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int
    research_plan: list[ResearchNode] = Field(default_factory=list)

class ConductResearch(BaseModel):
    node_id: str = Field(description="The node_id from the research_plan this task fulfills")
    research_topic: str = Field(description="The topic to research.")

class ResearchComplete(BaseModel):
    pass

class ClarifyWithUser(BaseModel):
    need_clarification: bool
    question: str
    verification: str

class ResearchQuestion(BaseModel):
    research_brief: str
    temporal_intent: str = "Current"
    hard_constraints: list[str] = Field(default_factory=list)

class Summary(BaseModel):
    summary: str
    key_excerpts: str

###################
# 2. Reducers (The Autonomic Nervous System)
###################
def intelligent_memory_reducer(current, new):
    if current is None: current = []
    new_items = new if isinstance(new, list) else [new]
    combined = current + new_items
    unique, seen = [], set()
    for m in combined:
        if not isinstance(m, str): continue
        fp = "".join(sorted(set(re.findall(r'\b\w{5,}\b', m.lower()))))
        if fp and fp not in seen: unique.append(m); seen.add(fp)
    return unique[-15:]

def advanced_evidence_graph_reducer(current, new):
    """Argus Topological Pruning & Temporal Conflict Resolution."""
    if current is None: current = []
    new_nodes = new if isinstance(new, list) else [new]
    claim_map = {}
    for n in current:
        fp = "".join(sorted(re.findall(r'\b\w{4,}\b', getattr(n, 'claim', '').lower())))
        if fp: claim_map[fp] = n
    next_cite = max([getattr(n, 'citation_index', 0) for n in current] + [0]) + 1
    for node in new_nodes:
        fp = "".join(sorted(re.findall(r'\b\w{4,}\b', getattr(node, 'claim', '').lower())))
        if not fp: continue
        if fp in claim_map:
            ex_date = getattr(claim_map[fp], 'date_published', None)
            new_date = getattr(node, 'date_published', None)
            if ex_date and new_date and str(new_date) > str(ex_date): claim_map[fp] = node
        else:
            node.citation_index = next_cite
            claim_map[fp] = node
            next_cite += 1
            
    # TOPOLOGICAL PRUNING (Argus Paradigm)
    support_counts = {n.citation_index: 0 for n in claim_map.values()}
    for n in claim_map.values():
        for s_idx in getattr(n, 'supports', []):
            if s_idx in support_counts: support_counts[s_idx] += 1
            
    final_nodes = []
    for n in claim_map.values():
        is_contradicted_by_stronger = False
        for c_idx in getattr(n, 'contradicts', []):
            if support_counts.get(c_idx, 0) > support_counts.get(n.citation_index, 0):
                is_contradicted_by_stronger = True
                break
        if not is_contradicted_by_stronger:
            final_nodes.append(n)
            
    return final_nodes[-100:] if len(final_nodes) > 100 else final_nodes

def merge_dicts(current: Any, new: Any) -> dict:
    if current is None: current = {}
    if isinstance(new, dict): current.update(new)
    return current

def override_reducer(current_value, new_value):
    if current_value is None: current_value = []
    if isinstance(new_value, dict) and new_value.get("type") == "override": return new_value.get("value", new_value)
    return operator.add(current_value, new_value)

def safe_add(current, new):
    if current is None: current = []
    return operator.add(current, new if isinstance(new, list) else [new])

###################
# 3. TypedDict State Definitions (STRICTLY NO PYDANTIC FIELDS HERE)
###################
class AgentInputState(TypedDict):
    messages: Annotated[list[MessageLikeRepresentation], add_messages]

class AgentState(TypedDict):
    messages: Annotated[list[MessageLikeRepresentation], add_messages]
    research_brief: Optional[str]
    temporal_intent: str
    hard_constraints: list[str]
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    raw_notes: Annotated[list[str], override_reducer]
    notes: Annotated[list[str], override_reducer]
    evidence_graph: Annotated[list[EvidenceNode], advanced_evidence_graph_reducer]
    final_report: str
    research_plan: list[dict]
    completed_nodes: list[str]
    research_artifacts: Annotated[dict, merge_dicts]
    master_synthesis: str
    red_team_findings: str
    devils_advocate_critique: str
    consensus_report: str
    confidence_score: float
    lessons_learned: Annotated[list[str], intelligent_memory_reducer]

class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    temporal_intent: str
    hard_constraints: list[str]
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int
    notes: Annotated[list[str], override_reducer]
    research_iterations: int
    research_plan: list[dict]
    completed_nodes: list[str]
    research_artifacts: Annotated[dict, merge_dicts]
    raw_notes: Annotated[list[str], override_reducer]
    evidence_graph: Annotated[list[EvidenceNode], advanced_evidence_graph_reducer]
    lessons_learned: Annotated[list[str], intelligent_memory_reducer]

class ResearcherState(TypedDict):
    researcher_messages: Annotated[list[MessageLikeRepresentation], safe_add]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    artifact_id: str
    executive_summary: str
    raw_notes: Annotated[list[str], override_reducer]
    evidence_graph: Annotated[list[EvidenceNode], advanced_evidence_graph_reducer]

class ResearcherOutputState(TypedDict):
    compressed_research: str
    artifact_id: str
    executive_summary: str
    raw_notes: list[str]
    evidence_graph: list[EvidenceNode]

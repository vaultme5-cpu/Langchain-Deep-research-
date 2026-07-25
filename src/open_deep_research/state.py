"""Graph state definitions aligned with Omega Supremacy Architecture."""
import operator
import re
import hashlib
from typing import Annotated, Optional, Any, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph.message import add_messages

###################
# 1. Pydantic Models (Strict LLM Output Contracts)
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
    confidence: float = Field(default=0.0)

    @model_validator(mode='before')
    @classmethod
    def sanitize_node(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'url' in data and isinstance(data['url'], str): 
                data['url'] = data['url'].strip()
            if 'citation_index' in data:
                try: data['citation_index'] = int(data['citation_index'])
                except: data['citation_index'] = 0
            for key in ['supports', 'contradicts']:
                if key in data and isinstance(data[key], list):
                    clean = []
                    for item in data[key]:
                        try: clean.append(int(item))
                        except: pass
                    data[key] = clean
        return data

class EvidenceGraphExtraction(BaseModel):
    nodes: List[EvidenceNode] = Field(default_factory=list)

class ResearchNode(BaseModel):
    node_id: str = Field(description="Unique ID (e.g., N1, N2)")
    topic: str = Field(description="The specific research topic for this node")
    depends_on: list[str] = Field(default_factory=list)

class RouterDecision(BaseModel):
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int
    research_plan: list[ResearchNode] = Field(default_factory=list)

class ConductResearch(BaseModel):
    node_id: str = Field(description="The node_id from the research_plan")
    research_topic: str = Field(description="The topic to research.")

class ResearchComplete(BaseModel):
    pass

class ClarifyWithUser(BaseModel):
    need_clarification: bool
    question: str = ""
    verification: str = ""

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

def evidence_key(node):
    source_url = getattr(node, "source_url", "") or getattr(node, "url", "")
    claim = getattr(node, "claim", "")
    date_published = getattr(node, "date_published", None)
    return (
        str(source_url).strip(),
        str(claim).strip().lower(),
        date_published,
    )


def merge_evidence_nodes(current, new):
    merged = {}
    nodes = (current or []) + (new if isinstance(new, list) else [new])
    for node in nodes:
        if not isinstance(node, EvidenceNode):
            continue
        key = evidence_key(node)
        existing = merged.get(key)
        if existing is None:
            merged[key] = node
            continue
        node_conf = float(getattr(node, "confidence", 0.0) or 0.0)
        existing_conf = float(getattr(existing, "confidence", 0.0) or 0.0)
        node_date = getattr(node, "date_published", None)
        existing_date = getattr(existing, "date_published", None)
        if node_conf > existing_conf:
            merged[key] = node
        elif node_conf == existing_conf and node_date and (not existing_date or str(node_date) > str(existing_date)):
            merged[key] = node
    return list(merged.values())[-100:]


def _simhash(text: str) -> str:
    words = re.findall(r'\b\w{4,}\b', text.lower())
    if not words: return ""
    v = [0] * 64
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        for i in range(64):
            if (h >> i) & 1: v[i] += 1
            else: v[i] -= 1
    return "".join('1' if x > 0 else '0' for x in v)

def _hamming(h1: str, h2: str) -> int:
    if not h1 or not h2: return 64
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

def advanced_evidence_graph_reducer(current, new):
    return merge_evidence_nodes(current, new)

def intelligent_memory_reducer(current, new):
    if current is None: current = []
    new_items = new if isinstance(new, list) else [new]
    combined = current + new_items
    unique, seen = [], set()
    for m in combined:
        if not isinstance(m, str): continue
        fp = hashlib.sha256(m.encode()).hexdigest()
        if fp not in seen: 
            unique.append(m)
            seen.add(fp)
    return unique[-15:]

def merge_dicts(current: Any, new: Any) -> dict:
    if current is None: current = {}
    if isinstance(new, dict): current.update(new)
    return current

def override_reducer(current_value, new_value):
    if current_value is None: current_value = []
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        val = new_value.get("value", [])
        return val if isinstance(val, list) else [val]
    if isinstance(new_value, list): return current_value + new_value
    return current_value + [new_value]

def safe_add(current, new):
    if current is None: current = []
    return operator.add(current, new if isinstance(new, list) else [new])

###################
# 3. TypedDict State Definitions (STRICT BOUNDARIES)
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
    virtual_filesystem: Annotated[dict, merge_dicts]
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
    virtual_filesystem: Annotated[dict, merge_dicts]
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

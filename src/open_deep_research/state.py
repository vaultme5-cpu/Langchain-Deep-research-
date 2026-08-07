import operator, re, hashlib
from typing import Annotated, Optional, Any, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, model_validator
from langgraph.graph.message import add_messages

class EvidenceNode(BaseModel):
    doc_id: str = Field(default="")
    url: str = Field(default="")
    title: str = Field(default="")
    claim: str = Field(default="")
    date_published: Optional[str] = Field(default=None)
    citation_index: int = Field(default=0)
    supports: list = Field(default_factory=list)
    contradicts: list = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def sanitize_node(cls, data):
        if isinstance(data, dict):
            if "url" in data and isinstance(data["url"], str): data["url"] = data["url"].strip()
            if "citation_index" in data:
                try: data["citation_index"] = int(data["citation_index"])
                except: data["citation_index"] = 0
        return data

class EvidenceGraphExtraction(BaseModel):
    nodes: List[EvidenceNode] = Field(default_factory=list)

# === PYTHON-SIDE GRAPH MATH (Kills LLM JSON Hallucinations) ===
def compute_epistemic_links(nodes):
    """Deterministically links claims using SimHash. Zero LLM tokens used."""
    def simhash(text):
        words = re.findall(r"\b\w{4,}\b", text.lower())
        if not words: return "0"*64
        v = [0]*64
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            for i in range(64):
                if (h >> i) & 1: v[i] += 1
                else: v[i] -= 1
        return "".join("1" if x > 0 else "0" for x in v)
    hashes = [simhash(n.claim) for n in nodes]
    for i, n1 in enumerate(nodes):
        n1.supports = []
        n1.contradicts = []
        for j, n2 in enumerate(nodes):
            if i == j: continue
            dist = sum(c1 != c2 for c1, c2 in zip(hashes[i], hashes[j]))
            if dist < 10: n1.supports.append(j)
    return nodes

class ResearchNode(BaseModel):
    node_id: str
    topic: str
    depends_on: list = Field(default_factory=list)

class RouterDecision(BaseModel):
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int
    research_plan: list = Field(default_factory=list)

class ConductResearch(BaseModel):
    node_id: str
    research_topic: str

class ResearchComplete(BaseModel): pass

class ClarifyWithUser(BaseModel):
    need_clarification: bool
    question: str = ""
    verification: str = ""

class ResearchQuestion(BaseModel):
    research_brief: str
    temporal_intent: str = "Current"
    hard_constraints: list = Field(default_factory=list)

class Summary(BaseModel):
    summary: str
    key_excerpts: str

# === REDUCERS (The Autonomic Nervous System) ===
def _simhash(text):
    words = re.findall(r"\b\w{4,}\b", text.lower())
    if not words: return ""
    v = [0] * 64
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        for i in range(64):
            if (h >> i) & 1: v[i] += 1
            else: v[i] -= 1
    return "".join("1" if x > 0 else "0" for x in v)

def _hamming(h1, h2):
    if not h1 or not h2: return 64
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

def advanced_evidence_graph_reducer(current, new):
    if current is None: current = []
    new_nodes = new if isinstance(new, list) else [new]
    claim_map = {}
    for n in current: claim_map[hashlib.sha256(n.claim.encode()).hexdigest()] = n
    next_cite = max([n.citation_index for n in current] + [0]) + 1
    for node in new_nodes:
        fp = hashlib.sha256(node.claim.encode()).hexdigest()
        if fp not in claim_map:
            node.citation_index = next_cite
            claim_map[fp] = node
            next_cite += 1
    return list(claim_map.values())[-100:]

def intelligent_memory_reducer(current, new):
    if current is None: current = []
    new_items = new if isinstance(new, list) else [new]
    combined = current + new_items
    unique, seen = [], set()
    for m in combined:
        if not isinstance(m, str): continue
        fp = hashlib.sha256(m.encode()).hexdigest()
        if fp not in seen: unique.append(m); seen.add(fp)
    return unique[-15:]

def merge_dicts(current, new):
    if current is None: current = {}
    if isinstance(new, dict): current.update(new)
    return current

def override_reducer(current, new):
    if isinstance(new, dict) and new.get("type") == "override": return new.get("value", [])
    return (current or []) + (new if isinstance(new, list) else [new])

def safe_add(current, new): return (current or []) + (new if isinstance(new, list) else [new])

# === TYPEDDICT STATES ===
class AgentInputState(TypedDict): messages: Annotated[list, add_messages]

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_brief: Optional[str]
    temporal_intent: str
    hard_constraints: list
    supervisor_messages: Annotated[list, override_reducer]
    notes: Annotated[list, override_reducer]
    evidence_graph: Annotated[list, advanced_evidence_graph_reducer]
    final_report: str
    research_plan: list
    completed_nodes: list
    virtual_filesystem: Annotated[dict, merge_dicts]
    master_synthesis: str
    confidence_score: float

class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list, override_reducer]
    research_brief: str
    temporal_intent: str
    hard_constraints: list
    research_iterations: int
    research_plan: list
    completed_nodes: list
    virtual_filesystem: Annotated[dict, merge_dicts]
    evidence_graph: Annotated[list, advanced_evidence_graph_reducer]
    notes: Annotated[list, override_reducer]

class ResearcherState(TypedDict):
    researcher_messages: Annotated[list, safe_add]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    artifact_id: str
    executive_summary: str
    evidence_graph: Annotated[list, advanced_evidence_graph_reducer]

class ResearcherOutputState(TypedDict):
    compressed_research: str
    artifact_id: str
    executive_summary: str
    evidence_graph: list

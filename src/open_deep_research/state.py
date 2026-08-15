import hashlib
import re
from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from langgraph.graph.message import add_messages

SUPPORT_DISTANCE = 8
CONTRADICTION_DISTANCE = 14
MAX_EVIDENCE_NODES = 120
MAX_SUPPORT_LINKS = 5
MAX_CONTRADICTION_LINKS = 3
NEGATION_TOKENS = {"not", "no", "never", "false", "fails", "failed", "denied", "denies", "without", "lacks", "lack", "cannot"}

def _simhash(text):
    words = re.findall(r"[a-z0-9_]{4,}", str(text or "").lower())
    if not words: return "0" * 64
    v = [0] * 64
    for w in words:
        h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            if (h >> i) & 1: v[i] += 1
            else: v[i] -= 1
    return "".join("1" if x > 0 else "0" for x in v)

def _hamming(h1, h2):
    if not h1 or not h2 or len(h1) != len(h2): return 64
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

_STOPWORDS = {
    "about", "after", "again", "also", "because", "being", "between",
    "could", "from", "have", "into", "more", "most", "other", "over",
    "should", "than", "that", "their", "there", "these", "this", "those",
    "through", "under", "were", "which", "while", "with", "would"
}

def _extract_numbers(text):
    out = set()
    for n in re.findall(r"[0-9]+(?:[.][0-9]+)?", str(text or "")):
        try:
            out.add(float(n))
        except Exception:
            continue
    return out

def _claim_tokens(text):
    words = re.findall(r"[a-z0-9_]{3,}", str(text or "").lower())
    return {w for w in words if w not in _STOPWORDS}

def _has_negation(text):
    words = set(re.findall(r"[a-z]+", str(text or "").lower()))
    return bool(words.intersection(NEGATION_TOKENS))

def _shared_ratio(a, b):
    ta = _claim_tokens(a)
    tb = _claim_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta.intersection(tb)) / max(1, min(len(ta), len(tb)))

def _numbers_conflict(a, b):
    na = _extract_numbers(a)
    nb = _extract_numbers(b)
    if not na or not nb:
        return False
    for x in na:
        for y in nb:
            if abs(x - y) <= max(0.01, abs(x) * 0.01):
                return False
    return True

def _claims_contradict(a, b):
    shared = _shared_ratio(a, b)

    # Never label two weakly-related claims contradictory.
    if shared < 0.60:
        return False

    negation_flip = _has_negation(a) != _has_negation(b)
    numeric_conflict = _numbers_conflict(a, b)

    return negation_flip or (numeric_conflict and shared >= 0.72)

def compute_epistemic_links(nodes):
    """Deterministically links closely related claims while reducing false contradictions."""
    if not nodes:
        return []

    for idx, node in enumerate(nodes, start=1):
        if getattr(node, "citation_index", 0) <= 0:
            node.citation_index = idx
        if not getattr(node, "doc_id", ""):
            node.doc_id = hashlib.sha256(
                str(getattr(node, "claim", "")).encode("utf-8")
            ).hexdigest()[:16]

    hashes = [_simhash(getattr(n, "claim", "")) for n in nodes]

    for i, n1 in enumerate(nodes):
        n1.supports = []
        n1.contradicts = []
        candidates = []

        for j, n2 in enumerate(nodes):
            if i == j:
                continue

            c1 = str(getattr(n1, "claim", ""))
            c2 = str(getattr(n2, "claim", ""))
            shared = _shared_ratio(c1, c2)
            dist = _hamming(hashes[i], hashes[j])
            target = getattr(n2, "citation_index", j + 1)

            if dist <= CONTRADICTION_DISTANCE and _claims_contradict(c1, c2):
                candidates.append((dist, target, 1))
            elif dist <= SUPPORT_DISTANCE and shared >= 0.72:
                if not _numbers_conflict(c1, c2):
                    candidates.append((dist, target, 0))

        candidates.sort(key=lambda x: (x[0], x[1]))

        s_count = 0
        c_count = 0

        for _, target, kind in candidates:
            if kind == 0 and s_count < MAX_SUPPORT_LINKS:
                if target not in n1.supports:
                    n1.supports.append(target)
                    s_count += 1
            elif kind == 1 and c_count < MAX_CONTRADICTION_LINKS:
                if target not in n1.contradicts:
                    n1.contradicts.append(target)
                    c_count += 1

    return nodes

class EvidenceNode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    doc_id: str = ""
    url: str = ""
    title: str = ""
    claim: str = ""
    date_published: Optional[str] = None

    evidence_span: str = ""
    source_kind: str = "unknown"
    verification_status: str = "UNVERIFIED"
    entailment_score: float = 0.0
    provenance_id: str = ""

    evidence_span: str = ""
    source_kind: str = "unknown"
    verification_status: str = "UNVERIFIED"
    entailment_score: float = 0.0
    provenance_id: str = ""
    citation_index: int = 0
    supports: List[int] = Field(default_factory=list)
    contradicts: List[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def sanitize_node(cls, data):
        if not isinstance(data, dict): return data
        for key in ["url", "title", "claim", "date_published"]:
            if key in data and isinstance(data[key], str): data[key] = data[key].strip()
        if "citation_index" in data:
            try: data["citation_index"] = int(data["citation_index"])
            except Exception: data["citation_index"] = 0
        for key in ["supports", "contradicts"]:
            val = data.get(key)
            if not isinstance(val, list): data[key] = []
            else:
                clean = []
                for item in val:
                    try: clean.append(int(item))
                    except Exception: continue
                data[key] = clean
        return data

class EvidenceGraphExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    nodes: List[EvidenceNode] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_list(cls, data):
        if isinstance(data, list): return {"nodes": data}
        if isinstance(data, dict) and "nodes" not in data:
            for value in data.values():
                if isinstance(value, list): return {"nodes": value}
        return data

class ResearchNode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    node_id: str = ""
    topic: str = ""
    depends_on: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_id(self):
        if not self.node_id: self.node_id = hashlib.sha256(self.topic.encode("utf-8")).hexdigest()[:10]
        return self

class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query_paradigm: str = "General"
    complexity_tier: str = "Medium"
    dynamic_tool_budget: int = 6
    dynamic_research_units: int = 2
    research_plan: List[ResearchNode] = Field(default_factory=list)

    @field_validator("dynamic_tool_budget", mode="before")
    @classmethod
    def cast_budget(cls, v):
        try: return max(1, min(10, int(v)))
        except Exception: return 6

    @field_validator("dynamic_research_units", mode="before")
    @classmethod
    def cast_units(cls, v):
        try: return max(1, min(3, int(v)))
        except Exception: return 2

class ConductResearch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    node_id: str = ""
    research_topic: str = ""

class ResearchComplete(BaseModel):
    pass

class ClarifyWithUser(BaseModel):
    model_config = ConfigDict(extra="ignore")
    need_clarification: bool = False
    question: str = ""
    verification: str = ""

    @field_validator("need_clarification", mode="before")
    @classmethod
    def cast_bool(cls, v):
        if isinstance(v, str): return v.strip().lower() in {"1", "true", "yes", "on"}
        return bool(v)

class ResearchQuestion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    research_brief: str = ""
    temporal_intent: str = "Current"
    hard_constraints: List[str] = Field(default_factory=list)

    @field_validator("hard_constraints", mode="before")
    @classmethod
    def cast_constraints(cls, v):
        if isinstance(v, str): return [v.strip()] if v.strip() else []
        if isinstance(v, list): return [str(x).strip() for x in v if str(x).strip()]
        return []

class Summary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    summary: str = ""
    key_excerpts: str = ""


class ReportSection(BaseModel):
    heading: str = ""
    content: str = ""
    evidence_ids: List[int] = Field(
        default_factory=list
    )


class FinalReportArtifact(BaseModel):
    title: str = ""
    executive_summary: str = ""
    executive_evidence_ids: List[int] = Field(
        default_factory=list
    )
    sections: List[ReportSection] = Field(
        default_factory=list
    )
    key_uncertainties: List[str] = Field(
        default_factory=list
    )
    watchlist: List[str] = Field(
        default_factory=list
    )

def advanced_evidence_graph_reducer(current, new):
    if current is None: current = []
    if new is None: return current
    new_nodes = new if isinstance(new, list) else [new]
    claim_map = {}
    for n in current:
        claim = str(getattr(n, "claim", "")).strip()
        if claim: claim_map[hashlib.sha256(claim.encode("utf-8")).hexdigest()] = n
    next_cite = max([getattr(n, "citation_index", 0) for n in current] + [0]) + 1
    for node in new_nodes:
        if not isinstance(node, EvidenceNode):
            try: node = EvidenceNode.model_validate(node)
            except Exception: continue
        claim = str(getattr(node, "claim", "")).strip()
        if not claim: continue
        fp = hashlib.sha256(claim.encode("utf-8")).hexdigest()
        if fp not in claim_map:
            node.citation_index = next_cite
            if not getattr(node, "doc_id", ""): node.doc_id = fp[:16]
            claim_map[fp] = node
            next_cite += 1
    return compute_epistemic_links(list(claim_map.values())[-MAX_EVIDENCE_NODES:])

def intelligent_memory_reducer(current, new):
    if current is None: current = []
    if new is None: return current
    new_items = new if isinstance(new, list) else [new]
    combined = list(current) + [str(x).strip() for x in new_items if str(x).strip()]
    unique, seen = [], set()
    for m in combined:
        fp = hashlib.sha256(m.encode("utf-8")).hexdigest()
        if fp not in seen:
            unique.append(m)
            seen.add(fp)
    return unique[-20:]

def merge_dicts(current, new):
    if current is None: current = {}
    if new is None: return current
    if isinstance(new, dict): current.update(new)
    return current

def override_reducer(current, new):
    if current is None: current = []
    if new is None: return current
    if isinstance(new, dict) and new.get("type") == "override":
        value = new.get("value", [])
        return value if isinstance(value, list) else [value]
    if isinstance(new, list): return list(current) + new
    return list(current) + [new]

def safe_add(current, new):
    if current is None: current = []
    if new is None: return current
    if isinstance(new, list): return list(current) + new
    return list(current) + [new]

class AgentInputState(TypedDict, total=False):
    messages: Annotated[list, add_messages]

class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    research_brief: str
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
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int
    research_iterations: int
    lessons_learned: Annotated[list, intelligent_memory_reducer]
    red_team_findings: str
    devils_advocate_critique: str
    consensus_report: str
    erc_frontier_fingerprint: str
    erc_no_progress_count: int
    erc_last_plan_hash: str
    erc_last_evidence_hash: str
    erc_last_completed_hash: str

class SupervisorState(TypedDict, total=False):
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
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int
    lessons_learned: Annotated[list, intelligent_memory_reducer]
    erc_frontier_fingerprint: str
    erc_no_progress_count: int
    erc_last_plan_hash: str
    erc_last_evidence_hash: str
    erc_last_completed_hash: str

class ResearcherState(TypedDict, total=False):
    researcher_messages: Annotated[list, safe_add]
    tool_call_iterations: int
    research_topic: str
    temporal_intent: str
    hard_constraints: list
    compressed_research: str
    artifact_id: str
    executive_summary: str
    evidence_graph: Annotated[list, advanced_evidence_graph_reducer]

class ResearcherOutputState(TypedDict, total=False):
    compressed_research: str
    artifact_id: str
    executive_summary: str
    evidence_graph: list

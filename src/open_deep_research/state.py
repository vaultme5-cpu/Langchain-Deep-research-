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

# ============================================================
# I14.11: ENUMERATED EPISTEMIC STATES
# No arbitrary string statuses. Invalid state = normalized.
# ============================================================
VERIFICATION_STATUSES = frozenset({
    "UNVERIFIED", "CLEAR_SUPPORT", "PARTIAL_SUPPORT",
    "CONTRADICTORY", "UNSUPPORTED", "AMBIGUOUS", "QUARANTINED"
})

SOURCE_KINDS = frozenset({
    "PRIMARY", "OFFICIAL", "RESEARCH", "TECHNICAL",
    "SECONDARY", "COMMENTARY", "UNKNOWN"
})

_VERIFICATION_STATUS_ALIASES = {
    "VERIFIED": "CLEAR_SUPPORT",
    "SUPPORTED": "CLEAR_SUPPORT",
    "CONFIRMED": "CLEAR_SUPPORT",
    "PARTIAL": "PARTIAL_SUPPORT",
    "WEAK": "PARTIAL_SUPPORT",
    "CONTRADICTED": "CONTRADICTORY",
    "CONTRADICTION": "CONTRADICTORY",
    "NOT_SUPPORTED": "UNSUPPORTED",
    "REFUTED": "UNSUPPORTED",
    "UNCLEAR": "AMBIGUOUS",
    "UNCERTAIN": "AMBIGUOUS",
    "QUARANTINE": "QUARANTINED",
    "BLOCKED": "QUARANTINED",
}

_SOURCE_KIND_ALIASES = {
    "GOVERNMENT": "OFFICIAL",
    "ACADEMIC": "RESEARCH",
    "MAJOR_MEDIA": "SECONDARY",
    "GENERAL": "COMMENTARY",
    "NEWS": "SECONDARY",
    "BLOG": "COMMENTARY",
}

def _normalize_verification_status(value):
    """I14.11: Normalize arbitrary string to valid VerificationStatus."""
    v = str(value or "").strip().upper()
    if v in VERIFICATION_STATUSES:
        return v
    if v in _VERIFICATION_STATUS_ALIASES:
        return _VERIFICATION_STATUS_ALIASES[v]
    return "UNVERIFIED"

def _normalize_source_kind(value):
    """I14.11: Normalize arbitrary string to valid SourceKind."""
    v = str(value or "").strip().upper()
    if v in SOURCE_KINDS:
        return v
    if v in _SOURCE_KIND_ALIASES:
        return _SOURCE_KIND_ALIASES[v]
    return "UNKNOWN"

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
    source_result_id: str = ""
    retrieval_timestamp: float = 0.0
    evidence_hash: str = ""

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


    @field_validator("verification_status", mode="before")
    @classmethod
    def normalize_verification_status(cls, v):
        """I14.11: Ensure verification_status is a valid enum value."""
        return _normalize_verification_status(v)

    @field_validator("source_kind", mode="before")
    @classmethod
    def normalize_source_kind(cls, v):
        """I14.11: Ensure source_kind is a valid enum value."""
        return _normalize_source_kind(v)


    def validate_provenance_complete(self):
        """I18.8: Validate all provenance fields are present and valid.
        Returns (is_valid, reason). A claim can enter a normal final report
        ONLY IF all provenance fields are valid.
        """
        srid = str(getattr(self, "source_result_id", "") or "").strip()
        if not srid or srid == "unknown_artifact":
            return False, "UNTRACEABLE:no_source_result_id"
        span = str(getattr(self, "evidence_span", "") or "").strip()
        if not span:
            return False, "UNTRACEABLE:no_evidence_span"
        ehash = str(getattr(self, "evidence_hash", "") or "").strip()
        if not ehash:
            return False, "UNTRACEABLE:no_evidence_hash"
        prov_id = str(getattr(self, "provenance_id", "") or "").strip()
        if not prov_id:
            return False, "UNTRACEABLE:no_provenance_id"
        return True, "TRACEABLE"

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

# ============================================================
# I17.8: STRICT ROUTER + DAG CONTRACT
# Invalid router output = HARD VALIDATION FAILURE.
# No silent coercion of malformed planning into defaults.
# ============================================================
_I17_8_VALID_COMPLEXITY_TIERS = frozenset({"Simple", "Medium", "Complex", "Expert"})
_I17_8_VALID_QUERY_PARADIGMS = frozenset({"Technical", "Financial", "Scientific", "General"})
_I17_8_MAX_PLAN_SIZE = 15
_I17_8_MAX_DEPENDENCIES_PER_NODE = 5

class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query_paradigm: str = "General"
    complexity_tier: str = "Medium"
    dynamic_tool_budget: int = 6
    dynamic_research_units: int = 2
    research_plan: List[ResearchNode] = Field(default_factory=list)

    @field_validator("query_paradigm", mode="before")
    @classmethod
    def i17_8_validate_paradigm(cls, v):
        v = str(v or "").strip()
        if v not in _I17_8_VALID_QUERY_PARADIGMS:
            raise ValueError(
                "I17.8: invalid query_paradigm '" + v
                + "'. Must be one of: " + str(sorted(_I17_8_VALID_QUERY_PARADIGMS))
            )
        return v

    @field_validator("complexity_tier", mode="before")
    @classmethod
    def i17_8_validate_tier(cls, v):
        v = str(v or "").strip()
        if v not in _I17_8_VALID_COMPLEXITY_TIERS:
            raise ValueError(
                "I17.8: invalid complexity_tier '" + v
                + "'. Must be one of: " + str(sorted(_I17_8_VALID_COMPLEXITY_TIERS))
            )
        return v

    @field_validator("dynamic_tool_budget", mode="before")
    @classmethod
    def i17_8_validate_budget(cls, v):
        try:
            val = int(v)
        except Exception:
            raise ValueError("I17.8: dynamic_tool_budget must be an integer, got '" + str(v) + "'")
        if not 1 <= val <= 10:
            raise ValueError("I17.8: dynamic_tool_budget must be 1..10, got " + str(val))
        return val

    @field_validator("dynamic_research_units", mode="before")
    @classmethod
    def i17_8_validate_units(cls, v):
        try:
            val = int(v)
        except Exception:
            raise ValueError("I17.8: dynamic_research_units must be an integer, got '" + str(v) + "'")
        if not 1 <= val <= 3:
            raise ValueError("I17.8: dynamic_research_units must be 1..3, got " + str(val))
        return val

    @model_validator(mode="after")
    def i17_8_validate_research_plan(self):
        """I17.8: Strict DAG validation. HARD FAILURE on any violation."""
        plan = self.research_plan
        if not plan:
            return self

        # Rule 8: max plan size bounded
        if len(plan) > _I17_8_MAX_PLAN_SIZE:
            raise ValueError(
                "I17.8: research_plan has " + str(len(plan))
                + " nodes, max is " + str(_I17_8_MAX_PLAN_SIZE)
            )

        # Rule 1: unique node_id
        node_ids = []
        for node in plan:
            nid = str(getattr(node, "node_id", "") or "").strip()
            if not nid:
                raise ValueError("I17.8: research_plan node has empty node_id")
            node_ids.append(nid)
        if len(node_ids) != len(set(node_ids)):
            seen = set()
            dupes = set()
            for nid in node_ids:
                if nid in seen:
                    dupes.add(nid)
                seen.add(nid)
            raise ValueError("I17.8: duplicate node_ids in research_plan: " + str(sorted(dupes)))

        # Rule 2: non-empty topic
        for node in plan:
            topic = str(getattr(node, "topic", "") or "").strip()
            if not topic:
                raise ValueError(
                    "I17.8: node '" + str(getattr(node, "node_id", "?")) + "' has empty topic"
                )

        # Build adjacency for DAG checks
        node_id_set = set(node_ids)
        deps_map = {}
        for node in plan:
            nid = str(getattr(node, "node_id", "") or "").strip()
            deps = [str(d or "").strip() for d in (getattr(node, "depends_on", []) or [])]
            deps_map[nid] = deps

        for nid, deps in deps_map.items():
            # Rule 9: dependency count bounded
            if len(deps) > _I17_8_MAX_DEPENDENCIES_PER_NODE:
                raise ValueError(
                    "I17.8: node '" + nid + "' has " + str(len(deps))
                    + " dependencies, max is " + str(_I17_8_MAX_DEPENDENCIES_PER_NODE)
                )
            for dep in deps:
                # Rule 3: no self dependency
                if dep == nid:
                    raise ValueError("I17.8: node '" + nid + "' has self-dependency")
                # Rule 4: every dependency exists
                if dep not in node_id_set:
                    raise ValueError(
                        "I17.8: node '" + nid + "' depends on unknown node '" + dep + "'"
                    )

        # Build reverse adjacency for topological sort
        dependents = {nid: [] for nid in node_ids}
        for nid, deps in deps_map.items():
            for dep in deps:
                dependents[dep].append(nid)

        # Rule 5: no cycles (Kahn's algorithm)
        in_degree = {nid: len(deps_map[nid]) for nid in node_ids}
        queue = [nid for nid in node_ids if in_degree[nid] == 0]
        visited_count = 0
        while queue:
            current = queue.pop(0)
            visited_count += 1
            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if visited_count != len(node_ids):
            raise ValueError("I17.8: research_plan contains a cycle")

        # Rule 6: at least one root (node with no dependencies)
        roots = [nid for nid in node_ids if len(deps_map.get(nid, [])) == 0]
        if not roots:
            raise ValueError(
                "I17.8: research_plan has no root node (all nodes have dependencies)"
            )

        # Rule 7: all nodes reachable from roots
        # (proven by topological sort visiting all nodes above)

        return self


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



    @field_validator('evidence_ids', mode='before')
    @classmethod
    def i17_9_validate_evidence_ids(cls, v):
        """I17.9: No negative IDs. No zero IDs."""
        if not isinstance(v, list):
            return []
        clean = []
        for item in v:
            try:
                val = int(item)
            except Exception:
                continue
            if val <= 0:
                raise ValueError(
                    "I17.9: evidence_id must be positive, got "
                    + str(val)
                )
            clean.append(val)
        return clean

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


    @field_validator('executive_evidence_ids', mode='before')
    @classmethod
    def i17_9_validate_executive_ids(cls, v):
        """I17.9: No negative IDs. No zero IDs."""
        if not isinstance(v, list):
            return []
        clean = []
        for item in v:
            try:
                val = int(item)
            except Exception:
                continue
            if val <= 0:
                raise ValueError(
                    "I17.9: executive_evidence_id must be positive, got "
                    + str(val)
                )
            clean.append(val)
        return clean

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
    confidence_breakdown: dict
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
    research_frontier: list
    reasoning_depth_signal: float
    research_status: str

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
    research_frontier: list
    reasoning_depth_signal: float
    research_status: str

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

# ============================================================
# I15.5: STRICT SOURCE ARTIFACT PROVENANCE
# ============================================================
class SourceArtifact(BaseModel):
    """I15.5: A canonical retrieved source artifact. Every source retrieval
    creates one; EvidenceNode references it via source_result_id."""
    model_config = ConfigDict(extra="ignore")
    source_result_id: str = ""
    run_id: str = ""
    canonical_url: str = ""
    retrieved_at: float = 0.0
    raw_content_hash: str = ""
    normalized_content_hash: str = ""
    source_status: str = "RETRIEVED"



# ============================================================
# I17.10: NATIVE ToolResult CONTRACT
# Every tool returns a typed ToolResult. FAILED never evidence.
# QUARANTINED never trusted. No tool returns a bare error string
# as its canonical object.
# ============================================================
_I17_10_TOOL_STATUSES = ("SUCCESS", "DEGRADED", "FAILED", "QUARANTINED")

class ToolResult(BaseModel):
    """I17.10: Canonical typed tool result contract."""
    model_config = ConfigDict(extra="ignore")
    status: str = "FAILED"
    source: str = "unknown"
    content: str = ""
    request_id: str = ""
    retrieved_at: float = 0.0
    error_class: Optional[str] = None
    source_result_id: Optional[str] = None
    final_url: Optional[str] = None

    @field_validator("status", mode="before")
    @classmethod
    def i17_10_validate_status(cls, v):
        v = str(v or "").strip().upper()
        if v not in _I17_10_TOOL_STATUSES:
            return "FAILED"
        return v

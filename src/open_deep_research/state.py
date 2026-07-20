"""Graph state definitions aligned with deep_researcher.py and Master Doc."""
import operator
from typing import Annotated, Optional, Any, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph.message import add_messages

###################
# 1. The Knowledge Graph Node (Sector 3 & 4)
###################
class EvidenceNode(BaseModel):
    """Represents a verified source in the evidence graph."""
    doc_id: str = Field(default="")
    url: str = Field(default="")
    title: str = Field(default="")
    snippet: str = Field(default="")
    # SECTOR 4: Epistemic Verification Fields
    claim: str = Field(default="", description="Core factual claim for CMI satiation checks")
    date_published: Optional[str] = Field(default=None, description="YYYY-MM-DD for temporal grounding")
    citation_index: int = Field(default=0)

class EvidenceGraphExtraction(BaseModel):
    """Structured output for compressing research into an Evidence Graph."""
    nodes: List[EvidenceNode] = Field(default_factory=list)

###################
# 2. Structured Outputs
###################
class ConductResearch(BaseModel):
    research_topic: str = Field(description="The topic to research.")

class ResearchComplete(BaseModel):
    pass

class ClarifyWithUser(BaseModel):
    need_clarification: bool
    question: str
    verification: str

class ResearchQuestion(BaseModel):
    research_brief: str

###################
# 3. Advanced Reducers
###################
def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if current_value is None: current_value = []
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

def safe_add(current: Any, new: Any) -> list:
    if current is None: current = []
    return operator.add(current, new if isinstance(new, list) else [new])

def evidence_graph_reducer(current: Any, new: Any) -> list[EvidenceNode]:
    """Auto-deduplicates sources and assigns global citation indexes."""
    if current is None: current = []
    existing_ids = {node.doc_id for node in current if getattr(node, 'doc_id', None)}
    new_nodes = new if isinstance(new, list) else [new]
    next_citation = len(current) + 1
    for node in new_nodes:
        doc_id = getattr(node, 'doc_id', None)
        if doc_id and doc_id not in existing_ids:
            node.citation_index = next_citation
            current.append(node)
            existing_ids.add(doc_id)
            next_citation += 1
        elif not doc_id:
            current.append(node)
    return current

###################
# 4. State Definitions
###################
class AgentInputState(TypedDict):
    messages: Annotated[list[MessageLikeRepresentation], add_messages]

class AgentState(TypedDict):
    messages: Annotated[list[MessageLikeRepresentation], add_messages]
    research_brief: Optional[str]
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    raw_notes: Annotated[list[str], override_reducer]
    notes: Annotated[list[str], override_reducer]
    evidence_graph: Annotated[list[EvidenceNode], evidence_graph_reducer]
    final_report: str

class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    research_iterations: int
    raw_notes: Annotated[list[str], override_reducer]
    evidence_graph: Annotated[list[EvidenceNode], evidence_graph_reducer]

class ResearcherState(TypedDict):
    researcher_messages: Annotated[list[MessageLikeRepresentation], safe_add]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer]
    evidence_graph: Annotated[list[EvidenceNode], evidence_graph_reducer]

class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: list[str]
    evidence_graph: list[EvidenceNode]

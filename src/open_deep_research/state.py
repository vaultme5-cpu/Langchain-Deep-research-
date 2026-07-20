"""Graph state definitions and data structures for the Deep Research Swarm."""
import operator
from typing import Annotated, Optional, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph.message import add_messages

###################
# 1. The Knowledge Graph Node
###################
class EvidenceNode(BaseModel):
    """Represents a verified source in the evidence graph."""
    doc_id: str = Field(description="Unique hash of the URL to prevent duplicates")
    url: str
    title: str
    snippet: str
    citation_index: int = Field(description="Global citation index e.g., 1 for [1]")

###################
# 2. Structured Outputs (Enforcing Grounding)
###################
class ConductResearch(BaseModel):
    """Call this tool to spawn a researcher for a specific topic."""
    research_topic: str = Field(
        description="Highly detailed topic description. Include specific entities, timeframes, and questions to answer.",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""
    is_complete: bool = Field(description="True if all knowledge gaps are filled.")
    final_reasoning: str = Field(description="Brief explanation of why the gathered data is sufficient.")

class Summary(BaseModel):
    """Research summary with strict citation enforcement."""
    summary: str = Field(description="Detailed summary. MUST use <cit>X</cit> tags to cite sources.")
    key_excerpts: list[str] = Field(description="Key quotes or data points with <cit>X</cit> tags.")
    citation_indices: list[int] = Field(description="List of integer citation indices used in this summary.")

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    need_clarification: bool
    question: str = Field(description="A targeted question to clarify the report scope.")
    verification: str = Field(description="Message confirming we will start research after clarification.")

class ResearchQuestion(BaseModel):
    """Initial research strategy."""
    research_brief: str = Field(description="The core strategy and research question.")
    initial_gaps: list[str] = Field(description="Potential missing info or edge cases to look out for.")

class IdentifyGaps(BaseModel):
    """Supervisor uses this to perform autonomous gap analysis."""
    covered_topics: list[str] = Field(description="What we have successfully learned.")
    missing_gaps: list[str] = Field(description="What is still missing, contradictory, or unclear.")
    next_action: str = Field(description="Either 'research_more' or 'synthesize_report'", enum=["research_more", "synthesize_report"])

###################
# 3. Advanced Reducers (The Engine)
###################
def safe_add(current: Any, new: Any) -> list:
    """Safely appends lists, handling None initialization."""
    if current is None:
        current = []
    return operator.add(current, new if isinstance(new, list) else [new])

def evidence_graph_reducer(current: Any, new: Any) -> list[EvidenceNode]:
    """Auto-deduplicates sources and assigns global citation indexes."""
    if current is None:
        current = []
        
    existing_ids = {node.doc_id for node in current}
    new_nodes = new if isinstance(new, list) else [new]
    
    # Auto-assign citation index based on current graph size
    next_citation = len(current) + 1
    
    for node in new_nodes:
        if node.doc_id not in existing_ids:
            node.citation_index = next_citation
            current.append(node)
            existing_ids.add(node.doc_id)
            next_citation += 1
            
    return current

###################
# 4. State Definitions (Pure TypedDicts for Flawless Routing)
###################
class AgentInputState(TypedDict):
    """Entry point state."""
    messages: Annotated[list[MessageLikeRepresentation], add_messages]

class AgentState(TypedDict):
    """Main global state. Kept on a strict data diet to prevent context overflow."""
    messages: Annotated[list[MessageLikeRepresentation], add_messages]
    research_brief: str
    
    # Lightweight compressed data only (no heavy raw_notes here!)
    compressed_research: Annotated[list[str], safe_add]
    
    # The auto-deduplicating knowledge graph
    evidence_graph: Annotated[list[EvidenceNode], evidence_graph_reducer]
    
    # Autonomous tracking
    research_log: Annotated[list[str], safe_add] 
    knowledge_gaps: Annotated[list[str], safe_add]
    
    final_report: str

class SupervisorState(TypedDict):
    """State for the orchestrator."""
    messages: Annotated[list[MessageLikeRepresentation], add_messages]
    research_brief: str
    research_log: Annotated[list[str], safe_add]
    knowledge_gaps: Annotated[list[str], safe_add]
    evidence_graph: Annotated[list[EvidenceNode], evidence_graph_reducer]
    research_iterations: int

class ResearcherState(TypedDict):
    """State for individual parallel researchers."""
    messages: Annotated[list[MessageLikeRepresentation], add_messages]
    research_topic: str
    tool_call_iterations: int
    compressed_research: str
    evidence_graph: Annotated[list[EvidenceNode], evidence_graph_reducer]

class ResearcherOutputState(TypedDict):
    """Output mapping back to the main graph."""
    compressed_research: list[str]
    evidence_graph: list[EvidenceNode]
    messages: list[MessageLikeRepresentation]
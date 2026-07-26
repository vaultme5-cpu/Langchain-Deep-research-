"""State definitions for Open Deep Research."""
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class ResearchNode(BaseModel):
    """Core research evidence node."""
    claim: str = Field(description="The factual claim or finding")
    url: str = Field(default="", description="Source URL")
    date_published: Optional[str] = Field(default=None, description="Publication date")
    confidence: float = Field(default=0.5, description="Confidence score 0-1")
    contradictions: List[str] = Field(default_factory=list, description="Contradicting claims")

class AgentState(TypedDict):
    """Primary agent state."""
    research_brief: str
    evidence_graph: List[ResearchNode]
    virtual_filesystem: Dict[str, str]
    confidence_score: float
    consensus_report: str
    master_synthesis: str
    temporal_intent: str
    research_budget: str
    final_report: str
    messages: Annotated[list, add_messages]

class ResearcherState(AgentState):
    """Specialized researcher subgraph state."""
    current_query: str
    search_results: List[Dict[str, Any]]
    raw_notes: List[str]
    compressed_research: str
    artifact_id: str
    executive_summary: str

class SupervisorState(TypedDict):
    """Supervisor orchestration state."""
    task_queue: List[str]
    active_agents: List[str]
    progress_tracker: Dict[str, float]
    resource_allocation: Dict[str, int]
    research_status: Literal["planning", "executing", "verifying", "complete"]
    next_action: str
    error_log: List[str]

class Configuration(TypedDict):
    """Runtime configuration."""
    model: str
    max_tokens: int
    api_key: str
    thread_id: str
    research_budget: Literal["Fast", "Balanced", "Comprehensive"]
    temporal_intent: Literal["Current", "Historical", "Predictive"]

class AgentInputState(TypedDict):
    """Entry point input state."""
    initial_query: str
    research_intent: str
    custom_instructions: str

class ResearcherOutputState(TypedDict):
    """Researcher subgraph output."""
    findings: List[Dict[str, Any]]
    confidence_score: float
    source_quality_metrics: Dict[str, float]
    artifact_paths: List[str]

class ClarifyWithUser(BaseModel):
    need_clarification: bool
    question: str
    verification: str

class ResearchQuestion(BaseModel):
    research_brief: str
    temporal_intent: str
    hard_constraints: List[str]
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int

class RouterDecision(BaseModel):
    next_step: str
    query_paradigm: str
    complexity_tier: str
    dynamic_tool_budget: int
    dynamic_research_units: int

class ConductResearch(BaseModel):
    query: str
    depth: int

class EvidenceGraphExtraction(BaseModel):
    research_nodes: List[ResearchNode]

class ResearchComplete(BaseModel):
    final_report: str
    confidence_score: float

class Summary(BaseModel):
    summary: str
    key_points: List[str]

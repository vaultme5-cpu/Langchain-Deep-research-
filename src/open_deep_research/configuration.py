"""Configuration definitions for Open Deep Research."""
from typing import Literal, TypedDict

TEMPORAL_INTENT_OPTIONS = ["Current", "Historical", "Predictive"]
RESEARCH_BUDGET_OPTIONS = ["Fast", "Balanced", "Comprehensive"]

class Configuration(TypedDict):
    """Runtime configuration schema."""
    model: str
    max_tokens: int
    api_key: str
    thread_id: str
    research_budget: Literal["Fast", "Balanced", "Comprehensive"]
    temporal_intent: Literal["Current", "Historical", "Predictive"]
    research_model: str
    research_model_max_tokens: int
    mcp_prompt: str
    max_concurrent_research_units: int
    max_researcher_iterations: int

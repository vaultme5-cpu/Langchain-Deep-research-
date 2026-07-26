"""Configuration management for the Open Deep Research system."""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class SearchAPI(Enum):
    TAVILY = "tavily"
    SEARXNG = "searxng"
    NONE = "none"

class MCPConfig(BaseModel):
    url: Optional[str] = Field(default=None)
    tools: Optional[List[str]] = Field(default=None)
    auth_required: Optional[bool] = Field(default=False)

class Configuration(BaseModel):
    max_structured_output_retries: int = Field(default=3)
    research_model: str = Field(default="groq:llama-3.3-70b-versatile")
    research_model_max_tokens: int = Field(default=8192)
    compression_model: str = Field(default="groq:llama-3.3-70b-versatile")
    compression_model_max_tokens: int = Field(default=4096)
    mcp_prompt: Optional[str] = Field(default=None)
    max_concurrent_research_units: int = Field(default=3)
    max_researcher_iterations: int = Field(default=15)
    allow_clarification: bool = Field(default=True)
    search_api: SearchAPI = Field(default=SearchAPI.SEARXNG)
    temporal_intent: str = Field(default="Current")
    research_budget: str = Field(default="Balanced")
    max_react_tool_calls: int = Field(default=10)

"""Configuration management for the Open Deep Research system."""
import os
from enum import Enum
from typing import Any, List, Optional
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    SEARXNG = "searxng"  # <-- SECTOR 1 UNLOCKED
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    url: Optional[str] = Field(default=None, optional=True)
    tools: Optional[List[str]] = Field(default=None, optional=True)
    auth_required: Optional[bool] = Field(default=False, optional=True)

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=5)
    
    # SECTOR 1: SearXNG Config
    searxng_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for your self-hosted SearXNG instance."
    )
    
    # Default to Free SearXNG
    search_api: SearchAPI = Field(
        default=SearchAPI.SEARXNG,
        metadata={"x_oap_ui_config": {"type": "select", "default": "searxng"}}
    )
    max_researcher_iterations: int = Field(default=6)
    max_react_tool_calls: int = Field(default=10)
    
    # SECTOR 5: Decoupled Free LLMs (Gemini)
    # Fast/Cheap model for summarization and exploration
    summarization_model: str = Field(default="google_genai:gemini-2.5-pro")
    summarization_model_max_tokens: int = Field(default=8192)
    max_content_length: int = Field(default=50000)
    
    # Heavy reasoning model for research planning
    research_model: str = Field(default="google_genai:gemini-2.5-pro")
    research_model_max_tokens: int = Field(default=10000)
    
    # Fast model for compressing evidence graphs
    compression_model: str = Field(default="google_genai:gemini-2.5-pro")
    compression_model_max_tokens: int = Field(default=8192)
    
    # Heavy logic model for final report synthesis
    final_report_model: str = Field(default="google_genai:gemini-2.5-pro")
    final_report_model_max_tokens: int = Field(default=10000)

    mcp_config: Optional[MCPConfig] = Field(default=None, optional=True)
    mcp_prompt: Optional[str] = Field(default=None, optional=True)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        arbitrary_types_allowed = True

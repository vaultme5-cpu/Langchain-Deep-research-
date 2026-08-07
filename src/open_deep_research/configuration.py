"""Configuration management for the Omega Supremacy Engine (Path B: Purist Route)."""
import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.runnables import RunnableConfig

class SearchAPI(Enum):
    JINA = "jina"
    SEARXNG = "searxng"
    NONE = "none"

class MCPConfig(BaseModel):
    url: Optional[str] = Field(default=None)
    tools: Optional[List[str]] = Field(default=None)
    auth_required: Optional[bool] = Field(default=False)

class Configuration(BaseModel):
    # Core Execution Parameters
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=3)
    max_researcher_iterations: int = Field(default=6)
    max_react_tool_calls: int = Field(default=10)
    erc_max_stagnation_iterations: int = Field(default=3)
    
    # Search & Ingestion
    search_api: SearchAPI = Field(default=SearchAPI.JINA)
    searxng_base_url: str = Field(default="http://localhost:8080")
    max_content_length: int = Field(default=12000)
    
    # LLM Routing (THE PURIST ROUTE: 100% Groq Supremacy)
    summarization_model: str = Field(default="groq:llama-3.1-8b-instant")
    summarization_model_max_tokens: int = Field(default=4096)
    research_model: str = Field(default="groq:llama-3.3-70b-versatile")
    research_model_max_tokens: int = Field(default=8192)
    # UPGRADE: Compressor moved from fragile 8B to Titan 70B
    compression_model: str = Field(default="groq:llama-3.3-70b-versatile")
    compression_model_max_tokens: int = Field(default=8192)
    final_report_model: str = Field(default="groq:llama-3.3-70b-versatile")
    final_report_model_max_tokens: int = Field(default=8192)
    
    # MCP & Extensions
    mcp_config: Optional[MCPConfig] = Field(default=None)
    mcp_prompt: Optional[str] = Field(default=None)

    @field_validator(
        "max_structured_output_retries", "max_concurrent_research_units",
        "max_researcher_iterations", "max_react_tool_calls",
        "erc_max_stagnation_iterations",
        "summarization_model_max_tokens", "max_content_length",
        "research_model_max_tokens", "compression_model_max_tokens",
        "final_report_model_max_tokens", mode="before"
    )
    @classmethod
    def strict_int_cast(cls, v):
        if v is None or v == "": return v
        if isinstance(v, str):
            try: return int(v)
            except ValueError: raise ValueError(f"Cannot cast '{v}' to int")
        return int(v)

    @field_validator("search_api", mode="before")
    @classmethod
    def strict_enum_cast(cls, v):
        if v is None or v == "": return SearchAPI.JINA
        if isinstance(v, str):
            try: return SearchAPI(v.lower())
            except ValueError: return SearchAPI.JINA
        return v

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        values = {}
        for field_name in cls.model_fields.keys():
            env_val = os.environ.get(field_name.upper())
            cfg_val = configurable.get(field_name)
            if cfg_val is not None:
                values[field_name] = cfg_val
            elif env_val is not None and env_val != "":
                values[field_name] = env_val
        return cls(**values)

    class Config:
        arbitrary_types_allowed = True

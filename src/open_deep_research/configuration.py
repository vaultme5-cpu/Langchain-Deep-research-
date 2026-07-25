"""Configuration management for the Open Deep Research system."""
import os
from enum import Enum
from typing import Any, List, Optional
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, field_validator

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
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=5)
    searxng_base_url: str = Field(default="http://localhost:8080")
    search_api: SearchAPI = Field(default=SearchAPI.SEARXNG)
    max_researcher_iterations: int = Field(default=6)
    max_react_tool_calls: int = Field(default=10)
    
    summarization_model: str = Field(default="groq:llama-3.1-8b-instant")
    summarization_model_max_tokens: int = Field(default=8192)
    max_content_length: int = Field(default=50000)
    
    research_model: str = Field(default="groq:llama-3.3-70b-versatile")
    research_model_max_tokens: int = Field(default=10000)
    
    compression_model: str = Field(default="groq:llama-3.1-8b-instant")
    compression_model_max_tokens: int = Field(default=8192)
    
    final_report_model: str = Field(default="groq:llama-3.3-70b-versatile")
    final_report_model_max_tokens: int = Field(default=10000)
    
    mcp_config: Optional[MCPConfig] = Field(default=None)
    mcp_prompt: Optional[str] = Field(default=None)

    @field_validator(
        'max_structured_output_retries', 'max_concurrent_research_units', 
        'max_researcher_iterations', 'max_react_tool_calls',
        'summarization_model_max_tokens', 'max_content_length',
        'research_model_max_tokens', 'compression_model_max_tokens',
        'final_report_model_max_tokens', mode='before'
    )
    @classmethod
    def strict_int_cast(cls, v):
        if isinstance(v, str):
            try: return int(v)
            except ValueError: raise ValueError(f"Cannot cast '{v}' to int")
        return int(v)

    @field_validator('search_api', mode='before')
    @classmethod
    def strict_enum_cast(cls, v):
        if isinstance(v, str):
            try:
                return SearchAPI(v.lower())
            except ValueError:
                raise ValueError(f"Invalid search_api: {v}")
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
            elif env_val is not None:
                values[field_name] = env_val
        return cls(**values)

    class Config:
        arbitrary_types_allowed = True

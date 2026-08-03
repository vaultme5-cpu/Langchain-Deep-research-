import os
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_core.runnables import RunnableConfig

class SearchAPI(Enum):
    JINA = "jina"
    SEARXNG = "searxng"
    NONE = "none"

class Configuration(BaseModel):
    max_structured_output_retries: int = Field(default=2)
    allow_clarification: bool = Field(default=True)
    max_concurrent_research_units: int = Field(default=2)
    searxng_base_url: str = Field(default="http://localhost:8080")
    search_api: SearchAPI = Field(default=SearchAPI.JINA)
    max_researcher_iterations: int = Field(default=4)
    max_react_tool_calls: int = Field(default=4)
    summarization_model: str = Field(default="groq:llama-3.1-8b-instant")
    summarization_model_max_tokens: int = Field(default=4096)
    max_content_length: int = Field(default=12000)
    research_model: str = Field(default="groq:llama-3.3-70b-versatile")
    research_model_max_tokens: int = Field(default=8192)
    compression_model: str = Field(default="groq:llama-3.1-8b-instant")
    compression_model_max_tokens: int = Field(default=4096)
    final_report_model: str = Field(default="groq:llama-3.3-70b-versatile")
    final_report_model_max_tokens: int = Field(default=8192)
    mcp_prompt: Optional[str] = Field(default=None)

    @field_validator("search_api", mode="before")
    @classmethod
    def strict_enum_cast(cls, v):
        if isinstance(v, str):
            try: return SearchAPI(v.lower())
            except ValueError: return SearchAPI.JINA
        return v

    @classmethod
    def from_runnable_config(cls, config=None):
        configurable = config.get("configurable", {}) if config else {}
        values = {}
        for field_name in cls.model_fields.keys():
            env_val = os.environ.get(field_name.upper())
            cfg_val = configurable.get(field_name)
            if cfg_val is not None: values[field_name] = cfg_val
            elif env_val is not None and env_val != "": values[field_name] = env_val
        return cls(**values)

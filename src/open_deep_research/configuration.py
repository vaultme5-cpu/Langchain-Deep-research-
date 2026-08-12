"""Configuration management for the Omega Supremacy Engine (Hybrid Multi-Brain)."""
import os
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator
from langchain_core.runnables import RunnableConfig


class SearchAPI(Enum):
    JINA = "jina"
    SEARXNG = "searxng"
    NONE = "none"


class MCPConfig(BaseModel):
    url: Optional[str] = None
    tools: Optional[List[str]] = None
    auth_required: bool = False


class Configuration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_structured_output_retries: int = 3
    allow_clarification: bool = True
    max_concurrent_research_units: int = 3
    max_researcher_iterations: int = 6
    max_react_tool_calls: int = 10
    erc_max_stagnation_iterations: int = 3

    search_api: SearchAPI = SearchAPI.JINA
    searxng_base_url: str = "http://localhost:8080"
    max_content_length: int = 12000

    summarization_model: str = "groq:llama-3.1-8b-instant"
    summarization_model_max_tokens: int = 2048

    research_model: str = "groq:llama-3.3-70b-versatile"
    research_model_max_tokens: int = 2048

    compression_model: str = "groq:llama-3.3-70b-versatile"
    compression_model_max_tokens: int = 3072

    final_report_model: str = "groq:llama-3.3-70b-versatile"
    final_report_model_max_tokens: int = 4096

    gemini_model: str = "google_genai:gemini-2.0-flash"
    gemini_model_max_tokens: int = 4096
    gemini_models: str = "google_genai:gemini-2.5-flash,google_genai:gemini-2.0-flash,google_genai:gemini-2.5-flash-lite"
    intake_model: str = "groq:llama-3.1-8b-instant"
    intake_model_max_tokens: int = 2048

    mcp_config: Optional[MCPConfig] = None
    mcp_prompt: Optional[str] = None

    groq_request_timeout: float = 60.0
    max_rate_limit_retries: int = 4
    max_tool_payload_chars: int = 5500
    max_compression_chunk_chars: int = 2800
    min_final_confidence: float = 0.65

    @field_validator(
        "max_structured_output_retries",
        "max_concurrent_research_units",
        "max_researcher_iterations",
        "max_react_tool_calls",
        "erc_max_stagnation_iterations",
        "summarization_model_max_tokens",
        "max_content_length",
        "research_model_max_tokens",
        "compression_model_max_tokens",
        "final_report_model_max_tokens",
        "gemini_model_max_tokens",
        "intake_model_max_tokens",
        "max_rate_limit_retries",
        "max_tool_payload_chars",
        "max_compression_chunk_chars",
        mode="before",
    )
    @classmethod
    def _cast_int(cls, v):
        if v is None or v == "":
            return v
        if isinstance(v, str):
            v = v.strip()
            if "." in v:
                return int(float(v))
            return int(v)
        return int(v)

    @field_validator("allow_clarification", mode="before")
    @classmethod
    def _cast_bool(cls, v):
        if v is None or v == "":
            return v
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return bool(v)

    @field_validator("search_api", mode="before")
    @classmethod
    def _cast_search_api(cls, v):
        if v is None or v == "":
            return SearchAPI.JINA
        if isinstance(v, SearchAPI):
            return v
        if isinstance(v, str):
            try:
                return SearchAPI(v.strip().lower())
            except ValueError:
                return SearchAPI.JINA
        return SearchAPI.JINA

    @field_validator(
        "searxng_base_url",
        "summarization_model",
        "research_model",
        "compression_model",
        "final_report_model",
        "gemini_model",
        "gemini_models",
        "intake_model",
        "mcp_prompt",
        mode="before",
    )
    @classmethod
    def _cast_str(cls, v):
        if v is None:
            return v
        return str(v).strip()

    @field_validator(
        "groq_request_timeout",
        "min_final_confidence",
        mode="before",
    )
    @classmethod
    def _cast_float(cls, v):
        if v is None or v == "":
            return v
        return float(v)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        values = {}
        for field_name in cls.model_fields.keys():
            cfg_val = configurable.get(field_name)
            env_val = os.environ.get(field_name.upper())
            if cfg_val is not None and str(cfg_val).strip() != "":
                values[field_name] = cfg_val
            elif env_val is not None and str(env_val).strip() != "":
                values[field_name] = env_val
        return cls(**values)

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
    max_concurrent_research_units: int = 2
    max_researcher_iterations: int = 4
    max_react_tool_calls: int = 6
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





    reasoning_model: str = "groq:llama-3.3-70b-versatile"
    reasoning_model_max_tokens: int = 3072

    # Conservative defaults for Groq free-tier operation.
    groq_concurrency: int = 1
    groq_tpm_soft_limit: int = 10000
    run_token_budget: int = 24000
    intake_model: str = "groq:llama-3.1-8b-instant"
    intake_model_max_tokens: int = 2048

    mcp_config: Optional[MCPConfig] = None
    mcp_prompt: Optional[str] = None

    groq_request_timeout: float = 60.0
    max_rate_limit_retries: int = 4
    max_tool_payload_chars: int = 5500
    max_compression_chunk_chars: int = 2800
    min_final_confidence: float = 0.65
    enable_python_repl: bool = True

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
        "reasoning_model_max_tokens",
        "groq_concurrency",
        "groq_tpm_soft_limit",
        "run_token_budget",

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


    @model_validator(mode="after")
    def _validate_ranges(self):
        if not 1 <= self.max_concurrent_research_units <= 3:
            raise ValueError(
                "max_concurrent_research_units must be 1..3"
            )

        if not 1 <= self.max_researcher_iterations <= 20:
            raise ValueError(
                "max_researcher_iterations must be 1..20"
            )

        if not 1 <= self.max_react_tool_calls <= 20:
            raise ValueError(
                "max_react_tool_calls must be 1..20"
            )

        if not 1 <= self.erc_max_stagnation_iterations <= 10:
            raise ValueError(
                "erc_max_stagnation_iterations must be 1..10"
            )

        if not 1 <= self.groq_concurrency <= 3:
            raise ValueError(
                "groq_concurrency must be 1..3"
            )

        if not 1000 <= self.groq_tpm_soft_limit <= 12000:
            raise ValueError(
                "groq_tpm_soft_limit must be 1000..12000"
            )

        if not 4000 <= self.run_token_budget <= 50000:
            raise ValueError(
                "run_token_budget must be 4000..50000"
            )

        if not 1000 <= self.max_content_length <= 50000:
            raise ValueError(
                "max_content_length must be 1000..50000"
            )

        if not 1000 <= self.max_tool_payload_chars <= 20000:
            raise ValueError(
                "max_tool_payload_chars must be 1000..20000"
            )

        if not 500 <= self.max_compression_chunk_chars <= 10000:
            raise ValueError(
                "max_compression_chunk_chars must be 500..10000"
            )

        if not 5 <= self.groq_request_timeout <= 180:
            raise ValueError(
                "groq_request_timeout must be 5..180 seconds"
            )

        if not 1 <= self.max_rate_limit_retries <= 8:
            raise ValueError(
                "max_rate_limit_retries must be 1..8"
            )

        if not 0.0 <= self.min_final_confidence <= 1.0:
            raise ValueError(
                "min_final_confidence must be 0..1"
            )

        return self

    @field_validator("allow_clarification", "enable_python_repl", mode="before")
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
            raw = v.strip().lower()

            if raw in {
                "none",
                "disabled",
                "off",
            }:
                return SearchAPI.NONE

            try:
                return SearchAPI(raw)
            except ValueError:
                return SearchAPI.JINA

        return SearchAPI.JINA

    @field_validator(
        "searxng_base_url",
        "summarization_model",
        "research_model",
        "compression_model",
        "final_report_model",
        "reasoning_model",


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

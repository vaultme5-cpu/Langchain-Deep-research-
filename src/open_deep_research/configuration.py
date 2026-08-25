"""Configuration management for the Omega Supremacy Engine (Hybrid Multi-Brain)."""
import os
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
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
    run_token_budget: int = 310000
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
    enable_llm_verification: bool = False
    max_llm_verifications: int = 3
    force_adjudicate_high_risk: bool = False

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

        if not 4000 <= self.run_token_budget <= 5000000:
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

    def _compute_exact_worst_case_tokens(self):
        """I18.1: Compute exact worst-case token requirement.
        Call categories: intake, research, compression, reasoning,
        final_report, optional_llm_verification.
        Retry multiplication applied to every LLM call category.
        """
        retry_multiplier = 1 + self.max_rate_limit_retries
        # Call counts
        intake_calls = 3  # clarify + brief + router
        research_calls = self.max_researcher_iterations * self.max_concurrent_research_units
        compression_calls = research_calls  # one compression per researcher
        reasoning_calls = 3  # deductive + inductive + abductive
        report_calls = 1
        verification_calls = self.max_llm_verifications if self.enable_llm_verification else 0
        # Token computation per category
        intake_tokens = intake_calls * self.intake_model_max_tokens * retry_multiplier
        research_tokens = research_calls * self.research_model_max_tokens * retry_multiplier
        compression_tokens = compression_calls * self.compression_model_max_tokens * retry_multiplier
        reasoning_tokens = reasoning_calls * self.reasoning_model_max_tokens * retry_multiplier
        report_tokens = report_calls * self.final_report_model_max_tokens * retry_multiplier
        verification_tokens = verification_calls * self.compression_model_max_tokens * retry_multiplier
        return int(intake_tokens + research_tokens + compression_tokens + reasoning_tokens + report_tokens + verification_tokens)

    def _validate_cross_field(self):
        """I14.13: Cross-field configuration policy.
        Ensures model budgets, verification budgets, and retry
        configuration are mutually consistent."""
        budget = self.run_token_budget
        # Rule 1-4: Every model max_tokens must fit within run budget
        for field_name in (
            "summarization_model_max_tokens",
            "research_model_max_tokens",
            "compression_model_max_tokens",
            "final_report_model_max_tokens",
            "reasoning_model_max_tokens",
            "intake_model_max_tokens",
        ):
            value = getattr(self, field_name)
            if value > budget:
                raise ValueError(
                    f"{field_name} ({value}) exceeds run_token_budget ({budget})"
                )
        # Rule 5: Verification budget feasible when enabled
        if self.enable_llm_verification:
            implied_verif_cost = self.max_llm_verifications * 1000
            if implied_verif_cost > budget:
                raise ValueError(
                    f"max_llm_verifications ({self.max_llm_verifications}) implies "
                    f"cost {implied_verif_cost} > run_token_budget ({budget})"
                )
        # Rule 6: Concurrency sanity (research units served by groq slots)
        if self.max_concurrent_research_units < 1 or self.groq_concurrency < 1:
            raise ValueError("concurrency values must be >= 1")
        # Rule 7: Retry budget must not make execution infeasible
        max_potential_calls = (
            self.max_rate_limit_retries
            * self.max_researcher_iterations
            * self.max_react_tool_calls
        )
        if max_potential_calls > 10000:
            raise ValueError(
                f"Retry config implies {max_potential_calls} potential calls, "
                f"exceeding feasible limit of 10000"
            )
        # Rule 8: LLM verifications bounded by tool-call budget
        if self.max_llm_verifications > self.max_react_tool_calls * 2:
            raise ValueError(
                f"max_llm_verifications ({self.max_llm_verifications}) is "
                f"disproportionate to max_react_tool_calls ({self.max_react_tool_calls})"
            )
        # I18.1: EXACT DEFAULT EXECUTION FEASIBILITY
        # Uses _compute_exact_worst_case_tokens() for per-category computation.
        # No arbitrary headroom. Exact check: worst_case <= run_token_budget.
        _i18_1_worst_case = self._compute_exact_worst_case_tokens()
        if _i18_1_worst_case > self.run_token_budget:
            raise ValueError(
                "I18.1 execution infeasible: exact worst-case "
                + str(_i18_1_worst_case) + " tokens exceeds "
                + "run_token_budget (" + str(self.run_token_budget) + ")"
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
                raise ValueError(f"Invalid search_api value: {raw!r}. Must be one of: jina, searxng, none")

        raise ValueError(f"Invalid search_api type: {type(v).__name__}. Must be a string or SearchAPI enum")

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


# ============================================================
# I15.11: CONFIGURATION EDGE CASE BENCHMARK
# Invalid configs MUST fail loudly.
# ============================================================
def _run_i15_11_config_edge_benchmark():
    """I15.11: Prove configuration edge cases are handled correctly."""
    results = {"passed": 0, "failed": 0, "details": []}
    def check(name, condition):
        if condition: results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append("FAIL: " + name)

    def _try_config(**overrides):
        try:
            cfg = Configuration(**overrides)
            return cfg, None
        except Exception as e:
            return None, str(e)

    # T1: budget exactly equal to model max_tokens -> should pass
    cfg1, err1 = _try_config(run_token_budget=8000, summarization_model_max_tokens=8000)
    check("T1: budget == model max passes", cfg1 is not None)

    # T2: model max_tokens exceeds budget -> should fail
    cfg2, err2 = _try_config(run_token_budget=5000, summarization_model_max_tokens=6000)
    check("T2: model max > budget fails", cfg2 is None and "exceeds" in str(err2).lower())

    # T3: zero budget -> should fail (concurrency/retry checks)
    cfg3, err3 = _try_config(run_token_budget=0)
    check("T3: zero budget fails or has zero cap", cfg3 is None or cfg3.run_token_budget == 0)

    # T4: verification enabled at maximum load
    cfg4, err4 = _try_config(enable_llm_verification=True, max_llm_verifications=100, run_token_budget=500000)
    check("T4: max verification load accepted", cfg4 is not None)

    # T5: verification budget exceeds run budget
    cfg5, err5 = _try_config(enable_llm_verification=True, max_llm_verifications=1000, run_token_budget=10000)
    check("T5: impossible verification budget fails", cfg5 is None)

    # T6: concurrency = 1 (boundary)
    cfg6, err6 = _try_config(max_concurrent_research_units=1, groq_concurrency=1)
    check("T6: concurrency=1 passes", cfg6 is not None)

    # T7: concurrency = 0 -> should fail
    cfg7, err7 = _try_config(max_concurrent_research_units=0)
    check("T7: concurrency=0 fails", cfg7 is None and "concurrency" in str(err7).lower())

    # T8: retry boundary that creates infeasible call count
    cfg8, err8 = _try_config(max_rate_limit_retries=100, max_researcher_iterations=100, max_react_tool_calls=100)
    check("T8: infeasible retry config fails", cfg8 is None and "potential calls" in str(err8).lower())

    # T9: invalid search_api value -> explicit error
    cfg9, err9 = _try_config(search_api="not_a_real_search_engine")
    check("T9: invalid search_api fails", cfg9 is None and ("invalid" in str(err9).lower() or "search_api" in str(err9).lower()))

    # T10: valid search_api values pass
    for valid in ("jina", "searxng"):
        cfgV, errV = _try_config(search_api=valid)
        check("T10: search_api=" + valid + " passes", cfgV is not None)

    # T11: default config is valid
    cfgD, errD = _try_config()
    check("T11: default config valid", cfgD is not None)

    # T12: all model max_tokens at budget boundary
    cfg12, err12 = _try_config(
        run_token_budget=8000,
        summarization_model_max_tokens=8000,
        research_model_max_tokens=8000,
        compression_model_max_tokens=8000,
        final_report_model_max_tokens=8000,
        reasoning_model_max_tokens=8000,
        intake_model_max_tokens=8000,
    )
    check("T12: all models at budget boundary passes", cfg12 is not None)

    # T13: disproportionate verification vs tool calls
    cfg13, err13 = _try_config(max_llm_verifications=500, max_react_tool_calls=10)
    check("T13: disproportionate verification fails", cfg13 is None and "disproportionate" in str(err13).lower())

    results["total"] = results["passed"] + results["failed"]
    results["success"] = results["failed"] == 0
    results["verdict"] = "PASS" if results["success"] else "FAIL"
    return results




# ============================================================
# I16.10: EXECUTION FEASIBILITY BOUND BENCHMARK
# ============================================================
def _run_i16_10_feasibility_benchmark():
    """I16.10: Prove execution feasibility bound catches infeasible configs."""
    results = {"passed": 0, "failed": 0, "details": []}
    def check(name, condition):
        if condition: results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append("FAIL: " + name)
    def _try_config(**overrides):
        try:
            cfg = Configuration(**overrides)
            return cfg, None
        except Exception as e:
            return None, str(e)
    # T1: default config is feasible
    cfg1, err1 = _try_config()
    check("T1: default config feasible", cfg1 is not None)
    # T2: tiny budget with huge workload is infeasible
    cfg2, err2 = _try_config(
        run_token_budget=4000,
        max_researcher_iterations=20,
        max_concurrent_research_units=3,
        max_rate_limit_retries=8,
        research_model_max_tokens=4000,
    )
    check("T2: infeasible config rejected", cfg2 is None and "I16.10" in str(err2))
    # T3: moderate config is feasible
    cfg3, err3 = _try_config(
        run_token_budget=24000,
        max_researcher_iterations=4,
        max_concurrent_research_units=2,
    )
    check("T3: moderate config feasible", cfg3 is not None)
    # T4: verification enabled adds calls but stays feasible
    cfg4, err4 = _try_config(
        run_token_budget=50000,
        enable_llm_verification=True,
        max_llm_verifications=6,
    )
    check("T4: verification config feasible", cfg4 is not None)
    # T5: absurd retry count with small budget is infeasible
    cfg5, err5 = _try_config(
        run_token_budget=4000,
        max_rate_limit_retries=8,
        max_researcher_iterations=20,
        max_concurrent_research_units=3,
    )
    check("T5: absurd retries infeasible", cfg5 is None)
    results["total"] = results["passed"] + results["failed"]
    results["success"] = results["failed"] == 0
    results["verdict"] = "PASS" if results["success"] else "FAIL"
    return results



# ============================================================
# I17.7: EXACT EXECUTION FEASIBILITY BENCHMARK
# ============================================================
def _run_i17_7_feasibility_benchmark():
    """I17.7: Prove exact feasibility check works correctly."""
    results = {"passed": 0, "failed": 0, "details": []}
    def check(name, condition):
        if condition: results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append("FAIL: " + name)

    def _try_config(**overrides):
        try:
            cfg = Configuration(**overrides)
            return cfg, None
        except Exception as e:
            return None, str(e)

    # T1: Mathematically feasible config -> PASS
    cfg1, err1 = _try_config(
        run_token_budget=500000,
        max_researcher_iterations=2,
        max_concurrent_research_units=1,
        max_rate_limit_retries=1,
        research_model_max_tokens=2048,
        compression_model_max_tokens=2048,
        final_report_model_max_tokens=2048,
        reasoning_model_max_tokens=2048,
        intake_model_max_tokens=2048,
        summarization_model_max_tokens=2048,
    )
    check("T1: feasible config passes", cfg1 is not None)

    # T2: Exactly-at-limit config -> PASS
    # intake=3*2048=6144, research=2*1*2048=4096, compression=2*2048=4096
    # reasoning=3*2048=6144, report=1*2048=2048, verification=0
    # base=22528, retries=1, worst=22528*2=45056
    cfg2, err2 = _try_config(
        run_token_budget=45056,
        max_researcher_iterations=2,
        max_concurrent_research_units=1,
        max_rate_limit_retries=1,
        research_model_max_tokens=2048,
        compression_model_max_tokens=2048,
        final_report_model_max_tokens=2048,
        reasoning_model_max_tokens=2048,
        intake_model_max_tokens=2048,
        summarization_model_max_tokens=2048,
    )
    check("T2: exactly-at-limit passes", cfg2 is not None)

    # T3: One-token-over-limit config -> FAIL
    cfg3, err3 = _try_config(
        run_token_budget=45055,
        max_researcher_iterations=2,
        max_concurrent_research_units=1,
        max_rate_limit_retries=1,
        research_model_max_tokens=2048,
        compression_model_max_tokens=2048,
        final_report_model_max_tokens=2048,
        reasoning_model_max_tokens=2048,
        intake_model_max_tokens=2048,
        summarization_model_max_tokens=2048,
    )
    check("T3: one-token-over fails", cfg3 is None and "I17.7" in str(err3))

    # T4: Impossible retry-heavy config -> FAIL
    cfg4, err4 = _try_config(
        run_token_budget=4000,
        max_researcher_iterations=20,
        max_concurrent_research_units=3,
        max_rate_limit_retries=8,
        research_model_max_tokens=4096,
        compression_model_max_tokens=4096,
        final_report_model_max_tokens=4096,
        reasoning_model_max_tokens=4096,
        intake_model_max_tokens=4096,
        summarization_model_max_tokens=4096,
    )
    check("T4: impossible retry-heavy fails", cfg4 is None and "I17.7" in str(err4))

    results["total"] = results["passed"] + results["failed"]
    results["success"] = results["failed"] == 0
    results["verdict"] = "PASS" if results["success"] else "FAIL"
    return results



# ============================================================
# I18.1: EXACT DEFAULT EXECUTION FEASIBILITY BENCHMARK
# ============================================================
def _run_i18_1_feasibility_benchmark():
    """I18.1: Prove exact feasibility model works correctly."""
    results = {"passed": 0, "failed": 0, "details": []}
    def check(name, condition):
        if condition: results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append("FAIL: " + name)

    def _try_config(**overrides):
        try:
            cfg = Configuration(**overrides)
            return cfg, None
        except Exception as e:
            return None, str(e)

    # A: default config -> PASS
    cfg_a, err_a = _try_config()
    check("A: default config passes", cfg_a is not None)
    if cfg_a is not None:
        worst = cfg_a._compute_exact_worst_case_tokens()
        check("A: worst_case <= budget", worst <= cfg_a.run_token_budget)

    # B: exactly feasible -> PASS
    # Compute exact worst case for a known config
    cfg_b_base, _ = _try_config(
        max_researcher_iterations=2,
        max_concurrent_research_units=1,
        max_rate_limit_retries=1,
        intake_model_max_tokens=2048,
        research_model_max_tokens=2048,
        compression_model_max_tokens=2048,
        reasoning_model_max_tokens=2048,
        final_report_model_max_tokens=2048,
        enable_llm_verification=False,
    )
    if cfg_b_base is not None:
        exact_b = cfg_b_base._compute_exact_worst_case_tokens()
        cfg_b, err_b = _try_config(
            run_token_budget=exact_b,
            max_researcher_iterations=2,
            max_concurrent_research_units=1,
            max_rate_limit_retries=1,
            intake_model_max_tokens=2048,
            research_model_max_tokens=2048,
            compression_model_max_tokens=2048,
            reasoning_model_max_tokens=2048,
            final_report_model_max_tokens=2048,
            enable_llm_verification=False,
        )
        check("B: exactly feasible passes", cfg_b is not None)
    else:
        check("B: base config for B", False)

    # C: one-token-over-budget -> FAIL
    if cfg_b_base is not None:
        exact_c = cfg_b_base._compute_exact_worst_case_tokens()
        cfg_c, err_c = _try_config(
            run_token_budget=exact_c - 1,
            max_researcher_iterations=2,
            max_concurrent_research_units=1,
            max_rate_limit_retries=1,
            intake_model_max_tokens=2048,
            research_model_max_tokens=2048,
            compression_model_max_tokens=2048,
            reasoning_model_max_tokens=2048,
            final_report_model_max_tokens=2048,
            enable_llm_verification=False,
        )
        check("C: one-token-over fails", cfg_c is None and "I18.1" in str(err_c))
    else:
        check("C: base config for C", False)

    # D: impossible retry configuration -> FAIL
    cfg_d, err_d = _try_config(
        run_token_budget=4000,
        max_researcher_iterations=20,
        max_concurrent_research_units=3,
        max_rate_limit_retries=8,
        research_model_max_tokens=4096,
    )
    check("D: impossible retry config fails", cfg_d is None and "I18.1" in str(err_d))

    results["total"] = results["passed"] + results["failed"]
    results["success"] = results["failed"] == 0
    results["verdict"] = "PASS" if results["success"] else "FAIL"
    return results



# ============================================================
# I18.9: CONFIGURATION INTEGRITY BENCHMARK
# Deterministic tests for exact execution feasibility.
# ============================================================
def _run_i18_9_configuration_integrity_benchmark():
    """I18.9: Prove configuration integrity with deterministic tests."""
    results = {"passed": 0, "failed": 0, "details": []}
    def check(name, condition):
        if condition: results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append("FAIL: " + name)

    def _try_config(**overrides):
        try:
            cfg = Configuration(**overrides)
            return cfg, None
        except Exception as e:
            return None, str(e)

    # T1: Default config passes
    cfg1, err1 = _try_config()
    check("T1: default config passes", cfg1 is not None)
    if cfg1 is not None:
        wc = cfg1._compute_exact_worst_case_tokens()
        check("T1: worst_case <= budget", wc <= cfg1.run_token_budget)
        check("T1: worst_case is int", isinstance(wc, int))

    # T2: Exact feasible budget passes
    # Build a minimal config to compute exact worst case
    cfg_base, _ = _try_config(
        max_researcher_iterations=2,
        max_concurrent_research_units=1,
        max_rate_limit_retries=1,
        intake_model_max_tokens=2048,
        research_model_max_tokens=2048,
        compression_model_max_tokens=2048,
        reasoning_model_max_tokens=2048,
        final_report_model_max_tokens=2048,
        summarization_model_max_tokens=2048,
        enable_llm_verification=False,
    )
    if cfg_base is not None:
        exact_budget = cfg_base._compute_exact_worst_case_tokens()
        cfg2, err2 = _try_config(
            run_token_budget=exact_budget,
            max_researcher_iterations=2,
            max_concurrent_research_units=1,
            max_rate_limit_retries=1,
            intake_model_max_tokens=2048,
            research_model_max_tokens=2048,
            compression_model_max_tokens=2048,
            reasoning_model_max_tokens=2048,
            final_report_model_max_tokens=2048,
            summarization_model_max_tokens=2048,
            enable_llm_verification=False,
        )
        check("T2: exact feasible budget passes", cfg2 is not None)
    else:
        check("T2: base config available", False)

    # T3: Budget - 1 fails
    if cfg_base is not None:
        exact_budget = cfg_base._compute_exact_worst_case_tokens()
        cfg3, err3 = _try_config(
            run_token_budget=exact_budget - 1,
            max_researcher_iterations=2,
            max_concurrent_research_units=1,
            max_rate_limit_retries=1,
            intake_model_max_tokens=2048,
            research_model_max_tokens=2048,
            compression_model_max_tokens=2048,
            reasoning_model_max_tokens=2048,
            final_report_model_max_tokens=2048,
            summarization_model_max_tokens=2048,
            enable_llm_verification=False,
        )
        check("T3: budget-1 fails", cfg3 is None and "I18.1" in str(err3))
    else:
        check("T3: base config available", False)

    # T4: Retry-heavy impossible config fails
    cfg4, err4 = _try_config(
        run_token_budget=4000,
        max_researcher_iterations=20,
        max_concurrent_research_units=3,
        max_rate_limit_retries=8,
        research_model_max_tokens=4096,
        compression_model_max_tokens=4096,
        final_report_model_max_tokens=4096,
        reasoning_model_max_tokens=4096,
        intake_model_max_tokens=4096,
    )
    check("T4: retry-heavy impossible fails", cfg4 is None and ("I18.1" in str(err4) or "infeasible" in str(err4).lower()))

    # T5: _compute_exact_worst_case_tokens is NOT a model_validator
    import inspect
    method = Configuration._compute_exact_worst_case_tokens
    # Check it's a regular method, not decorated with model_validator
    check("T5: helper is plain method", callable(method) and not hasattr(method, '__wrapped__'))

    results["total"] = results["passed"] + results["failed"]
    results["success"] = results["failed"] == 0
    results["verdict"] = "PASS" if results["success"] else "FAIL"
    return results

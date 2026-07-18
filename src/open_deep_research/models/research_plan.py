from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from .common import OutputType, ROSModel, StrategyType, make_id, utc_now
from .research_task import ResearchTask


class BudgetEstimate(ROSModel):
    max_searches: int = 0
    max_crawls: int = 0
    max_llm_calls: int = 0
    max_tokens: int = 0
    max_minutes: int = 0


class ResearchPlan(ROSModel):
    plan_id: str = Field(default_factory=lambda: make_id("plan"))
    query_id: str
    strategy_type: StrategyType
    objective: str
    research_questions: list[str] = Field(default_factory=list)
    task_graph: list[ResearchTask] = Field(default_factory=list)
    expected_output_type: OutputType = OutputType.report
    success_criteria: list[str] = Field(default_factory=list)
    stop_conditions: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    estimated_budget: BudgetEstimate = Field(default_factory=BudgetEstimate)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        object.__setattr__(self, "updated_at", utc_now())
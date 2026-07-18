from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from .common import ExecutionStatus, PhaseType, ROSModel, make_id, utc_now


class BudgetRemaining(ROSModel):
    searches: int = 0
    crawls: int = 0
    llm_calls: int = 0
    tokens: int = 0
    minutes: int = 0


class ExecutionState(ROSModel):
    execution_id: str = Field(default_factory=lambda: make_id("exec"))
    query_id: str
    plan_id: str
    status: ExecutionStatus = ExecutionStatus.initialized
    active_tasks: list[str] = Field(default_factory=list)
    completed_tasks: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)
    cached_tasks: list[str] = Field(default_factory=list)
    checkpoint_id: Optional[str] = None
    current_phase: PhaseType = PhaseType.intake
    progress_percent: float = 0.0
    last_error: Optional[str] = None
    retry_budget_remaining: int = 0
    budget_remaining: BudgetRemaining = Field(default_factory=BudgetRemaining)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        object.__setattr__(self, "updated_at", utc_now())
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from .common import ROSModel, make_id, utc_now


class SoftStopThresholds(ROSModel):
    low_novelty_limit: int = 3
    duplicate_rate_limit: float = 0.5
    confidence_floor: float = 0.55


class Budget(ROSModel):
    budget_id: str = Field(default_factory=lambda: make_id("budget"))
    execution_id: str
    max_searches: int = 0
    max_crawls: int = 0
    max_llm_calls: int = 0
    max_tokens: int = 0
    max_minutes: int = 0
    used_searches: int = 0
    used_crawls: int = 0
    used_llm_calls: int = 0
    used_tokens: int = 0
    elapsed_minutes: float = 0.0
    soft_stop_thresholds: SoftStopThresholds = Field(default_factory=SoftStopThresholds)
    hard_stop_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        object.__setattr__(self, "updated_at", utc_now())
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from .common import Priority, ROSModel, TaskStatus, TaskType, make_id, utc_now


class TaskInputs(ROSModel):
    query_terms: list[str] = Field(default_factory=list)
    source_constraints: list[str] = Field(default_factory=list)
    time_constraints: Optional[str] = None
    extra_context: list[str] = Field(default_factory=list)


class TaskOutputs(ROSModel):
    search_request_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ResearchTask(ROSModel):
    task_id: str = Field(default_factory=lambda: make_id("task"))
    plan_id: str
    task_type: TaskType
    question: str
    inputs: TaskInputs = Field(default_factory=TaskInputs)
    dependencies: list[str] = Field(default_factory=list)
    priority: int = 0
    status: TaskStatus = TaskStatus.pending
    assigned_worker: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    outputs: TaskOutputs = Field(default_factory=TaskOutputs)
    created_at: datetime = Field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    @classmethod
    def from_question(
        cls,
        plan_id: str,
        question: str,
        task_type: TaskType,
        *,
        priority: Priority = Priority.balance,
    ) -> "ResearchTask":
        priority_value = {Priority.speed: 2, Priority.balance: 1, Priority.depth: 0}[priority]
        return cls(
            plan_id=plan_id,
            task_type=task_type,
            question=question,
            priority=priority_value,
        )
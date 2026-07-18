from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from .common import AmbiguityLevel, Priority, QueryIntent, ROSModel, make_id, utc_now


class QueryConstraints(ROSModel):
    time_range: Optional[str] = None
    geography: Optional[str] = None
    audience: Optional[str] = None
    format: Optional[str] = None
    depth: Optional[str] = None
    language: Optional[str] = None
    must_include: list[str] = Field(default_factory=list)
    must_exclude: list[str] = Field(default_factory=list)


class Query(ROSModel):
    query_id: str = Field(default_factory=lambda: make_id("query"))
    raw_text: str
    intent: QueryIntent
    topic: str
    subtopics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    constraints: QueryConstraints = Field(default_factory=QueryConstraints)
    ambiguity_level: AmbiguityLevel = AmbiguityLevel.low
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    priority: Priority = Priority.balance
    created_at: datetime = Field(default_factory=utc_now)
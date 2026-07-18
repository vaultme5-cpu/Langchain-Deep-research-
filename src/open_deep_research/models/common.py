from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class ROSModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        use_enum_values=True,
        str_strip_whitespace=True,
    )


class QueryIntent(str, Enum):
    compare = "compare"
    timeline = "timeline"
    ranking = "ranking"
    investigation = "investigation"
    forecast = "forecast"
    explanation = "explanation"
    tutorial = "tutorial"
    mixed = "mixed"


class StrategyType(str, Enum):
    comparison = "comparison"
    timeline = "timeline"
    forensic = "forensic"
    ranking = "ranking"
    exploratory = "exploratory"
    teaching = "teaching"
    evidence_first = "evidence_first"
    mixed = "mixed"


class TaskType(str, Enum):
    search = "search"
    crawl = "crawl"
    extract = "extract"
    rank = "rank"
    verify = "verify"
    compare = "compare"
    synthesize = "synthesize"
    critique = "critique"


class Priority(str, Enum):
    speed = "speed"
    balance = "balance"
    depth = "depth"


class AmbiguityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ExecutionStatus(str, Enum):
    initialized = "initialized"
    planning = "planning"
    searching = "searching"
    verifying = "verifying"
    synthesizing = "synthesizing"
    completed = "completed"
    failed = "failed"
    paused = "paused"


class PhaseType(str, Enum):
    intake = "intake"
    strategy = "strategy"
    planning = "planning"
    retrieval = "retrieval"
    evidence = "evidence"
    verification = "verification"
    reasoning = "reasoning"
    memory = "memory"
    evaluation = "evaluation"


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    skipped = "skipped"
    cached = "cached"


class SearchStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cached = "cached"
    skipped = "skipped"


class SearchProvider(str, Enum):
    searxng = "searxng"
    tavily = "tavily"
    google = "google"
    brave = "brave"
    exa = "exa"
    bing = "bing"
    other = "other"


class SourceType(str, Enum):
    web = "web"
    news = "news"
    academic = "academic"
    code = "code"
    forum = "forum"
    docs = "docs"


class OutputType(str, Enum):
    report = "report"
    table = "table"
    brief = "brief"
    analysis = "analysis"
    mixed = "mixed"
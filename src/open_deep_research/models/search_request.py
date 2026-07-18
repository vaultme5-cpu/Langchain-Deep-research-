from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from .common import ROSModel, SearchProvider, SearchStatus, SourceType, make_id, utc_now


class SearchFilters(ROSModel):
    time_range: Optional[str] = None
    domain_allowlist: list[str] = Field(default_factory=list)
    domain_blocklist: list[str] = Field(default_factory=list)
    language: Optional[str] = None


class SearchRequest(ROSModel):
    search_id: str = Field(default_factory=lambda: make_id("search"))
    task_id: str
    normalized_query: str
    original_query: str
    provider: SearchProvider
    source_type: SourceType = SourceType.web
    filters: SearchFilters = Field(default_factory=SearchFilters)
    dedupe_key: str
    status: SearchStatus = SearchStatus.pending
    result_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
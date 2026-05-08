"""
Session memory — tracks state within a single ReAct loop run.
Not persisted between sessions (use SQLite for that).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceSummary:
    url: str
    title: str
    domain: str
    summary: str
    token_count: int


@dataclass
class SessionMemory:
    goal: str
    session_id: str
    seen_urls: set[str] = field(default_factory=set)
    summaries: list[SourceSummary] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    iteration: int = 0

    def add_url(self, url: str) -> None:
        self.seen_urls.add(url)

    def is_seen(self, url: str) -> bool:
        return url in self.seen_urls

    def add_summary(self, s: SourceSummary) -> None:
        self.summaries.append(s)

    def add_query(self, q: str) -> None:
        if q not in self.search_queries:
            self.search_queries.append(q)

    def set_gaps(self, gaps: list[str]) -> None:
        self.gaps = [g for g in gaps if g]

    def has_enough_sources(self, min_sources: int = 2) -> bool:
        return len(self.summaries) >= min_sources

    def combined_summary_text(self) -> str:
        parts = []
        for i, s in enumerate(self.summaries, 1):
            parts.append(f"[{i}] Source: {s.url}\nTitle: {s.title}\n\n{s.summary}")
        return "\n\n---\n\n".join(parts)

    def citation_list(self) -> list[dict]:
        return [
            {"index": i + 1, "url": s.url, "title": s.title, "domain": s.domain}
            for i, s in enumerate(self.summaries)
        ]

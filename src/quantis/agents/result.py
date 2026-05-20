"""Analysis result dataclasses and extraction helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SingleResult:
    code: str = ""
    name: str = ""
    trend: str = ""
    rating: Optional[int] = None
    action_hint: str = ""
    full_text: str = ""
    snapshot: Optional[dict] = None
    prompt_tokens: int = 0
    cache_hit_tokens: int = 0


@dataclass
class BatchResult:
    results: list = field(default_factory=list)
    prompt: str = ""
    period: str = ""

    @property
    def successful(self) -> list[SingleResult]:
        return [r for r in self.results if isinstance(r, SingleResult)]

    @property
    def errors(self) -> list:
        return [r for r in self.results if isinstance(r, Exception)]

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.successful)

    @property
    def total_cache_hit_tokens(self) -> int:
        return sum(r.cache_hit_tokens for r in self.successful)


def extract_result(text: str) -> dict:
    """Extract trend, rating, and action hint from LLM response."""
    trend = ""
    for line in text.split("\n"):
        line = line.strip()
        if line and not trend:
            trend = line

    m = re.search(r"买卖评级[：:\s]*([-+]?\d+)", text)
    rating = int(m.group(1)) if m else None

    hint = ""
    hint_match = re.search(r"操作提示[：:]\s*(.+)", text)
    if hint_match:
        hint = hint_match.group(1).strip()

    return {"trend": trend, "rating": rating, "action_hint": hint}

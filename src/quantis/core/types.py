from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Union
import pandas as pd


class OutputMode(str, Enum):
    """Controls what an analyzer returns.
    - SERIES : pd.DataFrame aligned to input index (per-bar).
    - LAST   : dict of scalar values for the most recent bar.
    """
    SERIES = "series"
    LAST = "last"


@dataclass(frozen=True)
class AnalysisSpec:
    """Declarative request for a single analyzer.  kind = "indicator"|"pattern"."""
    name: str
    kind: str = "indicator"
    params: Mapping[str, Any] = field(default_factory=dict)
    alias: Optional[str] = None

    @property
    def key(self) -> str:
        return self.alias or self.name


@dataclass
class PhaseResult:
    """Aggregated phase state of a pattern (typically for the last bar)."""
    active: bool
    pattern: str = ""
    start_position: int = -1
    start_index: Any = None
    current_position: int = -1
    current_index: Any = None
    bar_count: int = 0
    confidence: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": self.active,
            "pattern": self.pattern,
            "start_position": self.start_position,
            "start_index": self.start_index,
            "current_position": self.current_position,
            "current_index": self.current_index,
            "bar_count": self.bar_count,
            "confidence": self.confidence,
            "extra": self.extra,
        }


AnalyzerOutput = Union[pd.DataFrame, Dict[str, Any]]


@dataclass
class AnalysisResult:
    """Result object returned by AnalysisEngine.run().
    items maps analyzer key -> DataFrame (SERIES) or dict (LAST).
    """
    mode: OutputMode
    items: Dict[str, AnalyzerOutput] = field(default_factory=dict)

    def __getitem__(self, key: str) -> AnalyzerOutput:
        return self.items[key]

    def __contains__(self, key: str) -> bool:
        return key in self.items

    def keys(self):
        return self.items.keys()

    def get(self, key: str, default=None):
        return self.items.get(key, default)

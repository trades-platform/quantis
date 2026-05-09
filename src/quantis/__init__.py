"""Quantis — OHLCV indicator & pattern analysis engine."""

from .core.types import (
    AnalysisResult,
    AnalysisSpec,
    OutputMode,
    PhaseResult,
)
from .core.base import BaseAnalyzer, BaseIndicator, BasePattern
from .core.registry import (
    INDICATORS,
    PATTERNS,
    get_indicator,
    get_pattern,
    list_indicators,
    list_patterns,
    register_indicator,
    register_pattern,
)
from .core.engine import AnalysisEngine, analyze
from .core.snapshot import snapshot

# Trigger registrations of built-in analyzers
from . import indicators, patterns  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "OutputMode",
    "AnalysisSpec",
    "AnalysisResult",
    "PhaseResult",
    "BaseAnalyzer",
    "BaseIndicator",
    "BasePattern",
    "INDICATORS",
    "PATTERNS",
    "get_indicator",
    "get_pattern",
    "list_indicators",
    "list_patterns",
    "register_indicator",
    "register_pattern",
    "AnalysisEngine",
    "analyze",
    "snapshot",
]

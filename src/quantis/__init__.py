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
from .prompts.registry import (
    PROMPTS,
    get_prompt,
    list_prompts,
    register_prompt,
)

# Trigger registrations of built-in analyzers and prompts
from . import indicators, patterns, prompts  # noqa: F401

# Lazy import — only needed when fetching live market data
def __getattr__(name):  # type: ignore[no-redef]
    if name in ("fetch_klines", "fetch_klines_multi"):
        from .data import fetch_klines, fetch_klines_multi
        globals()["fetch_klines"] = fetch_klines
        globals()["fetch_klines_multi"] = fetch_klines_multi
        return globals()[name]
    _AGENT_EXPORTS = {
        "AnalysisAgent": ".agents",
        "BatchResult": ".agents",
        "SingleResult": ".agents",
        "LLMProvider": ".agents",
        "get_provider": ".agents",
    }
    if name in _AGENT_EXPORTS:
        import importlib
        mod = importlib.import_module(_AGENT_EXPORTS[name], __name__)
        obj = getattr(mod, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "PROMPTS",
    "get_prompt",
    "list_prompts",
    "register_prompt",
    "fetch_klines",
    "fetch_klines_multi",
    "AnalysisAgent",
    "BatchResult",
    "LLMProvider",
    "get_provider",
]

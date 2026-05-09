from __future__ import annotations
from typing import Iterable, Mapping, Sequence, Union
import pandas as pd
from ..utils.validation import ensure_ohlcv
from .registry import INDICATORS, PATTERNS
from .types import AnalysisResult, AnalysisSpec, OutputMode

SpecLike = Union[AnalysisSpec, str, Mapping, tuple]


def _coerce_spec(spec: SpecLike) -> AnalysisSpec:
    if isinstance(spec, AnalysisSpec):
        return spec
    if isinstance(spec, str):
        kind, _, name = spec.partition(":")
        if name:
            return AnalysisSpec(name=name, kind=kind)
        if spec in INDICATORS:
            return AnalysisSpec(name=spec, kind="indicator")
        if spec in PATTERNS:
            return AnalysisSpec(name=spec, kind="pattern")
        raise ValueError(f"Unknown analyzer '{spec}'. Use 'indicator:NAME' or 'pattern:NAME'.")
    if isinstance(spec, Mapping):
        return AnalysisSpec(
            name=spec["name"],
            kind=spec.get("kind", "indicator"),
            params=spec.get("params") or {},
            alias=spec.get("alias"),
        )
    if isinstance(spec, tuple):
        if len(spec) == 2:
            name, params = spec
            if name in PATTERNS:
                kind = "pattern"
            elif name in INDICATORS:
                kind = "indicator"
            else:
                raise ValueError(
                    f"Unknown analyzer '{name}'. Use a 3-tuple (kind, name, params) "
                    f"to disambiguate or register the analyzer first."
                )
            return AnalysisSpec(name=name, kind=kind, params=params or {})
        if len(spec) == 3:
            kind, name, params = spec
            return AnalysisSpec(name=name, kind=kind, params=params or {})
    raise TypeError(f"Cannot interpret spec: {spec!r}")


class AnalysisEngine:
    """Run a collection of analyzers against an OHLCV DataFrame."""

    def __init__(self, validate: bool = True):
        self.validate = validate

    def run(
        self,
        df: pd.DataFrame,
        specs: Iterable[SpecLike],
        mode: Union[OutputMode, str] = OutputMode.SERIES,
    ) -> AnalysisResult:
        if self.validate:
            df = ensure_ohlcv(df)
        if isinstance(mode, str):
            mode = OutputMode(mode.lower())
        result = AnalysisResult(mode=mode)
        for raw in specs:
            spec = _coerce_spec(raw)
            analyzer = (
                INDICATORS.get(spec.name) if spec.kind == "indicator"
                else PATTERNS.get(spec.name)
            )
            result.items[spec.key] = analyzer.run(df, spec.params, mode=mode)
        return result


def analyze(
    df: pd.DataFrame,
    specs: Sequence[SpecLike],
    mode: Union[OutputMode, str] = OutputMode.SERIES,
    *,
    validate: bool = True,
) -> AnalysisResult:
    """Convenience wrapper around AnalysisEngine."""
    return AnalysisEngine(validate=validate).run(df, specs, mode=mode)

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping
import pandas as pd
from .types import OutputMode, PhaseResult


class BaseAnalyzer(ABC):
    """Common interface shared by indicators and patterns.

    Subclasses implement compute() which always returns a per-bar DataFrame.
    run() dispatches to the requested OutputMode.
    """
    name: str = ""
    description: str = ""
    kind: str = "analyzer"
    default_params: Dict[str, Any] = {}

    def validate_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Merge caller params with defaults declared in default_params.

        Raises ValueError if ``params`` contains keys not declared in
        ``default_params`` (catches typos like ``peirod`` -> ``period``).
        """
        out: Dict[str, Any] = {}
        for key, spec in self.default_params.items():
            default = spec.get("default") if isinstance(spec, dict) else spec
            out[key] = params.get(key, default) if params else default
        if params:
            unknown = set(params) - set(self.default_params)
            if unknown:
                raise ValueError(
                    f"{self.name}: unknown parameter(s) {sorted(unknown)}. "
                    f"Allowed: {sorted(self.default_params)}"
                )
        return out

    @abstractmethod
    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Return a DataFrame aligned to df.index with computed columns."""

    def last(self, series_df: pd.DataFrame) -> Dict[str, Any]:
        """Reduce series DataFrame to the last bar."""
        if series_df.empty:
            return {}
        row = series_df.iloc[-1]
        return {col: _scalarize(row[col]) for col in series_df.columns}

    def run(
        self,
        df: pd.DataFrame,
        params: Mapping[str, Any] | None = None,
        mode: OutputMode = OutputMode.SERIES,
    ):
        validated = self.validate_params(params or {})
        series_df = self.compute(df, validated)
        if mode is OutputMode.SERIES:
            return series_df
        return self.last(series_df)


class BaseIndicator(BaseAnalyzer):
    """Adds numeric time-series columns to OHLCV (MA, MACD, BOLL, ...)."""
    kind = "indicator"


class BasePattern(BaseAnalyzer):
    """Marks per-bar form / market-state.

    Conventional compute() columns:
        active      bool   — whether this phase is in effect at bar i
        confidence  float  — [0, 1]
        bar_count   int    — length of the active phase ending at bar i
        + pattern-specific extra columns
    """
    kind = "pattern"
    confidence_desc: str = ""

    def last(self, series_df: pd.DataFrame) -> Dict[str, Any]:
        return self.summarize_phase(series_df).to_dict()

    def summarize_phase(self, series_df: pd.DataFrame) -> PhaseResult:
        if series_df.empty or "active" not in series_df.columns:
            return PhaseResult(active=False, pattern=self.name)
        last_pos = len(series_df) - 1
        if not bool(series_df["active"].iloc[-1]):
            return PhaseResult(active=False, pattern=self.name)
        bar_count = int(series_df["bar_count"].iloc[-1]) if "bar_count" in series_df.columns else 1
        start_pos = max(0, last_pos - bar_count + 1)
        confidence = float(series_df["confidence"].iloc[-1]) if "confidence" in series_df.columns else 1.0
        reserved = {"active", "confidence", "bar_count"}
        extra: Dict[str, Any] = {}
        row = series_df.iloc[-1]
        for col in series_df.columns:
            if col not in reserved:
                extra[col] = _scalarize(row[col])
        return PhaseResult(
            active=True,
            pattern=self.name,
            start_position=start_pos,
            start_index=series_df.index[start_pos],
            current_position=last_pos,
            current_index=series_df.index[last_pos],
            bar_count=bar_count,
            confidence=confidence,
            extra=extra,
        )


def _scalarize(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value

"""Produce an LLM-friendly snapshot from OHLCV analysis results."""
from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .engine import SpecLike, analyze
from .registry import INDICATORS, PATTERNS
from .types import OutputMode

_MAX_PHASES = 10


def snapshot(
    df: pd.DataFrame,
    specs: Sequence[SpecLike],
    *,
    recent_bars: int = 10,
) -> Dict[str, Any]:
    """Run full analysis and return an LLM-friendly summary dict.

    Internally calls ``analyze()`` only once in SERIES mode, then extracts
    both the per-bar trajectory and the last-bar snapshot from the result.

    Parameters
    ----------
    df : OHLCV DataFrame.
    specs : Analyzer specs (same format as ``analyze()``).
    recent_bars : Number of recent bars to include in the trajectory table.
    """
    series_result = analyze(df, specs, mode=OutputMode.SERIES)

    indicators, active_patterns, inactive_patterns = _classify_from_series(
        series_result.items,
    )

    result = {
        "last_bar": _build_last_bar(df, indicators),
        "recent_bars": _extract_recent_bars(df, series_result.items, recent_bars),
        "indicators": indicators,
        "active_patterns": active_patterns,
        "inactive_patterns": inactive_patterns,
        "pattern_phases": _extract_all_pattern_phases(series_result.items),
    }
    if "code" in df.attrs:
        result["symbol"] = df.attrs["code"]
    if "name" in df.attrs:
        result["name"] = df.attrs["name"]
    return result


# ── helpers ──────────────────────────────────────────────────────────


def _scalarize(v: Any) -> Any:
    if isinstance(v, (np.number, np.bool_)):
        v = v.item()
    if isinstance(v, float):
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, 4)
    return v


def _build_last_bar(df: pd.DataFrame, indicators: dict) -> dict:
    row = df.iloc[-1]
    close = float(row["close"])
    n = len(df)

    bar = {
        "date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
        "open": round(float(row["open"]), 4),
        "high": round(float(row["high"]), 4),
        "low": round(float(row["low"]), 4),
        "close": round(close, 4),
        "volume": int(row["volume"]),
    }

    # Multi-period change percentages (always expressed in calendar days)
    periods_days = [1, 3, 5, 7, 15, 30, 90, 180]
    changes: Dict[str, float] = {}
    last_ts = df.index[-1]
    for d in periods_days:
        target = last_ts - pd.Timedelta(days=d)
        loc = df.index.get_indexer([target], method="ffill")[0]
        if loc >= 0:
            prev_close = float(df["close"].iloc[loc])
            if prev_close > 0:
                changes[f"chgpct_{d}d"] = round((close - prev_close) / prev_close * 100, 2)
    if changes:
        bar["changes"] = changes

    if "ATR" in indicators:
        bar["atr"] = indicators["ATR"]
    return bar


def _classify_from_series(
    series_items: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, str]]]:
    """Extract indicators, active/inactive patterns from SERIES DataFrames.

    Reads the last row of each DataFrame to produce LAST-mode equivalents,
    avoiding a separate ``analyze()`` call.
    """
    indicators: Dict[str, Any] = {}
    active_patterns: List[Dict[str, Any]] = []
    inactive_patterns: List[Dict[str, str]] = []

    for key, sdf in series_items.items():
        if not isinstance(sdf, pd.DataFrame) or sdf.empty:
            continue

        is_pattern = "active" in sdf.columns
        last_row = sdf.iloc[-1]

        if is_pattern:
            pattern_name = key
            is_active = bool(last_row["active"])
            cls = PATTERNS._items.get(pattern_name)

            if is_active:
                entry: Dict[str, Any] = {
                    "pattern": pattern_name,
                    "bar_count": int(last_row["bar_count"]) if "bar_count" in sdf.columns else 1,
                    "confidence": float(last_row["confidence"]) if "confidence" in sdf.columns else 1.0,
                }
                if cls and getattr(cls, "confidence_desc", ""):
                    entry["confidence_desc"] = cls.confidence_desc
                reserved = {"active", "confidence", "bar_count"}
                for col in sdf.columns:
                    if col not in reserved:
                        v = _scalarize(last_row[col])
                        if v is not None:
                            entry[col] = v
                active_patterns.append(entry)
            else:
                ientry = {"pattern": pattern_name}
                if cls and getattr(cls, "description", ""):
                    ientry["description"] = cls.description
                inactive_patterns.append(ientry)
        else:
            # indicator DataFrame — take last row values
            for col in sdf.columns:
                v = _scalarize(last_row[col])
                if v is not None:
                    indicators[col] = v

    return indicators, active_patterns, inactive_patterns


def _extract_recent_bars(
    df: pd.DataFrame,
    series_items: Dict[str, Any],
    n: int,
) -> List[Dict[str, Any]]:
    tail = df.tail(n)
    indicator_dfs: Dict[str, pd.DataFrame] = {}
    for key, val in series_items.items():
        if isinstance(val, pd.DataFrame):
            cols = [c for c in val.columns if c not in ("active", "confidence", "bar_count")]
            if cols and "active" not in val.columns:
                indicator_dfs[key] = val[cols]

    result = []
    for i in range(len(tail)):
        idx = tail.index[i]
        row = tail.iloc[i]
        bar: Dict[str, Any] = {
            "date": str(idx.date()) if hasattr(idx, "date") else str(idx),
            "close": round(float(row["close"]), 4),
            "volume": int(row["volume"]),
        }
        for _key, ind_df in indicator_dfs.items():
            if idx in ind_df.index:
                for col in ind_df.columns:
                    v = _scalarize(ind_df.loc[idx, col])
                    if v is not None:
                        bar[col] = v
        result.append(bar)

    return result


def _extract_all_pattern_phases(
    series_items: Dict[str, Any],
) -> List[Dict[str, Any]]:
    phases: List[Dict[str, Any]] = []
    for key, val in series_items.items():
        if not isinstance(val, pd.DataFrame):
            continue
        if "active" not in val.columns:
            continue
        phases.extend(_extract_pattern_phases(key, val))
    # Keep only the most recent phases
    phases.sort(key=lambda p: p["end"], reverse=True)
    return phases[:_MAX_PHASES]


def _extract_pattern_phases(
    key: str,
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    active = df["active"].values
    confidence = df["confidence"].values if "confidence" in df.columns else None

    extra_cols = [c for c in df.columns if c not in ("active", "confidence", "bar_count")]

    phases = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < n and active[i]:
            i += 1
        end = i - 1

        phase: Dict[str, Any] = {
            "pattern": key,
            "start": _fmt_index(df.index[start]),
            "end": _fmt_index(df.index[end]),
            "length": end - start + 1,
        }
        if confidence is not None:
            phase["confidence_start"] = _scalarize(confidence[start])
            phase["confidence_end"] = _scalarize(confidence[end])
        last_row = df.iloc[end]
        for col in extra_cols:
            v = _scalarize(last_row[col])
            if v is not None:
                phase[col] = v
        phases.append(phase)

    return phases


def _fmt_index(idx) -> str:
    if hasattr(idx, "date"):
        return str(idx.date())
    return str(idx)

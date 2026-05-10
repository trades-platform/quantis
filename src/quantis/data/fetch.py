"""Fetch OHLCV klines via tickflow and normalise to quantis DataFrame format."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from tickflow import TickFlow

# ---------------------------------------------------------------------------
# Period mapping: quantis period  →  tickflow period
# 120min needs resample from 60min, so it maps to None as a sentinel.
# ---------------------------------------------------------------------------
_PERIOD_MAP: dict[Union[str, int], str] = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "60m",
    120: "_resample_60m",
    "daily": "1d",
    "weekly": "1w",
    "monthly": "1M",
}

_TICKFLOW_TO_QUANTIS_PERIOD: dict[str, Union[str, int]] = {
    v: k for k, v in _PERIOD_MAP.items() if not v.startswith("_")
}

# ---------------------------------------------------------------------------
# Symbol cache  (code without suffix  →  full tickflow symbol)
# ---------------------------------------------------------------------------
_CACHE_DIR = Path.home() / ".cache" / "quantis"
_CACHE_FILE = _CACHE_DIR / "symbol_map.json"
_SUFFIX_RE = re.compile(r"\.(SH|SZ|BJ|US|HK)$", re.IGNORECASE)


def _load_cache() -> dict[str, str]:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_cache(mapping: dict[str, str]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))


def _resolve_symbol(code: str, client) -> str:
    """Return a full tickflow symbol (e.g. ``'588000.SH'``).

    Accepts both ``'588000'`` and ``'588000.SH'``.  When the suffix is
    missing the mapping is looked up in a local JSON cache first; on cache
    miss the tickflow instruments API is queried and the result cached.
    """
    if _SUFFIX_RE.search(code):
        return code.upper()

    bare = code.upper()
    cache = _load_cache()
    if bare in cache:
        return cache[bare]

    for suffix in ("SH", "SZ", "BJ"):
        candidate = f"{bare}.{suffix}"
        try:
            info = client.instruments.get(candidate)
            if info and info.get("symbol"):
                full = info["symbol"]
                cache[bare] = full
                _save_cache(cache)
                return full
        except Exception:
            continue

    raise ValueError(f"Cannot resolve symbol for code '{code}' on SH/SZ/BJ")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_OHLCV_COLS = ("open", "high", "low", "close", "volume")


def _to_ms(t: Union[str, int, pd.Timestamp]) -> int:
    """Convert a time value to milliseconds timestamp for tickflow API."""
    if isinstance(t, int):
        return t
    ts = pd.Timestamp(t)
    return int(ts.timestamp() * 1000)


def _build_kwargs(
    tf_period: str,
    adjust: str,
    count: int,
    start_time,  # noqa: unused — kept for signature compat
    end_time,
) -> dict:
    # Always use count; tickflow start_time/end_time caps at 100 rows.
    # We slice locally in _trim instead.
    return dict(period=tf_period, adjust=adjust, as_dataframe=True, count=count)


def _fetch_raw(
    client,
    symbol: str,
    tf_period: str,
    adjust: str,
    count: int,
    start_time,
    end_time,
) -> pd.DataFrame:
    """Fetch raw klines from tickflow for a single period."""
    kwargs = _build_kwargs(tf_period, adjust, count, start_time, end_time)
    return client.klines.get(symbol, **kwargs)


def _extract_name(raw: pd.DataFrame) -> str:
    if isinstance(raw, pd.DataFrame) and not raw.empty and "name" in raw.columns:
        return str(raw["name"].iloc[-1])
    return ""


def _resample_120m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a 60min DataFrame to 120min (2-hour) bars."""
    result = df.resample("2h", closed="left", label="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    return result[result["volume"] > 0]


def _trim(
    df: pd.DataFrame,
    start_time: Optional[Union[str, int, pd.Timestamp]],
    end_time: Optional[Union[str, int, pd.Timestamp]],
) -> pd.DataFrame:
    """Slice normalised df by optional start/end time bounds."""
    if start_time is None and end_time is None:
        return df
    s = pd.Timestamp(start_time) if start_time is not None else None
    e = pd.Timestamp(end_time) if end_time is not None else None
    if s is not None:
        df = df[df.index >= s]
    if e is not None:
        df = df[df.index <= e]
    return df


def _normalise(
    raw: pd.DataFrame,
    symbol: str,
    name: str,
    period: Union[str, int],
) -> pd.DataFrame:
    """Convert a tickflow klines DataFrame to quantis format."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f"No kline data returned for {symbol}")

    if "timestamp" in raw.columns:
        idx = pd.to_datetime(raw["timestamp"], unit="ms")
    elif "trade_date" in raw.columns:
        idx = pd.to_datetime(raw["trade_date"])
    elif "trade_time" in raw.columns:
        idx = pd.to_datetime(raw["trade_time"])
    else:
        idx = raw.index

    df = pd.DataFrame(
        {
            "open": pd.to_numeric(raw["open"], errors="coerce").values,
            "high": pd.to_numeric(raw["high"], errors="coerce").values,
            "low": pd.to_numeric(raw["low"], errors="coerce").values,
            "close": pd.to_numeric(raw["close"], errors="coerce").values,
            "volume": pd.to_numeric(raw["volume"], errors="coerce").values,
        },
        index=idx,
    )
    df = df.dropna(subset=_OHLCV_COLS, how="all")
    df = df.sort_index()

    df.attrs["code"] = symbol
    if name:
        df.attrs["name"] = name
    df.attrs["period"] = period
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_klines(
    code: str,
    period: Union[str, int] = "daily",
    count: int = 800,
    *,
    start_time: Optional[Union[str, int, pd.Timestamp]] = None,
    end_time: Optional[Union[str, int, pd.Timestamp]] = None,
    adjust: str = "forward",
    api_key: Optional[str] = None,
    as_dict: bool = False,
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fetch klines for *code* and return a quantis-ready ``DataFrame``.

    Parameters
    ----------
    code : str
        Stock / ETF code, e.g. ``'588000'`` or ``'588000.SH'``.
    period : str | int
        K-line period.  Accepts quantis-style values (``'daily'``,
        ``'weekly'``, ``5``, ``15``, ``60``, ``120``, …) and
        tickflow-style values (``'1d'``, ``'5m'``, …).
        ``120`` fetches 60min bars and resamples to 120min.
    count : int
        Number of bars to fetch (max 10000).  Ignored when
        *start_time* / *end_time* are provided.
    start_time : str | int | pd.Timestamp, optional
        Start of the time range.
    end_time : str | int | pd.Timestamp, optional
        End of the time range (inclusive).
    adjust : str
        ``'forward'`` (default), ``'backward'``, or ``'none'``.
    api_key : str, optional
        TickFlow API key.  Falls back to free tier.
    as_dict : bool
        If *True*, return ``{"daily": df}`` keyed by period string.

    Returns
    -------
    pd.DataFrame  or  dict[str, pd.DataFrame]
    """
    tf_period = _PERIOD_MAP.get(period, str(period))
    needs_resample = tf_period == "_resample_60m"
    if needs_resample:
        tf_period = "60m"
        fetch_count = count * 2
    else:
        fetch_count = count

    ctx = TickFlow(api_key=api_key) if api_key else TickFlow.free()
    with ctx as client:
        full_symbol = _resolve_symbol(code, client)
        raw = _fetch_raw(client, full_symbol, tf_period, adjust, fetch_count, start_time, end_time)
        name = _extract_name(raw)

    df = _normalise(raw, full_symbol, name, period)
    if needs_resample:
        attrs = dict(df.attrs)
        df = _resample_120m(df)
        df.attrs.update(attrs)

    df = _trim(df, start_time, end_time)

    if as_dict:
        return {str(period): df}
    return df


def fetch_klines_multi(
    code: str,
    periods: Optional[list[Union[str, int]]] = None,
    count: int = 800,
    *,
    start_time: Optional[Union[str, int, pd.Timestamp]] = None,
    end_time: Optional[Union[str, int, pd.Timestamp]] = None,
    adjust: str = "forward",
    api_key: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Fetch klines for the same *code* across multiple periods.

    Parameters
    ----------
    code : str
        Stock / ETF code.
    periods : list, optional
        Defaults to ``["daily", "weekly", "monthly"]``.
    count : int
        Bars per period.  Ignored when *start_time* / *end_time* given.
    start_time, end_time : optional
        Time range filter.
    adjust : str
        ``'forward'``, ``'backward'``, or ``'none'``.
    api_key : str, optional
        TickFlow API key.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    if periods is None:
        periods = ["daily", "weekly", "monthly"]

    ctx = TickFlow(api_key=api_key) if api_key else TickFlow.free()
    results: dict[str, pd.DataFrame] = {}

    with ctx as client:
        full_symbol = _resolve_symbol(code, client)
        for p in periods:
            tf_period = _PERIOD_MAP.get(p, str(p))
            needs_resample = tf_period == "_resample_60m"
            if needs_resample:
                tf_period = "60m"
                fetch_count = count * 2
            else:
                fetch_count = count

            raw = _fetch_raw(client, full_symbol, tf_period, adjust, fetch_count, start_time, end_time)
            name = _extract_name(raw)
            df = _normalise(raw, full_symbol, name, p)
            if needs_resample:
                attrs = dict(df.attrs)
                df = _resample_120m(df)
                df.attrs.update(attrs)
            df = _trim(df, start_time, end_time)
            results[str(p)] = df

    return results

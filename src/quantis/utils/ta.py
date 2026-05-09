"""Pure pandas/numpy TA primitives — no external TA library required."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd


# ── moving averages ───────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ── oscillators ───────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0.0)
    loss  = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    # When avg_loss == 0, treat as "all gains" -> RSI = 100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0, 100.0)
    return out.where(avg_loss.notna() & avg_gain.notna())


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (DIFF, DEA, HIST)."""
    diff = ema(series, fast) - ema(series, slow)
    dea  = ema(diff, signal)
    return diff, dea, diff - dea


# ── bands / volatility ────────────────────────────────────────────────────────

def bbands(
    series: pd.Series,
    period: int = 20,
    stddev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (lower, middle, upper)."""
    middle = sma(series, period)
    std    = series.rolling(window=period, min_periods=period).std(ddof=0)
    return middle - stddev * std, middle, middle + stddev * std


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

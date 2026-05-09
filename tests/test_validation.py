"""Tests for OHLCV validation and TA primitive correctness."""
import numpy as np
import pandas as pd
import pytest
from quantis.utils.validation import ensure_ohlcv
from quantis.utils.ta import sma, ema, rsi, macd, bbands, atr


def _df(n=10):
    return pd.DataFrame({
        "open": np.arange(n, dtype=float),
        "high": np.arange(n, dtype=float) + 1,
        "low":  np.arange(n, dtype=float) - 1,
        "close": np.arange(n, dtype=float),
        "volume": np.full(n, 100.0),
    })


# ── ensure_ohlcv ────────────────────────────────────────────────────────────

def test_ensure_ohlcv_accepts_canonical():
    df = _df()
    out = ensure_ohlcv(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_ensure_ohlcv_normalises_case():
    df = _df().rename(columns=str.upper)
    out = ensure_ohlcv(df)
    assert "close" in out.columns and "CLOSE" not in out.columns


def test_ensure_ohlcv_missing_column():
    df = _df().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing"):
        ensure_ohlcv(df)


def test_ensure_ohlcv_non_numeric():
    df = _df()
    df["volume"] = df["volume"].astype(str)
    with pytest.raises(ValueError, match="numeric"):
        ensure_ohlcv(df)


def test_ensure_ohlcv_empty():
    with pytest.raises(ValueError, match="empty"):
        ensure_ohlcv(pd.DataFrame(columns=["open", "high", "low", "close", "volume"]))


def test_ensure_ohlcv_not_a_dataframe():
    with pytest.raises(TypeError):
        ensure_ohlcv([1, 2, 3])


# ── TA correctness ─────────────────────────────────────────────────────────

def test_sma_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = sma(s, 3)
    assert pd.isna(out.iloc[0]) and pd.isna(out.iloc[1])
    assert out.iloc[2] == 2.0  # (1+2+3)/3
    assert out.iloc[4] == 4.0  # (3+4+5)/3


def test_ema_known_values():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    # ema span=3 -> alpha=0.5; first val == 1.0
    out = ema(s, 3)
    assert out.iloc[0] == 1.0
    # subsequent: 0.5*x + 0.5*prev
    assert out.iloc[1] == 0.5 * 2 + 0.5 * 1.0


def test_rsi_all_gains_returns_100():
    """When every diff is positive, avg_loss==0 -> RSI must be 100, not NaN."""
    s = pd.Series(np.arange(1, 30, dtype=float))
    r = rsi(s, period=14)
    assert (r.dropna() == 100.0).all()


def test_rsi_all_losses_returns_0():
    s = pd.Series(np.arange(30, 1, -1, dtype=float))
    r = rsi(s, period=14)
    assert (r.dropna() == 0.0).all()


def test_macd_shapes():
    s = pd.Series(np.linspace(100, 200, 100))
    diff, dea, hist = macd(s, 12, 26, 9)
    assert len(diff) == len(dea) == len(hist) == len(s)
    assert np.allclose(hist.dropna(), (diff - dea).dropna())


def test_bbands_ordering():
    s = pd.Series(np.random.RandomState(0).normal(100, 5, 100))
    lower, middle, upper = bbands(s, 20, 2.0)
    valid = pd.concat([lower, middle, upper], axis=1).dropna()
    assert (valid.iloc[:, 2] >= valid.iloc[:, 1]).all()
    assert (valid.iloc[:, 1] >= valid.iloc[:, 0]).all()


def test_atr_positive():
    n = 50
    rng = np.random.RandomState(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + rng.uniform(0.1, 1, n)
    low  = close - rng.uniform(0.1, 1, n)
    a = atr(high, low, close, 14)
    assert a.dropna().gt(0).all()

import pytest
import pandas as pd
import numpy as np
import quantis
from quantis import OutputMode, analyze


def test_registered():
    names = quantis.list_patterns()
    assert "macd_dif_return_to_zero" in names
    assert "ma_fallback" in names
    assert "low_volume_pullback_ma" in names


def test_pattern_series_shape(sample_df):
    r = analyze(sample_df, ["pattern:macd_dif_return_to_zero"], mode=OutputMode.SERIES)
    df = r["macd_dif_return_to_zero"]
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_df)
    assert "active" in df.columns
    assert "confidence" in df.columns
    assert "bar_count" in df.columns


def test_pattern_last_returns_phase(sample_df):
    r = analyze(sample_df, ["pattern:macd_dif_return_to_zero"], mode=OutputMode.LAST)
    d = r["macd_dif_return_to_zero"]
    assert isinstance(d, dict)
    assert "active" in d
    assert "confidence" in d


def test_ma_fallback_series(sample_df):
    r = analyze(sample_df, ["pattern:ma_fallback"], mode=OutputMode.SERIES)
    df = r["ma_fallback"]
    assert "active" in df.columns
    # bar_count non-negative
    assert (df["bar_count"] >= 0).all()


def test_low_volume_pattern(sample_df):
    r = analyze(sample_df, ["pattern:low_volume_pullback_ma"], mode=OutputMode.SERIES)
    df = r["low_volume_pullback_ma"]
    assert "vol_ratio" in df.columns
    assert "dist_pct" in df.columns


def test_mix_indicator_and_pattern(sample_df):
    r = analyze(
        sample_df,
        ["indicator:ma", "pattern:ma_fallback"],
        mode=OutputMode.SERIES,
    )
    assert "ma" in r
    assert "ma_fallback" in r
    ma_df = r["ma"]
    pat_df = r["ma_fallback"]
    assert len(ma_df) == len(pat_df) == len(sample_df)


def test_alias(sample_df):
    r = analyze(
        sample_df,
        [{"name": "ma_fallback", "kind": "pattern", "alias": "pullback"}],
        mode=OutputMode.LAST,
    )
    assert "pullback" in r


def test_ma_fallback_triggers_on_pullback():
    """Construct a clear breakout-then-pullback price series and verify pattern fires."""
    import numpy as np
    import pandas as pd
    from quantis import analyze, OutputMode

    n = 120
    # Phase 1 (0-39): consolidation around 100
    p1 = np.full(40, 100.0) + np.random.RandomState(1).normal(0, 0.1, 40)
    # Phase 2 (40-89): strong breakout to 130
    p2 = np.linspace(100, 130, 50)
    # Phase 3 (90-119): shallow pullback to 126 (still well above MA60)
    p3 = np.linspace(130, 126, 30)
    close = np.concatenate([p1, p2, p3])
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(n, 1000.0),
    })

    r = analyze(df, ["pattern:ma_fallback"], mode=OutputMode.LAST)
    phase = r["ma_fallback"]
    assert phase["active"] is True
    assert phase["bar_count"] >= 1
    assert "peak_price" in phase["extra"]
    assert phase["extra"]["peak_price"] >= 129.0  # near 130


def test_macd_zero_golden_cross_triggers():
    """Verify pattern fires on a genuine zero-axis golden cross."""
    rs = np.random.RandomState(42)
    p1 = np.linspace(100, 96, 80)
    p2 = np.full(60, 96.0) + rs.normal(0, 0.05, 60)
    p3 = np.linspace(96, 100, 60)
    close = np.concatenate([p1, p2, p3])
    df = pd.DataFrame({
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.full(200, 1000.0),
    })
    r = analyze(df, ["pattern:macd_zero_golden_cross"], mode=OutputMode.SERIES)
    sdf = r["macd_zero_golden_cross"]
    assert sdf["active"].any(), "Should fire on zero-axis golden cross"
    assert "dif" in sdf.columns
    assert "dea" in sdf.columns
    assert "zero_bound" in sdf.columns


def test_macd_zero_golden_cross_rejects_far_from_zero():
    """Verify pattern does NOT fire when golden cross occurs far from zero axis."""
    rs = np.random.RandomState(42)
    close = np.linspace(100, 200, 200) + rs.normal(0, 0.2, 200)
    df = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(200, 1000.0),
    })
    r = analyze(df, ["pattern:macd_zero_golden_cross"], mode=OutputMode.LAST)
    assert r["macd_zero_golden_cross"]["active"] is False


def test_macd_zero_golden_cross_rejects_death_cross():
    """Verify pattern does NOT fire on a death cross (DIF down) near zero."""
    close = np.linspace(100, 95, 200)
    df = pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": np.full(200, 1000.0),
    })
    r = analyze(df, ["pattern:macd_zero_golden_cross"], mode=OutputMode.LAST)
    assert r["macd_zero_golden_cross"]["active"] is False

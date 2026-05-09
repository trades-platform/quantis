import pytest
import pandas as pd
import quantis
from quantis import OutputMode, analyze


def test_registered(sample_df):
    names = quantis.list_indicators()
    assert "ma" in names
    assert "macd" in names
    assert "boll" in names
    assert "rsi" in names
    assert "atr" in names
    assert "ema" in names
    assert "volume" in names


def test_ma_series(sample_df):
    r = analyze(sample_df, ["indicator:ma"], mode=OutputMode.SERIES)
    df = r["ma"]
    assert isinstance(df, pd.DataFrame)
    assert "MA20" in df.columns
    assert len(df) == len(sample_df)
    # first 19 rows should be NaN for MA20
    assert df["MA20"].iloc[:19].isna().all()
    assert not df["MA20"].iloc[19:].isna().any()


def test_ma_last(sample_df):
    r = analyze(sample_df, ["indicator:ma"], mode=OutputMode.LAST)
    d = r["ma"]
    assert isinstance(d, dict)
    assert "MA20" in d
    assert isinstance(d["MA20"], float)


def test_macd_series(sample_df):
    r = analyze(sample_df, ["indicator:macd"], mode=OutputMode.SERIES)
    df = r["macd"]
    assert {"DIFF", "DEA", "HIST"}.issubset(df.columns)


def test_macd_custom_params(sample_df):
    specs = [{"name": "macd", "kind": "indicator",
              "params": {"fast": 5, "slow": 10, "signal": 3}, "alias": "macd_fast"}]
    r = analyze(sample_df, specs, mode=OutputMode.LAST)
    assert "macd_fast" in r
    assert "DIFF" in r["macd_fast"]


def test_boll_series(sample_df):
    r = analyze(sample_df, ["indicator:boll"], mode=OutputMode.SERIES)
    df = r["boll"]
    assert {"BOLL_LOWER", "BOLL_MIDDLE", "BOLL_UPPER"}.issubset(df.columns)
    # upper > middle > lower for non-NaN rows
    valid = df.dropna()
    assert (valid["BOLL_UPPER"] > valid["BOLL_MIDDLE"]).all()
    assert (valid["BOLL_MIDDLE"] > valid["BOLL_LOWER"]).all()


def test_rsi_range(sample_df):
    r = analyze(sample_df, ["indicator:rsi"], mode=OutputMode.SERIES)
    vals = r["rsi"]["RSI14"].dropna()
    assert (vals >= 0).all() and (vals <= 100).all()


def test_atr_positive(sample_df):
    r = analyze(sample_df, ["indicator:atr"], mode=OutputMode.SERIES)
    assert r["atr"]["ATR14"].dropna().gt(0).all()


def test_volume_series(sample_df):
    r = analyze(sample_df, [{"name": "volume", "params": {"periods": [5]}}])
    assert "VOLUME" in r["volume"].columns
    assert "VOLUME_MA5" in r["volume"].columns


def test_multi_analyzer(sample_df):
    r = analyze(sample_df, ["indicator:ma", "indicator:rsi", "indicator:macd"])
    assert "ma" in r and "rsi" in r and "macd" in r


def test_invalid_macd_params(sample_df):
    with pytest.raises(ValueError):
        analyze(sample_df, [{"name": "macd", "params": {"fast": 26, "slow": 12}}])

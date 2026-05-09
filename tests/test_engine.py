import pytest
import pandas as pd
import numpy as np
from quantis import analyze, OutputMode, AnalysisEngine


def make_df(n=100):
    rng = np.random.default_rng(0)
    c = 50 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": c, "high": c + 0.5, "low": c - 0.5, "close": c,
        "volume": rng.integers(1000, 10000, n).astype(float),
    })


def test_output_mode_string():
    df = make_df()
    r = analyze(df, ["indicator:ma"], mode="last")
    assert r.mode == OutputMode.LAST


def test_empty_specs():
    df = make_df()
    r = analyze(df, [])
    assert list(r.keys()) == []


def test_invalid_df():
    with pytest.raises((TypeError, ValueError)):
        analyze(pd.DataFrame({"x": [1, 2]}), ["indicator:ma"])


def test_tuple_spec():
    df = make_df()
    r = analyze(df, [("ma", {"periods": [5]})], mode=OutputMode.LAST)
    assert "ma" in r
    assert "MA5" in r["ma"]


def test_engine_no_validate():
    df = pd.DataFrame({"open": [1], "high": [2], "low": [0], "close": [1], "volume": [100]})
    engine = AnalysisEngine(validate=False)
    r = engine.run(df, ["indicator:ma"], mode=OutputMode.LAST)
    assert "ma" in r


def test_indicator_last_no_magic_index_key():
    """LAST mode dict must not contain a magic '_index' key."""
    df = make_df()
    r = analyze(df, ["indicator:ma"], mode=OutputMode.LAST)
    assert "_index" not in r["ma"]


def test_unknown_param_raises():
    df = make_df()
    with pytest.raises(ValueError, match="unknown parameter"):
        analyze(df, [{"name": "ma", "params": {"peirod": 14}}])


def test_mode_string_case_insensitive():
    df = make_df()
    for s in ("LAST", "Last", "last", "SERIES", "Series"):
        r = analyze(df, ["indicator:ma"], mode=s)
        assert r.mode in (OutputMode.LAST, OutputMode.SERIES)


def test_tuple_spec_unknown_name_raises():
    df = make_df()
    with pytest.raises(ValueError, match="Unknown analyzer"):
        analyze(df, [("does_not_exist", {})])

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """200-bar synthetic OHLCV with a gentle uptrend."""
    rng = np.random.default_rng(42)
    n = 200
    close = 100.0 + np.cumsum(rng.normal(0.1, 1.0, n))
    high   = close + rng.uniform(0.2, 1.0, n)
    low    = close - rng.uniform(0.2, 1.0, n)
    open_  = close + rng.normal(0, 0.3, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

from __future__ import annotations
import pandas as pd

REQUIRED = ("open", "high", "low", "close", "volume")


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV columns; normalise to lowercase; return df."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if df.empty:
        raise ValueError("OHLCV DataFrame is empty")
    cols_lower = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED if c not in cols_lower]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    rename = {orig: canon for canon, orig in cols_lower.items()
               if canon != orig and canon in REQUIRED}
    if rename:
        df = df.rename(columns=rename)
    for col in REQUIRED:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")
    return df

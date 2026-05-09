from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import rsi as _rsi


@register_indicator
class RSI(BaseIndicator):
    name = "rsi"
    description = "Relative Strength Index (Wilder smoothing)."
    default_params = {"period": {"default": 14}}

    def validate_params(self, params):
        out = super().validate_params(params)
        out["period"] = int(out["period"])
        if out["period"] < 2:
            raise ValueError("rsi: period must be >= 2")
        return out

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            {f"RSI{params['period']}": _rsi(df["close"], params["period"])},
            index=df.index,
        )

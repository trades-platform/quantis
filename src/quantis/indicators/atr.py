from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import atr as _atr


@register_indicator
class ATR(BaseIndicator):
    name = "atr"
    description = "Average True Range."
    default_params = {"period": {"default": 14}}

    def validate_params(self, params):
        out = super().validate_params(params)
        out["period"] = int(out["period"])
        if out["period"] < 2:
            raise ValueError("atr: period must be >= 2")
        return out

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            {f"ATR{params['period']}": _atr(df["high"], df["low"], df["close"], params["period"])},
            index=df.index,
        )

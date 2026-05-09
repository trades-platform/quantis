from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import sma


@register_indicator
class MA(BaseIndicator):
    name = "ma"
    description = "Simple moving average (SMA) of close, one column per period."
    default_params = {"periods": {"default": [5, 10, 20, 60]}}

    def validate_params(self, params):
        out = super().validate_params(params)
        out["periods"] = [int(p) for p in (out.get("periods") or [])]
        if not out["periods"]:
            raise ValueError("ma: at least one period required")
        return out

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            {f"MA{p}": sma(df["close"], p) for p in params["periods"]},
            index=df.index,
        )

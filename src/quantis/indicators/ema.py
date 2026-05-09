from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import ema


@register_indicator
class EMA(BaseIndicator):
    name = "ema"
    description = "Exponential moving average of close, one column per period."
    default_params = {"periods": {"default": [12, 26]}}

    def validate_params(self, params):
        out = super().validate_params(params)
        out["periods"] = [int(p) for p in (out.get("periods") or [])]
        if not out["periods"]:
            raise ValueError("ema: at least one period required")
        return out

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            {f"EMA{p}": ema(df["close"], p) for p in params["periods"]},
            index=df.index,
        )

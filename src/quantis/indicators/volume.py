from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import sma


@register_indicator
class Volume(BaseIndicator):
    name = "volume"
    description = "Raw volume + volume moving averages."
    default_params = {"periods": {"default": [5, 10]}}

    def validate_params(self, params):
        out = super().validate_params(params)
        out["periods"] = [int(p) for p in (out.get("periods") or [])]
        return out

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = {"VOLUME": df["volume"]}
        for p in params["periods"]:
            data[f"VOLUME_MA{p}"] = sma(df["volume"], p)
        return pd.DataFrame(data, index=df.index)

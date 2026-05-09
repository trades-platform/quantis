from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import bbands


@register_indicator
class BOLL(BaseIndicator):
    name = "boll"
    description = "Bollinger Bands (BOLL_LOWER, BOLL_MIDDLE, BOLL_UPPER)."
    default_params = {"period": {"default": 20}, "stddev": {"default": 2.0}}

    def validate_params(self, params):
        out = super().validate_params(params)
        out["period"] = int(out["period"])
        out["stddev"] = float(out["stddev"])
        if out["period"] <= 1:
            raise ValueError("boll: period must be > 1")
        return out

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        lower, middle, upper = bbands(df["close"], params["period"], params["stddev"])
        return pd.DataFrame(
            {"BOLL_LOWER": lower, "BOLL_MIDDLE": middle, "BOLL_UPPER": upper},
            index=df.index,
        )

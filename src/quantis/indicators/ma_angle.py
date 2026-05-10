from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import sma


@register_indicator
class MaAngle(BaseIndicator):
    name = "ma_angle"
    description = "MA angle in degrees (arctan of change rate × 100), one column per period."
    default_params = {"periods": {"default": [20, 60]}}

    def validate_params(self, params):
        out = super().validate_params(params)
        out["periods"] = [int(p) for p in (out.get("periods") or [])]
        if not out["periods"]:
            raise ValueError("ma_angle: at least one period required")
        return out

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        result = {}
        for p in params["periods"]:
            ma = sma(df["close"], p).values
            pct = np.full_like(ma, np.nan)
            pct[1:] = (ma[1:] - ma[:-1]) / np.where(ma[:-1] != 0, ma[:-1], np.nan)
            angle = np.degrees(np.arctan(pct * 100))
            result[f"MA{p}_ANGLE"] = np.round(angle, 4)
        return pd.DataFrame(result, index=df.index)

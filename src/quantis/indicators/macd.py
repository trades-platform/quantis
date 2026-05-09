from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import macd as _macd


@register_indicator
class MACD(BaseIndicator):
    name = "macd"
    description = "MACD: DIFF (fast-slow EMA), DEA (signal), HIST (DIFF-DEA)."
    default_params = {
        "fast":   {"default": 12},
        "slow":   {"default": 26},
        "signal": {"default": 9},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        fast, slow, signal = int(out["fast"]), int(out["slow"]), int(out["signal"])
        if fast >= slow:
            raise ValueError("macd: fast must be < slow")
        return {"fast": fast, "slow": slow, "signal": signal}

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        diff, dea, hist = _macd(df["close"], params["fast"], params["slow"], params["signal"])
        return pd.DataFrame({"DIFF": diff, "DEA": dea, "HIST": hist}, index=df.index)

from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from ..core.base import BaseIndicator
from ..core.registry import register_indicator
from ..utils.ta import bias as _bias


@register_indicator
class Bias(BaseIndicator):
    name = "bias"
    description = "Bias (乖离率): (close - MA) / MA * 100, negative = oversold."
    default_params = {"period": {"default": 6}}

    def validate_params(self, params):
        out = super().validate_params(params)
        out["period"] = int(out["period"])
        if out["period"] < 2:
            raise ValueError("bias: period must be >= 2")
        return out

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            {f"BIAS{params['period']}": _bias(df["close"], params["period"])},
            index=df.index,
        )

"""连续阳线触及布林上轨：短线过热，预示回调风险。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import bbands


@register_pattern
class ConsecutiveBullBoll(BasePattern):
    name = "consecutive_bull_boll"
    description = "连续阳线触及布林上轨，短线过热信号"
    default_params = {
        "bb_period":      {"default": 20},
        "bb_stddev":      {"default": 2.0},
        "min_count":      {"default": 3},
        "touch_tolerance": {"default": 0.005},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        return {
            "bb_period":      int(out["bb_period"]),
            "bb_stddev":      float(out["bb_stddev"]),
            "min_count":      int(out["min_count"]),
            "touch_tolerance": float(out["touch_tolerance"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        _, _, upper = bbands(df["close"], params["bb_period"], params["bb_stddev"])
        u = upper.values
        close = df["close"].values
        opens = df["open"].values
        n = len(close)
        tol = params["touch_tolerance"]
        min_count = params["min_count"]

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)

        for i in range(n):
            if np.isnan(u[i]) or u[i] <= 0:
                continue
            # 从当前 bar 往前数连续阳线
            count = 0
            j = i
            while j >= 0:
                if np.isnan(close[j]) or np.isnan(opens[j]):
                    break
                if close[j] <= opens[j]:
                    break
                count += 1
                j -= 1
            if count < min_count:
                continue
            dist = abs(close[i] - u[i]) / u[i]
            if dist > tol:
                continue
            conf = round((1.0 - dist / tol) * 0.5 + min(count / 10.0, 1.0) * 0.5, 4)
            active_arr[i]     = True
            confidence_arr[i] = conf
            bar_count_arr[i]  = count

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
        }, index=df.index)

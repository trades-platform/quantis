"""MACD DIF 从零轴上方回落向零轴的持续过程。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import macd


@register_pattern
class MacdDifReturnToZero(BasePattern):
    name = "macd_dif_return_to_zero"
    description = "检测 DIF 从零轴上方逐步回落向零轴的持续过程"
    default_params = {
        "fast":            {"default": 12},
        "slow":            {"default": 26},
        "signal":          {"default": 9},
        "min_decline_pct": {"default": 0.1},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        fast, slow = int(out["fast"]), int(out["slow"])
        if fast >= slow:
            raise ValueError("fast must be < slow")
        return {
            "fast": fast, "slow": slow,
            "signal": int(out["signal"]),
            "min_decline_pct": float(out["min_decline_pct"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        diff, _, _ = macd(df["close"], params["fast"], params["slow"], params["signal"])
        d = diff.values
        n = len(d)
        min_decline_pct = params["min_decline_pct"]

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)
        peak_dif_arr   = np.full(n, np.nan)
        dif_arr        = d.copy()

        running_max = float("-inf")
        decline_start = -1

        for i in range(n):
            if np.isnan(d[i]) or d[i] <= 0:
                running_max = float("-inf")
                decline_start = -1
                continue
            if d[i] >= running_max:
                running_max = d[i]
                decline_start = -1
            elif decline_start == -1:
                decline_start = i

            if decline_start == -1 or running_max <= 0:
                continue

            decline_pct = 1.0 - d[i] / running_max
            if decline_pct < min_decline_pct:
                continue

            active_arr[i]     = True
            confidence_arr[i] = round(decline_pct, 4)
            bar_count_arr[i]  = i - decline_start + 1
            peak_dif_arr[i]   = round(running_max, 4)

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
            "dif":        np.round(dif_arr, 4),
            "peak_dif":   peak_dif_arr,
        }, index=df.index)

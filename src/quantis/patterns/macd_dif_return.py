"""MACD DIF 回零轴：检测 DIF 从零轴上方或下方回归零轴的持续过程。"""
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
    description = "DIF回归零轴：从零轴上方回归为上涨中继蓄势（偏多），从零轴下方回归为下跌中继（偏空）"
    confidence_desc = "回归幅度占极值比例，0.3=已回归30%；direction字段标识above（零轴上方）或below（零轴下方）"
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
        min_pct = params["min_decline_pct"]

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)
        peak_dif_arr   = np.full(n, np.nan)
        dif_arr        = d.copy()
        direction_arr  = np.empty(n, dtype=object)

        run_max = float("-inf")
        run_min = float("inf")
        above_start = -1
        below_start = -1

        for i in range(n):
            if np.isnan(d[i]):
                run_max = float("-inf")
                run_min = float("inf")
                above_start = -1
                below_start = -1
                continue

            if d[i] > 0:
                run_min = float("inf")
                below_start = -1

                if d[i] >= run_max:
                    run_max = d[i]
                    above_start = -1
                elif above_start == -1:
                    above_start = i

                if above_start != -1 and run_max > 0:
                    pct = 1.0 - d[i] / run_max
                    if pct >= min_pct:
                        active_arr[i]     = True
                        confidence_arr[i] = round(pct, 4)
                        bar_count_arr[i]  = i - above_start + 1
                        peak_dif_arr[i]   = round(run_max, 4)
                        direction_arr[i]  = "above"

            elif d[i] < 0:
                run_max = float("-inf")
                above_start = -1

                if d[i] <= run_min:
                    run_min = d[i]
                    below_start = -1
                elif below_start == -1:
                    below_start = i

                if below_start != -1 and run_min < 0:
                    pct = 1.0 - d[i] / run_min
                    if pct >= min_pct:
                        active_arr[i]     = True
                        confidence_arr[i] = round(pct, 4)
                        bar_count_arr[i]  = i - below_start + 1
                        peak_dif_arr[i]   = round(run_min, 4)
                        direction_arr[i]  = "below"

            else:
                run_max = float("-inf")
                run_min = float("inf")
                above_start = -1
                below_start = -1

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
            "dif":        np.round(dif_arr, 4),
            "peak_dif":   peak_dif_arr,
            "direction":  direction_arr,
        }, index=df.index)

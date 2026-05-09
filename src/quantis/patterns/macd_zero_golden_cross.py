"""MACD 零轴金叉：DIF 在零轴附近上穿 DEA，中线走强信号。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import macd


@register_pattern
class MacdZeroGoldenCross(BasePattern):
    name = "macd_zero_golden_cross"
    description = "DIF在零轴附近上穿DEA，比普通金叉更可靠的趋势确认"
    confidence_desc = "金叉时DIF/DEA距零轴的接近程度，越接近零轴信号越强"
    default_params = {
        "fast":           {"default": 12},
        "slow":           {"default": 26},
        "signal":         {"default": 9},
        "zero_threshold": {"default": 0.01},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        fast, slow = int(out["fast"]), int(out["slow"])
        if fast >= slow:
            raise ValueError("fast must be < slow")
        return {
            "fast": fast, "slow": slow,
            "signal": int(out["signal"]),
            "zero_threshold": float(out["zero_threshold"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        diff, dea, _ = macd(df["close"], params["fast"], params["slow"], params["signal"])
        d = diff.values
        e = dea.values
        close = df["close"].values
        n = len(d)
        threshold = params["zero_threshold"]

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)

        for i in range(1, n):
            if np.isnan(d[i]) or np.isnan(e[i]) or np.isnan(d[i - 1]) or np.isnan(e[i - 1]):
                continue
            if close[i] <= 0:
                continue
            if not (d[i - 1] < e[i - 1] and d[i] > e[i]):
                continue
            zero_bound = threshold * close[i]
            if abs(d[i]) > zero_bound or abs(e[i]) > zero_bound:
                continue
            conf = round(1.0 - max(abs(d[i]), abs(e[i])) / zero_bound, 4)
            active_arr[i]     = True
            confidence_arr[i] = conf
            bar_count_arr[i]  = 1

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
        }, index=df.index)

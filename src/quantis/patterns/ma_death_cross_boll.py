"""MA20 死叉 MA60 且价格触及布林带上轨 — 高位反转风险信号。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import sma, bbands


@register_pattern
class MaDeathCrossBollUpper(BasePattern):
    name = "ma_death_cross_boll_upper"
    description = "检测MA20死叉MA60时价格触及布林带上轨"
    default_params = {
        "bb_period":      {"default": 20},
        "bb_stddev":      {"default": 2.0},
        "touch_tolerance": {"default": 0.005},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        return {
            "bb_period":      int(out["bb_period"]),
            "bb_stddev":      float(out["bb_stddev"]),
            "touch_tolerance": float(out["touch_tolerance"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        ma20 = sma(df["close"], 20).values
        ma60 = sma(df["close"], 60).values
        _, _, upper = bbands(df["close"], params["bb_period"], params["bb_stddev"])
        u = upper.values
        close = df["close"].values
        n = len(close)
        tol = params["touch_tolerance"]

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)

        for i in range(1, n):
            if np.isnan(ma20[i]) or np.isnan(ma60[i]) or np.isnan(u[i]):
                continue
            if np.isnan(ma20[i - 1]) or np.isnan(ma60[i - 1]):
                continue
            # 死叉: 前一根 MA20 >= MA60, 当前 MA20 < MA60
            if not (ma20[i - 1] >= ma60[i - 1] and ma20[i] < ma60[i]):
                continue
            # 触及布林上轨
            if u[i] <= 0:
                continue
            dist = abs(close[i] - u[i]) / u[i]
            if dist > tol:
                continue
            active_arr[i]     = True
            confidence_arr[i] = round(1.0 - dist / tol, 4)
            bar_count_arr[i]  = 1

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
        }, index=df.index)

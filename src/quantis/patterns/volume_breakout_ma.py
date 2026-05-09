"""放量突破 MA 长线：价格带量突破长期均线，趋势启动信号。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import sma


@register_pattern
class VolumeBreakoutMA(BasePattern):
    name = "volume_breakout_ma"
    description = "价格带量突破MA长线，量价齐升确认趋势启动"
    confidence_desc = "成交量超出阈值的比例，放量越明显突破越有效"
    default_params = {
        "ma_period":     {"default": 60},
        "vol_ma_period": {"default": 20},
        "vol_ratio":     {"default": 1.5},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        return {
            "ma_period":     int(out["ma_period"]),
            "vol_ma_period": int(out["vol_ma_period"]),
            "vol_ratio":     float(out["vol_ratio"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        ma = sma(df["close"], params["ma_period"]).values
        vol_ma = sma(df["volume"], params["vol_ma_period"]).values
        close = df["close"].values
        volume = df["volume"].values.astype(float)
        n = len(close)

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)
        vol_ratio_arr  = np.full(n, np.nan)

        for i in range(1, n):
            if np.isnan(ma[i]) or np.isnan(ma[i - 1]) or np.isnan(vol_ma[i]):
                continue
            if vol_ma[i] <= 0:
                continue
            if not (close[i - 1] < ma[i - 1] and close[i] > ma[i]):
                continue
            vr = volume[i] / vol_ma[i]
            if vr < params["vol_ratio"]:
                continue
            active_arr[i]     = True
            confidence_arr[i] = round(min(1.0, (vr - params["vol_ratio"]) / params["vol_ratio"]), 4)
            bar_count_arr[i]  = 1
            vol_ratio_arr[i]  = round(vr, 4)

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
            "vol_ratio":  vol_ratio_arr,
        }, index=df.index)

"""缩量回踩 MA：价格缩量回落到短期均线附近，健康回调买点。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import sma


@register_pattern
class LowVolumePullbackMA(BasePattern):
    name = "low_volume_pullback_ma"
    description = "价格缩量回踩MA附近，成交量低于均量，回调蓄势"
    default_params = {
        "ma_period":      {"default": 20},
        "vol_ma_period":  {"default": 20},
        "touch_tolerance": {"default": 0.005},
        "vol_threshold":  {"default": 0.7},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        return {
            "ma_period":      int(out["ma_period"]),
            "vol_ma_period":  int(out["vol_ma_period"]),
            "touch_tolerance": float(out["touch_tolerance"]),
            "vol_threshold":  float(out["vol_threshold"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        ma     = sma(df["close"],  params["ma_period"]).values
        vol_ma = sma(df["volume"], params["vol_ma_period"]).values
        close  = df["close"].values
        volume = df["volume"].values
        n = len(close)
        tol = params["touch_tolerance"]
        vth = params["vol_threshold"]

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)
        vol_ratio_arr  = np.full(n, np.nan)
        dist_arr       = np.full(n, np.nan)

        lookback = 10
        for i in range(n):
            if np.isnan(ma[i]) or np.isnan(vol_ma[i]) or ma[i] <= 0 or vol_ma[i] <= 0:
                continue
            dist = abs(close[i] - ma[i]) / ma[i]
            if dist > tol:
                continue
            vol_ratio = volume[i] / vol_ma[i]
            if vol_ratio > vth:
                continue
            # price must have been above MA before this bar
            lb = max(0, i - lookback)
            if not np.any(close[lb:i] > ma[lb:i]):
                continue
            conf = round((1.0 - vol_ratio / vth) * 0.5 + (1.0 - dist / tol) * 0.5, 4)
            active_arr[i]     = True
            confidence_arr[i] = conf
            bar_count_arr[i]  = 1
            vol_ratio_arr[i]  = round(vol_ratio, 4)
            dist_arr[i]       = round(dist, 4)

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
            "vol_ratio":  vol_ratio_arr,
            "dist_pct":   dist_arr,
        }, index=df.index)

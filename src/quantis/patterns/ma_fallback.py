"""价格突破 MA 长线后回落向短/长均线的持续过程。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import sma


@register_pattern
class MaFallback(BasePattern):
    name = "ma_fallback"
    description = "检测价格突破MA长线后逐步回落向MA短/长线的过程"
    default_params = {
        "ma_short":        {"default": 20},
        "ma_long":         {"default": 60},
        "min_breakout_pct": {"default": 0.01},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        ms, ml = int(out["ma_short"]), int(out["ma_long"])
        if ms >= ml:
            raise ValueError("ma_short must be < ma_long")
        return {"ma_short": ms, "ma_long": ml,
                "min_breakout_pct": float(out["min_breakout_pct"])}

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        ma_s = sma(df["close"], params["ma_short"]).values
        ma_l = sma(df["close"], params["ma_long"]).values
        close = df["close"].values
        n = len(close)
        min_bp = params["min_breakout_pct"]

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)
        target_arr     = np.empty(n, dtype=object)
        peak_arr       = np.full(n, np.nan)

        # ``breakout_high`` is the running high of close while it stays above
        # ma_long; reset only when price closes back below ma_long.
        # ``decline_start`` marks the first bar of the most recent pullback
        # (the bar after a fresh ``breakout_high`` was set).
        breakout_high = 0.0
        decline_start = -1

        for i in range(n):
            if np.isnan(ma_s[i]) or np.isnan(ma_l[i]):
                continue

            if close[i] <= ma_l[i]:
                # Lost the long MA — phase ends, reset everything.
                breakout_high = 0.0
                decline_start = -1
                continue

            if close[i] > breakout_high:
                # New high above ma_long — invalidates any in-progress pullback.
                breakout_high = close[i]
                decline_start = -1
                continue

            # close[i] above ma_long but below breakout_high -> pulling back.
            if decline_start == -1:
                if breakout_high <= ma_l[i] * (1 + min_bp):
                    # Breakout was too shallow to count.
                    continue
                decline_start = i

            cur_close, cms, cml = close[i], ma_s[i], ma_l[i]
            dist_s = (cur_close - cms) / cms
            dist_l = (cur_close - cml) / cml
            total  = (breakout_high - cml) / cml if cml > 0 else 0.0
            conf   = (
                round(max(0.0, min(1.0, 1.0 - min(dist_s, dist_l) / total)), 4)
                if total > 0 else 0.0
            )

            active_arr[i]     = True
            confidence_arr[i] = conf
            bar_count_arr[i]  = i - decline_start + 1
            target_arr[i]     = "ma_short" if dist_s < dist_l else "ma_long"
            peak_arr[i]       = round(breakout_high, 4)

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
            "target_ma":  target_arr,
            "peak_price": peak_arr,
        }, index=df.index)

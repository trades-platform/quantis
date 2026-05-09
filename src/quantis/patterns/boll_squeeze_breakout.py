"""布林带收窄后价格突破，预示大行情启动。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import bbands


@register_pattern
class BollSqueezeBreakout(BasePattern):
    name = "boll_squeeze_breakout"
    description = "布林带带宽收窄至极值后价格突破上轨或下轨"
    default_params = {
        "bb_period":      {"default": 20},
        "bb_stddev":      {"default": 2.0},
        "squeeze_window": {"default": 60},
        "squeeze_pct":    {"default": 0.2},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        return {
            "bb_period":      int(out["bb_period"]),
            "bb_stddev":      float(out["bb_stddev"]),
            "squeeze_window": int(out["squeeze_window"]),
            "squeeze_pct":    float(out["squeeze_pct"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        lower, mid, upper = bbands(df["close"], params["bb_period"], params["bb_stddev"])
        bw = ((upper - lower) / mid).values
        close = df["close"].values
        n = len(close)
        win = params["squeeze_window"]
        sq_pct = params["squeeze_pct"]

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)
        direction_arr  = np.empty(n, dtype=object)
        pct_rank_arr   = np.full(n, np.nan)

        for i in range(n):
            if np.isnan(bw[i]) or np.isnan(upper.iloc[i]) or np.isnan(lower.iloc[i]):
                continue
            w = min(win, i + 1)
            recent_bw = bw[i - w + 1:i + 1]
            valid_bw = recent_bw[~np.isnan(recent_bw)]
            if len(valid_bw) < 10:
                continue
            pct_rank = float(np.mean(valid_bw <= bw[i]))
            pct_rank_arr[i] = round(pct_rank, 4)
            if pct_rank > sq_pct:
                continue

            c = close[i]
            u = float(upper.iloc[i])
            l = float(lower.iloc[i])
            m = float(mid.iloc[i])

            if c >= u:
                direction_arr[i] = "up"
                confidence_arr[i] = round(min(1.0, (c - u) / (u - m + 1e-10)), 4)
            elif c <= l:
                direction_arr[i] = "down"
                confidence_arr[i] = round(min(1.0, (l - c) / (m - l + 1e-10)), 4)
            else:
                dist_upper = abs(c - u) / u if u > 0 else 999
                dist_lower = abs(c - l) / l if l > 0 else 999
                if dist_upper < 0.005:
                    direction_arr[i] = "up"
                    confidence_arr[i] = round(1.0 - dist_upper / 0.005, 4)
                elif dist_lower < 0.005:
                    direction_arr[i] = "down"
                    confidence_arr[i] = round(1.0 - dist_lower / 0.005, 4)
                else:
                    continue

            active_arr[i] = True
            bar_count_arr[i] = 1

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
            "direction":  direction_arr,
            "pct_rank":   pct_rank_arr,
        }, index=df.index)

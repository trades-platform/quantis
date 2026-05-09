"""MACD 顶背离：价格创新高但 DIF 未创新高，预示上涨动能衰竭。"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import macd
from .macd_bottom_divergence import _find_local_extrema


@register_pattern
class MacdTopDivergence(BasePattern):
    name = "macd_top_divergence"
    description = "价格创新高但DIF未创新高，预示上涨动能衰竭"
    default_params = {
        "fast":     {"default": 12},
        "slow":     {"default": 26},
        "signal":   {"default": 9},
        "order":    {"default": 5},
        "lookback": {"default": 60},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        fast, slow = int(out["fast"]), int(out["slow"])
        if fast >= slow:
            raise ValueError("fast must be < slow")
        return {
            "fast": fast, "slow": slow,
            "signal": int(out["signal"]),
            "order": int(out["order"]),
            "lookback": int(out["lookback"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        diff, _, _ = macd(df["close"], params["fast"], params["slow"], params["signal"])
        d = diff.values
        close = df["close"].values
        n = len(close)

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)

        order = params["order"]
        lookback = params["lookback"]

        for i in range(n):
            lb = max(0, i - lookback)
            sub_close = close[lb:i + 1]
            sub_d = d[lb:i + 1]
            if len(sub_close) < order * 2 + 1:
                continue
            price_maxs = _find_local_extrema(sub_close, order, "max")
            dif_maxs = _find_local_extrema(sub_d, order, "max")
            if len(price_maxs) < 2 or len(dif_maxs) < 2:
                continue
            p1, p2 = price_maxs[-2], price_maxs[-1]
            d1, d2 = dif_maxs[-2], dif_maxs[-1]
            if np.isnan(sub_close[p1]) or np.isnan(sub_close[p2]):
                continue
            if np.isnan(sub_d[d1]) or np.isnan(sub_d[d2]):
                continue
            # 顶背离: 价格第二个高点更高，但 DIF 第二个高点更低
            if not (sub_close[p2] > sub_close[p1] and sub_d[d2] < sub_d[d1]):
                continue
            if abs(sub_d[d1]) < 1e-10:
                continue
            conf = round(min(1.0, abs(sub_d[d1] - sub_d[d2]) / abs(sub_d[d1])), 4)
            active_arr[i]     = True
            confidence_arr[i] = conf
            bar_count_arr[i]  = i - (lb + p1)

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
        }, index=df.index)

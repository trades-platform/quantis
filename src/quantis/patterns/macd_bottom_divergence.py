"""MACD 底背离：价格创新低但 DIF 未创新低，预示下跌动能衰竭。"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import macd


def _find_local_extrema(arr: np.ndarray, order: int = 5, mode: str = "min") -> List[int]:
    results = []
    for i in range(order, len(arr) - order):
        if np.isnan(arr[i]):
            continue
        window = arr[i - order: i + order + 1]
        if mode == "min" and arr[i] == np.nanmin(window):
            results.append(i)
        elif mode == "max" and arr[i] == np.nanmax(window):
            results.append(i)
    return results


@register_pattern
class MacdBottomDivergence(BasePattern):
    name = "macd_bottom_divergence"
    description = "价格创新低但DIF未创新低，预示下跌动能衰竭"
    confidence_desc = "DIF两低点差异占前低点绝对值的比例，越高背离越明显"
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
            price_mins = _find_local_extrema(sub_close, order, "min")
            dif_mins = _find_local_extrema(sub_d, order, "min")
            if len(price_mins) < 2 or len(dif_mins) < 2:
                continue
            p1, p2 = price_mins[-2], price_mins[-1]
            d1, d2 = dif_mins[-2], dif_mins[-1]
            if np.isnan(sub_close[p1]) or np.isnan(sub_close[p2]):
                continue
            if np.isnan(sub_d[d1]) or np.isnan(sub_d[d2]):
                continue
            # 底背离: 价格第二个低点更低，但 DIF 第二个低点更高
            if not (sub_close[p2] < sub_close[p1] and sub_d[d2] > sub_d[d1]):
                continue
            if abs(sub_d[d1]) < 1e-10:
                continue
            conf = round(min(1.0, abs(sub_d[d2] - sub_d[d1]) / abs(sub_d[d1])), 4)
            active_arr[i]     = True
            confidence_arr[i] = conf
            bar_count_arr[i]  = i - (lb + p1)

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
        }, index=df.index)

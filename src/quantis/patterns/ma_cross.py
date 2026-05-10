"""MA fast/slow 金叉/死叉 — 趋势转折信号。"""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import sma


@register_pattern
class MaCross(BasePattern):
    name = "ma_cross"
    description = "快均线上穿(金叉)或下穿(死叉)慢均线，趋势转折信号"
    confidence_desc = "交叉时均线斜率差与间距的综合强度，越大信号越强"
    default_params = {
        "fast": {"default": 20},
        "slow": {"default": 60},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        fast, slow = int(out["fast"]), int(out["slow"])
        if fast >= slow:
            raise ValueError("fast must be < slow")
        return {"fast": fast, "slow": slow}

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        close = df["close"]
        fast_ma = sma(close, params["fast"]).values
        slow_ma = sma(close, params["slow"]).values
        n = len(close)

        active_arr      = np.zeros(n, dtype=bool)
        confidence_arr  = np.zeros(n, dtype=float)
        bar_count_arr   = np.zeros(n, dtype=int)
        signal_type_arr = np.full(n, "", dtype=object)
        gap_arr         = np.full(n, np.nan)
        bars_since_arr  = np.zeros(n, dtype=int)

        last_cross_idx = -1

        for i in range(1, n):
            if np.isnan(fast_ma[i]) or np.isnan(slow_ma[i]):
                continue
            if np.isnan(fast_ma[i - 1]) or np.isnan(slow_ma[i - 1]):
                continue

            gap = (fast_ma[i] - slow_ma[i]) / slow_ma[i] if slow_ma[i] != 0 else 0.0
            gap_arr[i] = round(gap, 6)

            prev_above = fast_ma[i - 1] >= slow_ma[i - 1]
            curr_above = fast_ma[i] >= slow_ma[i]

            # 检测交叉
            crossed = False
            if prev_above and not curr_above:
                signal_type_arr[i] = "death_cross"
                crossed = True
            elif not prev_above and curr_above:
                signal_type_arr[i] = "golden_cross"
                crossed = True

            if crossed:
                # confidence: 基于交叉斜率差
                fast_slope = fast_ma[i] - fast_ma[i - 1]
                slow_slope = slow_ma[i] - slow_ma[i - 1]
                slope_diff = abs(fast_slope - slow_slope)
                ref_price = close.iloc[i] if hasattr(close, "iloc") else close[i]
                conf = min(slope_diff / ref_price * 100, 1.0) if ref_price > 0 else 0.0
                conf = round(max(conf, 0.1), 4)

                active_arr[i] = True
                confidence_arr[i] = conf
                bar_count_arr[i] = 1
                bars_since_arr[i] = 0
                last_cross_idx = i
            elif last_cross_idx >= 0:
                # 交叉后持续追踪
                bars_since = i - last_cross_idx
                # 仍维持交叉方向
                if (fast_ma[i] >= slow_ma[i]) == (fast_ma[last_cross_idx] >= slow_ma[last_cross_idx]):
                    active_arr[i] = True
                    bar_count_arr[i] = bar_count_arr[i - 1] + 1
                    confidence_arr[i] = round(confidence_arr[i - 1] * 0.95, 4)
                    signal_type_arr[i] = signal_type_arr[last_cross_idx]
                    bars_since_arr[i] = bars_since
                else:
                    last_cross_idx = -1

        return pd.DataFrame({
            "active":      active_arr,
            "confidence":  confidence_arr,
            "bar_count":   bar_count_arr,
            "signal_type": signal_type_arr,
            "gap_pct":     gap_arr,
            "bars_since":  bars_since_arr,
            "fast_period": np.full(n, params["fast"]),
            "slow_period": np.full(n, params["slow"]),
        }, index=df.index)

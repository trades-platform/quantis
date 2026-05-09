"""MACD 顶背离：价格创新高但 DIF 未创新高，预示上涨动能衰竭。"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from ..core.base import BasePattern
from ..core.registry import register_pattern
from ..utils.ta import macd
from .macd_bottom_divergence import _find_local_extrema, _swing_pct


@register_pattern
class MacdTopDivergence(BasePattern):
    name = "macd_top_divergence"
    description = "价格创新高但DIF未创新高，预示上涨动能衰竭"
    confidence_desc = "综合价格摆幅、DIF背离幅度及DIF位置的加权得分，越高背离越明显"
    default_params = {
        "fast":          {"default": 12},
        "slow":          {"default": 26},
        "signal":        {"default": 9},
        "order":         {"default": 5},
        "lookback":      {"default": 60},
        "min_swing_pct": {"default": 0.03},
        "cooldown":      {"default": 5},
    }

    def validate_params(self, params):
        out = super().validate_params(params)
        fast, slow = int(out["fast"]), int(out["slow"])
        if fast >= slow:
            raise ValueError("fast must be < slow")
        return {
            "fast":          fast,
            "slow":          slow,
            "signal":        int(out["signal"]),
            "order":         int(out["order"]),
            "lookback":      int(out["lookback"]),
            "min_swing_pct": float(out["min_swing_pct"]),
            "cooldown":      int(out["cooldown"]),
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        diff, _, _ = macd(df["close"], params["fast"], params["slow"], params["signal"])
        d = diff.values
        close = df["close"].values
        n = len(close)

        active_arr     = np.zeros(n, dtype=bool)
        confidence_arr = np.zeros(n, dtype=float)
        bar_count_arr  = np.zeros(n, dtype=int)

        order         = params["order"]
        lookback      = params["lookback"]
        min_swing_pct = params["min_swing_pct"]
        cooldown      = params["cooldown"]

        cooldown_remaining = 0

        for i in range(n):
            # --- cooldown logic ---
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                # Extend the active signal for a few bars after detection
                if cooldown_remaining + 1 < cooldown and bar_count_arr[i - 1] > 0:
                    active_arr[i] = True
                    bar_count_arr[i] = bar_count_arr[i - 1] + 1
                    confidence_arr[i] = confidence_arr[i - 1] * 0.80
                continue

            lb = max(0, i - lookback)
            sub_close = close[lb : i + 1]
            sub_d = d[lb : i + 1]
            if len(sub_close) < order * 2 + 1:
                continue

            # Find price highs as the anchor points
            price_maxs = _find_local_extrema(sub_close, order, "max")
            if len(price_maxs) < 2:
                continue

            p1, p2 = price_maxs[-2], price_maxs[-1]

            # --- Filter 1: minimum price swing magnitude ---
            if _swing_pct(sub_close, p1, p2) < min_swing_pct:
                continue

            # --- Filter 2: top divergence requires price making higher high ---
            if not (sub_close[p2] > sub_close[p1]):
                continue

            # --- Filter 3: use DIF values AT the price extrema (temporal alignment) ---
            if np.isnan(sub_d[p1]) or np.isnan(sub_d[p2]):
                continue

            # DIF at the second high must be lower than DIF at the first high -> divergence
            if not (sub_d[p2] < sub_d[p1]):
                continue

            # DIF at first high must be meaningful (not near-zero noise)
            dif_at_p1 = sub_d[p1]
            if abs(dif_at_p1) < 1e-8:
                continue

            # --- Confidence: multi-factor scoring ---
            # Factor A: DIF divergence magnitude (0..1)
            dif_divergence = (sub_d[p1] - sub_d[p2]) / abs(dif_at_p1)
            factor_a = min(1.0, abs(dif_divergence))

            # Factor B: price swing magnitude (0..1), scaled so that
            # min_swing_pct maps to 0 and 15% maps to 1
            price_swing = _swing_pct(sub_close, p1, p2)
            factor_b = min(1.0, price_swing / 0.15)

            # Factor C: DIF position relative to zero axis.
            # Top divergence is more meaningful when DIF is above zero (bullish momentum).
            # We don't hard-require it, but boost confidence when true.
            factor_c = 1.0
            if dif_at_p1 > 0 and sub_d[p2] > 0:
                factor_c = 1.0  # Both above zero — strong signal
            elif dif_at_p1 > 0:
                factor_c = 0.8  # First high above zero — decent
            else:
                factor_c = 0.5  # DIF below zero — weak context for top divergence

            # Weighted combination
            conf = round(
                min(1.0, 0.40 * factor_a + 0.35 * factor_b + 0.25 * factor_c),
                4,
            )

            active_arr[i]     = True
            confidence_arr[i] = conf
            bar_count_arr[i]  = i - (lb + p2) + 1  # bars since the second high confirmed
            cooldown_remaining = cooldown

        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
        }, index=df.index)

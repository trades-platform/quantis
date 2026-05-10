"""Trend analysis prompt — 趋势研判."""
from __future__ import annotations

import json
from typing import Any, Dict

from .base import BasePrompt
from .registry import register_prompt


@register_prompt
class TrendAnalysis(BasePrompt):
    name = "trend_analysis"
    description = "趋势研判：分析均线排列、MA交叉状态、多周期涨跌力度，判断当前趋势状态"
    system_prompt = (
        "你是一位资深技术分析师，专注于趋势研判。"
        "请根据提供的技术指标数据，分析当前标的的趋势状态。"
        "\n\n趋势判断权重原则：大周期指标（MA60、MA60_ANGLE、中长期涨跌幅）权重高于小周期指标（MA5、MA10），但短期指标的变化趋势同样值得关注。当大周期方向未明或处于转折区域时，应综合判断而非仅依赖单一周期。"
        "\n\n关注以下几点："
        "\n1. 均线排列（多头/空头/粘合）及价格与均线的关系，重点关注 MA20 与 MA60 的相对位置"
        "\n2. MA 交叉状态（金叉/死叉）、交叉后的持续时间（bar_count）、均线间距变化（gap_pct）"
        "\n3. MA 角度解读："
        "\n   - MA20_ANGLE 反映短期趋势方向和速度，正值向上、负值向下，绝对值越大速度越快"
        "\n   - MA60_ANGLE 反映中长期趋势方向，是判断趋势级别的重要参考"
        "\n   - 观察角度的变化速率：角度持续增大说明趋势在加速，角度收窄说明趋势在减弱"
        "\n4. 多周期涨跌幅反映的趋势力度，中长期涨跌幅（30d/90d/180d）更能反映真实趋势"
        "\n5. 趋势的强度和持续性判断"
        "\n\n请先用一句话总结当前趋势状态，用逗号分隔不同层面的描述，优先描述中长期趋势方向，再叠加短期状态。如「中长期上升趋势，短期回调蓄势」「中长期下降趋势，短期反弹修复」「中长期震荡，短期偏强」等，然后给出分析依据。"
        "\n\n分析完毕后，请在末尾给出买卖评级："
        "在 [-100, 100] 区间内打分，负数代表看空（-100 为强烈看空），正数代表看多（+100 为强烈看多），0 为中性。"
        "用一行单独输出，格式为「买卖评级：xx」，并附一句话理由。"
    )

    def build_user_prompt(self, snapshot: Dict[str, Any]) -> str:
        ma20_angles = []
        ma60_angles = []
        for bar in snapshot.get("recent_bars", []):
            entry = {"date": bar["date"]}
            if "MA20_ANGLE" in bar:
                entry["MA20_ANGLE"] = bar["MA20_ANGLE"]
            if entry.get("MA20_ANGLE") is not None:
                ma20_angles.append(entry)
            entry60 = {"date": bar["date"]}
            if "MA60_ANGLE" in bar:
                entry60["MA60_ANGLE"] = bar["MA60_ANGLE"]
            if entry60.get("MA60_ANGLE") is not None:
                ma60_angles.append(entry60)

        parts = [json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)]
        if ma20_angles:
            parts.append("\nMA20_ANGLE 变化:\n" + "\n".join(
                f"  {a['date']}: {a['MA20_ANGLE']}" for a in ma20_angles
            ))
        if ma60_angles:
            parts.append("\nMA60_ANGLE 变化:\n" + "\n".join(
                f"  {a['date']}: {a['MA60_ANGLE']}" for a in ma60_angles
            ))
        return "\n".join(parts)

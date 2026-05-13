"""Reversal detection prompt — 反转信号研判."""

from .base import BasePrompt
from .registry import register_prompt


@register_prompt
class ReversalDetection(BasePrompt):
    name = "reversal_detection"
    description = "反转检测：分析背离、超买超卖、形态信号，识别潜在趋势反转"
    specs = [
        "indicator:ma",
        "indicator:ma_angle",
        "indicator:macd",
        "indicator:boll",
        "indicator:rsi",
        "indicator:volume",
        "pattern:ma_cross",
        "pattern:ma_fallback",
        "pattern:macd_bottom_divergence",
        "pattern:macd_top_divergence",
    ]
    recent_bars = 21
    field_schema = (
        "数据格式说明："
        "\n- 标的：代码 名称"
        "\n- 最新K线：D=日期 O=开 H=高 L=低 C=收 V=量 ATR"
        "\n- 涨跌幅：chgpct_Nd=N日涨跌百分比"
        "\n- 指标：当前最新值（JSON），含 MACD 相关、RSI、BOLL、成交量均线等"
        "\n- 活跃形态：pattern名（描述） {bar_count, confidence, signal_type等}"
        "\n  关注 macd_bottom_divergence/macd_top_divergence 等反转形态"
        "\n- 不活跃：当前未触发的形态列表"
        "\n- 近期走势：TSV格式，每行一根K线"
        "\n- 阶段：历史形态触发区间 conf=置信度变化"
    )
    system_prompt = (
        "你是一位资深技术分析师，专注于识别趋势反转信号。"
        "请根据提供的技术指标和形态检测结果，判断当前标的是否存在反转迹象。"
        "关注以下几点："
        "1. 活跃的反转形态（MACD 背离、均线死叉/金叉等）及其置信度"
        "2. RSI 是否处于超买/超卖区域"
        "3. 布林带位置——价格是否触及上轨/下轨"
        "4. 成交量是否配合反转信号"
        "请明确指出是否存在反转风险或反转机会，并给出可靠度评估。"
        "\n\n分析完毕后，请在末尾给出买入评级："
        "评级分为五档：强烈推荐买入 / 建议买入 / 中性 / 建议观望 / 建议回避。"
        "用一行单独输出，格式为「买入评级：xxx」，并附一句话理由。"
    )

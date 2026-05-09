# Quantis — OHLCV Analysis Engine

## 项目定位

独立的技术分析库，接收标准 OHLCV DataFrame，输出指标序列或形态检测结果。
支持两种输出模式：**每根历史 bar** 或 **仅最后一根 bar**。

---

## 目录结构

```
src/quantis/
├── core/
│   ├── types.py        OutputMode, AnalysisSpec, PhaseResult, AnalysisResult
│   ├── base.py         BaseAnalyzer → BaseIndicator / BasePattern
│   ├── registry.py     _Registry + @register_indicator / @register_pattern
│   └── engine.py       AnalysisEngine.run() + analyze() 便捷函数
├── utils/
│   ├── validation.py   ensure_ohlcv() — 列检查、类型校验、列名归一化
│   └── ta.py           sma / ema / macd / bbands / rsi / atr（纯 pandas/numpy）
├── indicators/         ma, ema, macd, boll, rsi, atr, volume
└── patterns/           macd_dif_return_to_zero, ma_fallback, low_volume_pullback_ma
tests/
├── conftest.py         200-bar 合成 OHLCV fixture
├── test_indicators.py
├── test_patterns.py
└── test_engine.py
```

---

## 核心概念

### OutputMode

```python
class OutputMode(str, Enum):
    SERIES = "series"   # 每根 bar 都有结果 → pd.DataFrame
    LAST   = "last"     # 仅最后一根 bar   → dict
```

### 类层级

```
BaseAnalyzer          validate_params() / compute() / last() / run()
  ├── BaseIndicator   kind = "indicator"
  └── BasePattern     kind = "pattern"
                      额外提供 summarize_phase() → PhaseResult
```

### 合约

- `compute(df, params) -> pd.DataFrame`：列与 `df.index` 对齐，前缀 NaN 表示预热期。
- `Indicator.compute` 输出数值列（如 `MA20`、`DIFF`、`BOLL_UPPER`）。
- `Pattern.compute` 必须含 `active(bool)` / `confidence(float)` / `bar_count(int)` 列，可附加任意额外列。
- `OutputMode.LAST` 时 Indicator 返回 `.iloc[-1]` 字典，Pattern 返回 `PhaseResult.to_dict()`。

### PhaseResult（形态聚合）

```python
@dataclass
class PhaseResult:
    active: bool
    pattern: str
    start_position: int      # 形态起始 bar 位置
    start_index: Any         # 对应 df.index 值
    current_position: int    # 当前（最后）bar 位置
    current_index: Any
    bar_count: int           # 持续 bar 数
    confidence: float        # [0, 1]
    extra: dict              # pattern 专属字段
```

### AnalysisResult

```python
result = analyze(df, specs, mode)
result["ma"]          # pd.DataFrame 或 dict，取决于 mode
result.get("boll")    # 安全访问
"macd" in result      # 包含检查
```

---

## 使用示例

```python
from quantis import analyze, OutputMode

# SERIES 模式 — 每根 bar
r = analyze(df, [
    "indicator:ma",
    {"name": "macd", "params": {"fast": 5, "slow": 10, "signal": 3}, "alias": "macd_fast"},
    "pattern:ma_fallback",
], mode=OutputMode.SERIES)

r["ma"]          # pd.DataFrame: MA5 MA10 MA20 MA60
r["macd_fast"]   # pd.DataFrame: DIFF DEA HIST
r["ma_fallback"] # pd.DataFrame: active confidence bar_count target_ma

# LAST 模式 — 仅最后一根 bar
r = analyze(df, ["indicator:ma", "pattern:ma_fallback"], mode="last")
r["ma"]          # {"MA5": 123.4, "MA20": 121.1, ...}
r["ma_fallback"] # {"active": True, "confidence": 0.87, "bar_count": 5, "extra": {...}}
```

---

## 扩展指南

### 添加新指标

```python
# src/quantis/indicators/my_indicator.py
from ..core.base import BaseIndicator
from ..core.registry import register_indicator

@register_indicator
class MyIndicator(BaseIndicator):
    name = "my_indicator"
    description = "..."
    default_params = {"period": {"default": 14}}

    def compute(self, df, params):
        # 返回与 df.index 对齐的 DataFrame
        ...
```

在 `src/quantis/indicators/__init__.py` 加一行 import 触发注册。

### 添加新形态

```python
# src/quantis/patterns/my_pattern.py
from ..core.base import BasePattern
from ..core.registry import register_pattern

@register_pattern
class MyPattern(BasePattern):
    name = "my_pattern"
    default_params = {"period": {"default": 20}}

    def compute(self, df, params):
        # 必须含 active / confidence / bar_count 三列
        return pd.DataFrame({
            "active":     active_arr,
            "confidence": confidence_arr,
            "bar_count":  bar_count_arr,
            # 自定义额外列 → 自动进入 PhaseResult.extra
            "signal_type": signal_arr,
        }, index=df.index)
```

在 `src/quantis/patterns/__init__.py` 加 import。

---

## 依赖

- `numpy >= 1.23`
- `pandas >= 2.0`
- 无 TA-Lib / tulipy 依赖，TA 原语用纯 pandas/numpy 实现（`utils/ta.py`）

## 开发

```bash
pip install -e ".[dev]"
python -m pytest -v
```

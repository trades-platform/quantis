# Quantis — OHLCV Analysis Engine

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

## 输入 DataFrame 格式

所有 analyzer 和 `snapshot()` 接收的 DataFrame 必须符合以下结构：

```python
import pandas as pd

df = pd.DataFrame(
    {
        "open":   [...],   # float, 开盘价
        "high":   [...],   # float, 最高价
        "low":    [...],   # float, 最低价
        "close":  [...],   # float, 收盘价
        "volume": [...],   # float, 成交量
    },
    index=pd.DatetimeIndex([...]),  # 必须是 datetime 类型
)

# 可选：标的元信息
df.attrs["code"] = "159985.SZ"
df.attrs["name"] = "豆粕ETF华夏"
```

| 要求 | 说明 |
|------|------|
| 列名 | `open`, `high`, `low`, `close`, `volume`（小写） |
| index | `pd.DatetimeIndex`，时区无关即可 |
| 数据类型 | OHLCV 均为 `float` / `int`，不可含字符串 |
| 最少行数 | 需大于所使用指标的预热期（通常 ≥ 60 根 bar） |
| `df.attrs` | 可选，可包含 `"code"` (str)、`"name"` (str)、`"period"` (str/int)，`snapshot()` 会读取并输出 |

### period 取值

`df.attrs["period"]` 表示 K 线周期，仅用于记录，`snapshot()` 输出时透传。多周期涨跌幅 `chgpct_Nd` 始终按 index 时间差计算（`pd.Timedelta(days=N)` + `ffill`），不受 period 影响。

| 值 | 含义 |
|----|------|
| `1`, `5`, `15`, `30`, `60`, `120` | 分钟级 K 线 |
| `"daily"` | 日线 |
| `"weekly"` | 周线 |
| `"monthly"` | 月线 |

`ensure_ohlcv()`（`utils/validation.py`）会在 engine 入口做列检查、类型校验和列名归一化。

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

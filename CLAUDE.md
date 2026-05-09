# Quantis — OHLCV Analysis Engine

> 本项目所有指标和形态分析结果最终发送给 LLM API 进行综合研判，不直接产生交易信号。

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

### Git Commit 规范

- commit message 必须带 `Signed-off-by:` 行（`git commit -s`）
- 不带 `Co-Authored-By` 或其他 AI 协作标记

from __future__ import annotations

import json
from typing import Any, Dict, List


class BasePrompt:
    name: str = ""
    description: str = ""
    system_prompt: str = ""
    field_schema: str = ""
    preamble_ack: str = "好的，我将按照上述要求分析以下数据。"
    specs: list = []
    recent_bars: int = 21

    def build_user_prompt(self, snapshot: Dict[str, Any]) -> str:
        """Build user message from snapshot data — compact text format."""
        return _compact_snapshot(snapshot)

    def build(self, snapshot: Dict[str, Any]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if self.field_schema:
            messages.append({"role": "user", "content": self.field_schema})
            messages.append({"role": "assistant", "content": self.preamble_ack})
        messages.append({"role": "user", "content": self.build_user_prompt(snapshot)})
        return messages


def _compact_snapshot(snap: Dict[str, Any]) -> str:
    """Convert snapshot dict to compact text — TSV for recent_bars, key-value for rest."""
    parts: List[str] = []

    sym = snap.get("symbol", "")
    name = snap.get("name", "")
    if sym:
        parts.append(f"标的：{sym} {name}")

    # last_bar — single line
    lb = snap.get("last_bar", {})
    bar_fields = []
    for k in ("date", "open", "high", "low", "close", "volume"):
        if k in lb:
            bar_fields.append(f"{k[0].upper()}={lb[k]}")
    if "atr" in lb:
        bar_fields.append(f"ATR={lb['atr']}")
    parts.append(" | ".join(bar_fields))

    if "changes" in lb:
        chg = " ".join(f"{k}={v}%" for k, v in lb["changes"].items())
        parts.append(f"涨跌幅：{chg}")

    # indicators — one-line JSON (no indent)
    ind = snap.get("indicators", {})
    if ind:
        parts.append(f"指标：{json.dumps(ind, ensure_ascii=False)}")

    # active patterns
    for p in snap.get("active_patterns", []):
        desc = p.get("description", "")
        extra = {k: v for k, v in p.items() if k not in ("pattern", "description")}
        line = f"活跃：{p['pattern']}"
        if desc:
            line += f"（{desc}）"
        if extra:
            line += f" {json.dumps(extra, ensure_ascii=False)}"
        parts.append(line)

    # inactive patterns — just names
    ip = snap.get("inactive_patterns", [])
    if ip:
        parts.append("不活跃：" + ", ".join(p.get("pattern", "") for p in ip))

    # recent_bars — TSV (biggest savings here)
    rb = snap.get("recent_bars", [])
    if rb:
        cols = [k for k in rb[0].keys() if k != "date"]
        header = "date\t" + "\t".join(cols)
        rows = []
        for bar in rb:
            vals = [str(bar.get("date", ""))]
            for c in cols:
                v = bar.get(c)
                vals.append("" if v is None else str(v))
            rows.append("\t".join(vals))
        parts.append(f"近期走势({len(rb)}根)：\n{header}\n" + "\n".join(rows))

    # pattern phases
    for ph in snap.get("pattern_phases", []):
        line = f"阶段：{ph['pattern']} {ph['start']}~{ph['end']} len={ph['length']}"
        if "confidence_start" in ph:
            line += f" conf={ph['confidence_start']}→{ph['confidence_end']}"
        extra = {k: v for k, v in ph.items()
                 if k not in ("pattern", "start", "end", "length", "confidence_start", "confidence_end")}
        if extra:
            line += f" {json.dumps(extra, ensure_ascii=False)}"
        parts.append(line)

    return "\n".join(parts)

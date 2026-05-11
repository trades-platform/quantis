"""Quantis + LLM analysis — single or batch mode.

Usage:
    # Single code
    python scripts/llm_analysis.py 588000 120 --llm deepseek

    # Batch multiple codes (concurrent)
    python scripts/llm_analysis.py 688036 002594 300122 --period 120 --llm deepseek

    # Batch from file (one code per line)
    python scripts/llm_analysis.py --codes-file watchlist.txt --period 120 --llm deepseek

    # With end-time
    python scripts/llm_analysis.py 688036 120 --llm deepseek --end-time 2026-04-21
"""
import argparse
import asyncio
import json
import os
import re
import sys
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from quantis import fetch_klines, snapshot, get_prompt, list_prompts

# LLM 提供商配置（API key 从环境变量读取）
LLM_PROVIDERS = {
    "zhipu": {
        "api_key": os.environ.get("ZHIPU_API_KEY", ""),
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "model": "glm-5-turbo",
    },
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
}

# 各 prompt 专用指标配置
PROMPT_SPECS = {
    "trend_analysis": {
        "specs": [
            "indicator:ma",
            "indicator:ma_angle",
            "indicator:volume",
            "indicator:boll",
            "pattern:ma_cross",
            "pattern:ma_fallback",
        ],
        "recent_bars": 21,
    },
    "short_term": {
        "specs": [
            "indicator:ma",
            "indicator:ma_angle",
            "indicator:boll",
            "indicator:bias",
            "indicator:rsi",
            "pattern:ma_cross",
            "pattern:ma_fallback",
        ],
        "recent_bars": 55,
    },
}


def _extract_result(text: str) -> Dict:
    """Extract trend summary and rating from LLM response."""
    trend = ""
    for line in text.split("\n"):
        line = line.strip()
        if line and not trend:
            trend = line
    m = re.search(r"买卖评级[：:\s]*([-+]?\d+)", text)
    rating = int(m.group(1)) if m else None
    return {"trend": trend, "rating": rating}


async def analyze_one(
    code: str,
    period,
    prompt_name: str,
    llm: str,
    count: int,
    start_time: Optional[str],
    end_time: Optional[str],
    api_key: Optional[str],
    no_llm: bool,
) -> Dict:
    """Analyze a single code — runs in thread to avoid blocking."""
    loop = asyncio.get_event_loop()

    def _fetch():
        return fetch_klines(code, period=period, count=count,
                            start_time=start_time, end_time=end_time, api_key=api_key)

    def _snap(df):
        cfg = PROMPT_SPECS.get(prompt_name, PROMPT_SPECS["trend_analysis"])
        return snapshot(df, cfg["specs"], recent_bars=cfg.get("recent_bars", 21))

    df = await loop.run_in_executor(None, _fetch)
    snap = await loop.run_in_executor(None, _snap, df)
    snap["period"] = str(period)

    if no_llm:
        return {"code": code, "name": snap.get("name", ""), "snapshot": snap}

    prompt_obj = get_prompt(prompt_name)
    messages = prompt_obj.build(snap)

    provider = LLM_PROVIDERS[llm]
    client = AsyncOpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
    resp = await client.chat.completions.create(
        model=provider["model"],
        messages=messages,
        temperature=0.3,
        max_tokens=4096,
    )
    text = resp.choices[0].message.content
    result = _extract_result(text)
    result["code"] = code
    result["name"] = snap.get("name", "")
    result["full_text"] = text
    return result


async def run_batch(
    codes: List[str],
    period,
    prompt_name: str,
    llm: str,
    count: int,
    start_time: Optional[str],
    end_time: Optional[str],
    api_key: Optional[str],
    no_llm: bool,
    concurrency: int,
):
    sem = asyncio.Semaphore(concurrency)

    async def limited(code):
        async with sem:
            return await analyze_one(
                code, period, prompt_name, llm, count,
                start_time, end_time, api_key, no_llm,
            )

    tasks = [limited(code) for code in codes]
    return await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description="Quantis + LLM analysis")
    parser.add_argument("codes", nargs="*", help="Stock code(s), e.g. 588000 688036 002594")
    parser.add_argument("--codes-file", default=None, help="File with one code per line")
    parser.add_argument("-p", "--period", default="daily",
                        help="K-line period: 5, 15, 60, 120, daily, weekly, monthly")
    parser.add_argument("--prompt", default="trend_analysis",
                        help=f"Prompt: {', '.join(list_prompts())}")
    parser.add_argument("--llm", default="deepseek", choices=list(LLM_PROVIDERS))
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--api-key", default=None, help="TickFlow API key")
    parser.add_argument("--no-llm", action="store_true", help="Only print snapshot")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent LLM requests")
    args = parser.parse_args()

    # Collect codes
    codes = list(args.codes or [])
    if args.codes_file:
        with open(args.codes_file) as f:
            codes.extend(line.strip() for line in f if line.strip())
    if not codes:
        parser.error("No codes provided. Pass codes as args or --codes-file.")

    try:
        period = int(args.period)
    except ValueError:
        period = args.period

    # Single code — print full output
    if len(codes) == 1:
        result = asyncio.run(analyze_one(
            codes[0], period, args.prompt, args.llm, args.count,
            args.start_time, args.end_time, args.api_key, args.no_llm,
        ))
        if args.no_llm:
            print(json.dumps(result["snapshot"], ensure_ascii=False, indent=2, default=str))
        else:
            print(result["full_text"])
        return

    # Batch mode — summary table
    print(f"Analyzing {len(codes)} codes ({period}) with {args.llm}...")
    results = asyncio.run(run_batch(
        codes, period, args.prompt, args.llm, args.count,
        args.start_time, args.end_time, args.api_key, args.no_llm,
        args.concurrency,
    ))

    # Group by trend keyword
    groups: Dict[str, list] = {}
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}", file=sys.stderr)
            continue
        trend = r.get("trend", "")
        if "上升" in trend:
            key = "上升"
        elif "下降" in trend:
            key = "下降"
        else:
            key = "震荡"
        groups.setdefault(key, []).append(r)

    # Print grouped table
    for group_name in ["上升", "震荡", "下降"]:
        items = groups.get(group_name, [])
        if not items:
            continue
        print(f"\n{'='*60}")
        print(f" {group_name}")
        print(f"{'='*60}")
        for r in sorted(items, key=lambda x: x.get("rating") or 0, reverse=True):
            rating = r.get("rating", "")
            rating_str = f"{rating:+d}" if rating is not None else "—"
            print(f"  {r['code']:>6s} {r.get('name', ''):<8s} | {rating_str:>4s} | {r.get('trend', '')[:50]}")
    print()


if __name__ == "__main__":
    main()

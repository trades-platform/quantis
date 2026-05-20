"""Quantis + LLM analysis — CLI entry point.

Usage:
    # Single code
    python scripts/llm_analysis.py 588000 -p 120 --llm deepseek

    # Batch multiple codes
    python scripts/llm_analysis.py 688036 002594 300122 -p 120 --llm deepseek --api-key tk_...

    # Batch from file
    python scripts/llm_analysis.py --codes-file watchlist.txt -p 120 --llm deepseek

    # With end-time
    python scripts/llm_analysis.py 688036 -p 120 --llm deepseek --end-time 2026-04-21
"""
import argparse
import asyncio
import json
import sys

from quantis import AnalysisAgent, fetch_klines, list_prompts
from quantis.agents.providers import DEFAULT_PROVIDERS


def main():
    parser = argparse.ArgumentParser(description="Quantis + LLM analysis")
    parser.add_argument("codes", nargs="*", help="Stock code(s)")
    parser.add_argument("--codes-file", default=None, help="File with one code per line")
    parser.add_argument("-p", "--period", default="daily")
    parser.add_argument("--prompt", default="trend_analysis",
                        help=f"Prompt: {', '.join(list_prompts())}")
    parser.add_argument("--llm", default="deepseek", choices=list(DEFAULT_PROVIDERS))
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--api-key", default=None, help="TickFlow API key")
    parser.add_argument("--no-llm", action="store_true", help="Only print snapshot")
    parser.add_argument("--chunk-size", type=int, default=5, help="Codes per cache session chunk")
    parser.add_argument("--session", action="store_true", help="Enable session caching mode")
    parser.add_argument("--no-session", action="store_true",
                        help=argparse.SUPPRESS)  # deprecated: now the default
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

    agent = AnalysisAgent(
        args.prompt,
        llm=args.llm,
        session_mode=args.session,
    )
    if args.no_session:
        print("Warning: --no-session is deprecated, stateless is now the default",
              file=sys.stderr)

    if len(codes) == 1:
        df = fetch_klines(
            codes[0], period=period, count=args.count,
            start_time=args.start_time, end_time=args.end_time,
            api_key=args.api_key,
        )
        result = asyncio.run(agent.analyze(df, no_llm=args.no_llm))
        if args.no_llm:
            print(json.dumps(result.snapshot, ensure_ascii=False, indent=2, default=str))
        else:
            print(result.full_text)
        return

    # Batch mode
    print(f"Analyzing {len(codes)} codes ({period}) with {args.llm}...")
    dfs = [
        fetch_klines(
            c, period=period, count=args.count,
            start_time=args.start_time, end_time=args.end_time,
            api_key=args.api_key,
        )
        for c in codes
    ]
    batch = asyncio.run(agent.analyze_batch(
        dfs, chunk_size=args.chunk_size, no_llm=args.no_llm,
    ))

    # Group by trend keyword and print
    groups: dict[str, list] = {"上升": [], "震荡": [], "下降": []}
    for r in batch.successful:
        if "上升" in r.trend:
            groups["上升"].append(r)
        elif "下降" in r.trend:
            groups["下降"].append(r)
        else:
            groups["震荡"].append(r)

    for group_name in ("上升", "震荡", "下降"):
        items = groups.get(group_name, [])
        if not items:
            continue
        print(f"\n{'='*60}")
        print(f" {group_name}")
        print(f"{'='*60}")
        for r in sorted(items, key=lambda x: x.rating or 0, reverse=True):
            rating_str = f"{r.rating:+d}" if r.rating is not None else "—"
            print(f"  {r.code:>10s} {r.name:<8s} | {rating_str:>4s} | {r.trend[:50]}")

    for exc in batch.errors:
        print(f"  ERROR: {exc}", file=sys.stderr)

    # Cache stats
    pt = batch.total_prompt_tokens
    cht = batch.total_cache_hit_tokens
    if pt > 0:
        hit_rate = cht / pt * 100
        print(f"\n  Cache: {cht}/{pt} tokens hit ({hit_rate:.1f}%)")


if __name__ == "__main__":
    main()

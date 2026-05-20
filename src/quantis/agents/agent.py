"""AnalysisAgent — strategy-specialized LLM analysis agent."""
from __future__ import annotations

import asyncio
from typing import List, Optional

import pandas as pd
from openai import AsyncOpenAI

from ..core.snapshot import snapshot
from ..prompts.registry import get_prompt
from .providers import LLMProvider, get_provider
from .result import SingleResult, BatchResult, extract_result


class AnalysisAgent:
    """Agent that analyzes OHLCV DataFrames using a specific strategy prompt.

    Parameters
    ----------
    prompt_name : str
        Registered prompt name (e.g. ``"trend_analysis"``, ``"short_term"``).
    llm : str
        Built-in provider name (``"deepseek"`` or ``"zhipu"``).
    provider : LLMProvider, optional
        Direct provider object — takes precedence over *llm*.
    session_mode : bool
        If ``True``, accumulate conversation for prefix cache optimization.
    temperature : float
        LLM sampling temperature.
    max_tokens : int
        Max output tokens per request.
    """

    def __init__(
        self,
        prompt_name: str = "trend_analysis",
        *,
        llm: str = "deepseek",
        provider: Optional[LLMProvider] = None,
        session_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> None:
        self._prompt = get_prompt(prompt_name)
        self._session_mode = session_mode
        self._temperature = temperature
        self._max_tokens = max_tokens

        prov = provider or get_provider(llm)
        self._client = AsyncOpenAI(
            api_key=prov.api_key,
            base_url=prov.base_url,
        )
        self._model = prov.model

        # Session state
        self._messages: list[dict[str, str]] = []
        self._session_started = False

    # ── public API ──────────────────────────────────────────────────

    @property
    def prompt_name(self) -> str:
        return self._prompt.name

    @property
    def model(self) -> str:
        return self._model

    async def analyze(
        self,
        df: pd.DataFrame,
        *,
        no_llm: bool = False,
    ) -> SingleResult:
        """Analyze a single OHLCV DataFrame.

        Parameters
        ----------
        df : DataFrame with open/high/low/close/volume columns and datetime index.
            ``df.attrs`` may contain ``"code"``, ``"name"``, ``"period"``.
        no_llm : bool
            If ``True``, return snapshot only without calling the LLM.
        """
        loop = asyncio.get_event_loop()
        snap = await loop.run_in_executor(
            None,
            lambda: snapshot(
                df, self._prompt.specs, recent_bars=self._prompt.recent_bars
            ),
        )
        snap["period"] = str(df.attrs.get("period", ""))

        code = snap.get("symbol", df.attrs.get("code", ""))
        name = snap.get("name", df.attrs.get("name", ""))

        if no_llm:
            return SingleResult(code=code, name=name, snapshot=snap)

        if self._session_mode:
            return await self._analyze_session(snap, code, name)
        return await self._analyze_stateless(snap, code, name)

    async def analyze_batch(
        self,
        dfs: list[pd.DataFrame],
        *,
        chunk_size: int = 5,
        summary: bool = False,
        no_llm: bool = False,
    ) -> BatchResult:
        """Analyze multiple DataFrames.

        In session mode, splits into chunks of *chunk_size*: each chunk is
        processed serially (maximizing prefix cache), chunks run in parallel.
        In stateless mode, all calls run concurrently.
        """
        if self._session_mode and not no_llm:
            return await self._batch_chunked(dfs, chunk_size, no_llm)
        return await self._batch_stateless(dfs, no_llm)

    def reset_session(self) -> None:
        self._messages = []
        self._session_started = False

    def analyze_sync(self, df: pd.DataFrame, **kwargs) -> SingleResult:
        return asyncio.run(self.analyze(df, **kwargs))

    def analyze_batch_sync(
        self, dfs: list[pd.DataFrame], **kwargs
    ) -> BatchResult:
        return asyncio.run(self.analyze_batch(dfs, **kwargs))

    # ── internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_usage(resp) -> tuple[int, int]:
        usage = resp.usage
        pt = usage.prompt_tokens if usage else 0
        cht = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
        return pt, cht

    def _build_result(
        self, text: str, code: str, name: str, pt: int, cht: int,
    ) -> SingleResult:
        parsed = extract_result(text)
        return SingleResult(
            code=code,
            name=name,
            trend=parsed["trend"],
            rating=parsed["rating"],
            action_hint=parsed["action_hint"],
            full_text=text,
            prompt_tokens=pt,
            cache_hit_tokens=cht,
        )

    # ── session helpers ─────────────────────────────────────────────

    def _ensure_session_prefix(self) -> None:
        if self._session_started:
            return
        self._messages = self._build_prefix()
        self._session_started = True

    def _build_prefix(self) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self._prompt.system_prompt}]
        if self._prompt.field_schema:
            msgs.append({"role": "user", "content": self._prompt.field_schema})
            msgs.append({"role": "assistant", "content": self._prompt.preamble_ack})
        return msgs

    async def _analyze_session(
        self, snap: dict, code: str, name: str
    ) -> SingleResult:
        self._ensure_session_prefix()
        data_msg = self._prompt.build_user_prompt(snap)
        self._messages.append({"role": "user", "content": data_msg})

        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=list(self._messages),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = resp.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": text})

        pt, cht = self._extract_usage(resp)
        return self._build_result(text, code, name, pt, cht)

    async def _analyze_stateless(
        self, snap: dict, code: str, name: str
    ) -> SingleResult:
        messages = self._prompt.build(snap)
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        text = resp.choices[0].message.content or ""

        pt, cht = self._extract_usage(resp)
        return self._build_result(text, code, name, pt, cht)

    # ── batch helpers ───────────────────────────────────────────────

    async def _batch_chunked(
        self, dfs: list[pd.DataFrame], chunk_size: int, no_llm: bool
    ) -> BatchResult:
        chunks: list[list[pd.DataFrame]] = [
            dfs[i : i + chunk_size] for i in range(0, len(dfs), chunk_size)
        ]

        async def _process_chunk(
            chunk: list[pd.DataFrame],
        ) -> list[SingleResult | Exception]:
            results: list[SingleResult | Exception] = []
            # Each chunk gets its own session
            messages = list(self._build_prefix())
            for df in chunk:
                try:
                    loop = asyncio.get_event_loop()
                    snap = await loop.run_in_executor(
                        None,
                        lambda _df=df: snapshot(
                            _df, self._prompt.specs,
                            recent_bars=self._prompt.recent_bars,
                        ),
                    )
                    snap["period"] = str(df.attrs.get("period", ""))
                    code = snap.get("symbol", df.attrs.get("code", ""))
                    name = snap.get("name", df.attrs.get("name", ""))

                    if no_llm:
                        results.append(SingleResult(code=code, name=name, snapshot=snap))
                        continue

                    data_msg = self._prompt.build_user_prompt(snap)
                    messages.append({"role": "user", "content": data_msg})

                    resp = await self._client.chat.completions.create(
                        model=self._model,
                        messages=list(messages),
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                    text = resp.choices[0].message.content or ""
                    messages.append({"role": "assistant", "content": text})

                    pt, cht = self._extract_usage(resp)
                    results.append(self._build_result(text, code, name, pt, cht))
                except Exception as exc:
                    results.append(exc)
            return results

        all_chunks = await asyncio.gather(
            *[_process_chunk(c) for c in chunks]
        )
        flat = [r for chunk in all_chunks for r in chunk]
        return BatchResult(
            results=flat,
            prompt=self._prompt.name,
        )

    async def _batch_stateless(
        self, dfs: list[pd.DataFrame], no_llm: bool
    ) -> BatchResult:
        sem = asyncio.Semaphore(5)

        async def _limited(df: pd.DataFrame) -> SingleResult | Exception:
            async with sem:
                try:
                    return await self.analyze(df, no_llm=no_llm)
                except Exception as exc:
                    return exc

        results = await asyncio.gather(*[_limited(df) for df in dfs])
        return BatchResult(
            results=list(results),
            prompt=self._prompt.name,
        )

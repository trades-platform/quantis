"""LLM provider configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass
class LLMProvider:
    api_key: str
    base_url: str
    model: str


DEFAULT_PROVIDERS: dict[str, LLMProvider] = {
    "deepseek": LLMProvider(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
    ),
    "zhipu": LLMProvider(
        api_key=os.environ.get("ZHIPU_API_KEY", ""),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-5-turbo",
    ),
}


def get_provider(
    name: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> LLMProvider:
    """Resolve a provider by name with optional overrides.

    Accepts built-in names (``"deepseek"``, ``"zhipu"``) or arbitrary names
    when *api_key*, *base_url*, and *model* are all provided.
    """
    base = DEFAULT_PROVIDERS.get(name)
    if base is None:
        if not api_key or not base_url or not model:
            raise ValueError(
                f"Unknown provider '{name}'. "
                "Provide api_key, base_url, and model for custom providers."
            )
        return LLMProvider(api_key=api_key, base_url=base_url, model=model)

    overrides: dict = {}
    if api_key:
        overrides["api_key"] = api_key
    if base_url:
        overrides["base_url"] = base_url
    if model:
        overrides["model"] = model
    return replace(base, **overrides) if overrides else base

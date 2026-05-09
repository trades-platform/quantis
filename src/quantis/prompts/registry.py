from __future__ import annotations

from ..core.registry import _Registry
from .base import BasePrompt

PROMPTS = _Registry("prompt", BasePrompt)


def register_prompt(cls):
    return PROMPTS.register(cls)


def get_prompt(name: str) -> BasePrompt:
    return PROMPTS.get(name)


def list_prompts() -> list:
    return PROMPTS.list()

from __future__ import annotations
from typing import Dict, Type
from .base import BaseAnalyzer, BaseIndicator, BasePattern


class _Registry:
    def __init__(self, label: str, base_cls: type):
        self.label = label
        self.base_cls = base_cls
        self._items: Dict[str, Type[BaseAnalyzer]] = {}

    def register(self, cls: Type[BaseAnalyzer]) -> Type[BaseAnalyzer]:
        if not issubclass(cls, self.base_cls):
            raise TypeError(f"{cls.__name__} must subclass {self.base_cls.__name__}")
        if not cls.name:
            raise ValueError(f"{cls.__name__} must define a non-empty class attr 'name'")
        if cls.name in self._items:
            raise ValueError(f"{self.label} '{cls.name}' is already registered")
        self._items[cls.name] = cls
        return cls

    def get(self, name: str) -> BaseAnalyzer:
        cls = self._items.get(name)
        if cls is None:
            available = ", ".join(sorted(self._items)) or "(none)"
            raise ValueError(f"Unknown {self.label}: '{name}'. Registered: {available}")
        return cls()

    def list(self) -> list:
        return sorted(self._items.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def items(self):
        return self._items.items()


INDICATORS = _Registry("indicator", BaseIndicator)
PATTERNS   = _Registry("pattern",   BasePattern)


def register_indicator(cls): return INDICATORS.register(cls)
def register_pattern(cls):   return PATTERNS.register(cls)
def get_indicator(name: str): return INDICATORS.get(name)
def get_pattern(name: str):   return PATTERNS.get(name)
def list_indicators() -> list: return INDICATORS.list()
def list_patterns() -> list:   return PATTERNS.list()

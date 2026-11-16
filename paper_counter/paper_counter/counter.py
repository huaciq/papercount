from __future__ import annotations

"""Backward-compatible CounterConfig entry point for the recipe engine."""
from typing import Any
import numpy as np
from .config import CounterConfig
from .models import PaperCountResult


class PaperCounter:
    """Adapt existing configuration calls without owning material decisions."""

    def __init__(self, config: CounterConfig | dict[str, Any] | None = None):
        self.config = config if isinstance(config, CounterConfig) else CounterConfig(**(config or {}))

    def process(self, image: np.ndarray) -> PaperCountResult:
        from .recipe import Recipe
        from .engine import DetectionEngine
        # Resolve on every call to preserve historical mutable config behavior.
        return DetectionEngine(Recipe.from_legacy(self.config)).process(image)

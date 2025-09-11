from __future__ import annotations
from abc import ABC, abstractmethod
from ..interfaces import GenerationParams, GenerationResult

class BaseEngine(ABC):
    """Abstract base for engines."""
    @abstractmethod
    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        raise NotImplementedError
